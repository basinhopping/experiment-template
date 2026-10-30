#!/usr/bin/env bash
# The GPU schedule of a night (or a weekend): which GPUs get workers, from when to when. Run on the node in its own
# tmux session, which ends by itself after the last step:
#     S="$(date +%s):0,1,2,3 $(date -d 'tomorrow 08:00' +%s):end"
#     tmux new -d -s <project>_schedule "bash ~/<project>/node/schedule.sh 3 $S"
# The first argument is the number of worker processes per GPU; each further argument is a step <unix time>:<what>:
#     <time>:0,1,2,3   from that time, workers run on exactly these GPUs: a GPU in the list without workers gets them
#                      (if no other user is on it), a GPU not in the list gets queue/STOP_gpu<g> (its workers finish
#                      the job they are on, then exit)
#     <time>:none      every GPU's workers stop after their current jobs; the schedule goes on
#     <time>:end       queue/STOP (every worker stops after its current job), the per-GPU stop files removed, and the
#                      schedule exits
# Use the times the people who share the node agreed on (for example nights only on weekdays): the times are
# arguments, never written into this script. A step whose time has passed is skipped when the next step's time has
# passed too, so a late start applies only the current state.
# It refuses to start while Slurm's slurmd is active (Slurm would give the GPUs to batch jobs without seeing these
# workers), never starts a GPU another user is on (common.sh, gpu_busy_reason), and starts nothing while queue/STOP
# exists. Every step is checked before anything starts: each time a whole number of seconds (a time `date -d` could
# not read arrives empty), in order; each GPU one nvidia-smi lists; and the last step, and only the last, an end in
# the future, at most SCHEDULE_MAX_DAYS (default 7) days ahead, so no worker starts without a stop time.
# It also refuses when the environment's Python is missing or MIN_FREE_GB, MAX_FAILS or THREADS is not a number:
# workers started that way would die at once.
# With --check first, it only checks the steps and prints SCHEDULE OK, starting and writing nothing; it then also
# needs the end at least SCHEDULE_MIN_LEAD_MINUTES (default 30) ahead, for the work between the check and the start.
# A round's plan checks its schedule that way before it enqueues anything (swap_and_restart.sh too), so a refusal
# leaves no jobs waiting without workers.
# Nothing else is touched: no other tmux session, no queue file but these stop files. To stop the workers early:
# touch ~/<project>/queue/STOP (each finishes its job, then exits). Ending this session (tmux kill-session) only
# cancels its later steps, its end among them: the workers then run on until the queue is empty. Log:
# logs/schedule_<utc>.txt.
# The workers use the environment ENVIRONMENT names (main by default; ENVIRONMENT=rapids for cuML jobs) and the
# MIN_FREE_GB, MAX_FAILS and THREADS set before it (node/worker.sh has their defaults).
# For tests: TMUX_CMD and NVIDIA_SMI_CMD replace tmux and nvidia-smi.
set -uo pipefail
source "$(dirname "$0")/common.sh"
TMUX_CMD="${TMUX_CMD:-tmux}"
USAGE="usage: schedule.sh [--check] <workers per GPU> <time>:<gpus|none|end> ..."
CHECK=""
if [ "${1:-}" = --check ]; then CHECK=yes; shift; fi
WORKERS="${1:?$USAGE}"
shift
[ "$#" -ge 1 ] || { echo "$USAGE"; exit 1; }
STEPS=("$@")
Q="$ROOT/queue"
if [ -n "$CHECK" ]; then
    LOG=/dev/null
else
    mkdir -p "$Q"
    LOG="$ROOT/logs/schedule_$(stamp).txt"
fi
log() { echo "$(date -u +%FT%TZ) (local $(date '+%a %H:%M %Z')) $*" | tee -a "$LOG"; }
if slurm_active; then log "Slurm's slurmd is active, so Slurm may give these GPUs to batch jobs: not starting"; exit 1; fi
if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then log "REFUSING: '$WORKERS' workers per GPU: a whole number from 1; nothing started"; exit 1; fi
if [ ! -x "$VENV/bin/python" ]; then log "REFUSING: no environment $VENV (ENVIRONMENT=$ENVIRONMENT; node/setup_env.sh builds it); nothing started"; exit 1; fi
if ! [[ "${MIN_FREE_GB:-10}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then log "REFUSING: MIN_FREE_GB='$MIN_FREE_GB': a number of GB, such as 15; nothing started"; exit 1; fi
for setting in "MAX_FAILS=${MAX_FAILS:-5}" "THREADS=$THREADS"; do
    if ! [[ "${setting#*=}" =~ ^[1-9][0-9]*$ ]]; then log "REFUSING: $setting: a whole number from 1; nothing started"; exit 1; fi
done
ALL_GPUS=$("${NVIDIA_SMI_CMD:-nvidia-smi}" --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')

run_on() {   # $1: comma-separated GPUs that should have workers from now
    local wanted=",$1," g busy
    for g in $ALL_GPUS; do
        case "$wanted" in
            *",$g,"*)
                if [ -e "$Q/STOP" ]; then log "GPU $g: queue/STOP exists, so nothing starts"; continue; fi
                if "$TMUX_CMD" has-session -t "${PROJECT}_gpu$g" 2>/dev/null; then log "GPU $g: workers running already"; continue; fi
                busy=$(gpu_busy_reason "$g")
                if [ -n "$busy" ]; then log "GPU $g NOT started: $busy"; continue; fi
                rm -f "${Q:?}/STOP_gpu$g"
                # tmux gives a new session the tmux server's environment, not this script's, so every setting the
                # workers read goes into the command itself (without it they would run in the main environment)
                "$TMUX_CMD" new -d -s "${PROJECT}_gpu$g" "ENVIRONMENT=$ENVIRONMENT MIN_FREE_GB=${MIN_FREE_GB:-10} \
MAX_FAILS=${MAX_FAILS:-5} THREADS=$THREADS bash $ROOT/node/worker.sh gpu $g $WORKERS"
                log "GPU $g started: $WORKERS workers in tmux session ${PROJECT}_gpu$g" ;;
            *)
                if "$TMUX_CMD" has-session -t "${PROJECT}_gpu$g" 2>/dev/null; then
                    touch "$Q/STOP_gpu$g"; log "GPU $g: STOP_gpu$g written, its workers stop after their current jobs"
                fi ;;
        esac
    done
    log "pending jobs: $(ls "$Q/pending" 2>/dev/null | wc -l)"
}

# every step checked before anything starts
previous=0
for i in "${!STEPS[@]}"; do
    when="${STEPS[$i]%%:*}"; what="${STEPS[$i]#*:}"
    case "$when" in ''|*[!0-9]*) log "REFUSING: step '${STEPS[$i]}': the time must be seconds since 1970 (date +%s); nothing started"; exit 1 ;; esac
    case "$what" in
        end|none) ;;
        *)  if ! [[ "$what" =~ ^[0-9]+(,[0-9]+)*$ ]]; then log "REFUSING: step '${STEPS[$i]}': give GPU indices like 0,1,2,3, or none, or end; nothing started"; exit 1; fi
            for g in ${what//,/ }; do
                if ! printf '%s\n' $ALL_GPUS | grep -qx "$g"; then
                    log "REFUSING: step '${STEPS[$i]}': nvidia-smi lists no GPU $g here (it lists: $(echo $ALL_GPUS)); nothing started"; exit 1
                fi
            done ;;
    esac
    if [ "$what" = end ] && [ "$i" -lt $((${#STEPS[@]} - 1)) ]; then log "REFUSING: step '${STEPS[$i]}': an end must be the last step; nothing started"; exit 1; fi
done
last="${STEPS[$((${#STEPS[@]} - 1))]}"
END="${last%%:*}"
if [ "${last#*:}" != end ]; then log "REFUSING: the last step must be <time>:end, the workers' stop time; nothing started"; exit 1; fi
if [ "$END" -le "$(date +%s)" ]; then log "REFUSING: the end, $(date -d "@$END" '+%F %H:%M %Z'), has passed; nothing started"; exit 1; fi
if [ $((END - $(date +%s))) -gt $((${SCHEDULE_MAX_DAYS:-7} * 86400)) ]; then
    log "REFUSING: the end, $(date -d "@$END" '+%F %H:%M %Z'), is more than ${SCHEDULE_MAX_DAYS:-7} days away; nothing started"; exit 1
fi
for step in "${STEPS[@]}"; do
    if [ "${step%%:*}" -lt "$previous" ]; then log "REFUSING: step '$step' comes before the step ahead of it; nothing started"; exit 1; fi
    previous="${step%%:*}"
done
if [ -n "$CHECK" ]; then
    LEAD="${SCHEDULE_MIN_LEAD_MINUTES:-30}"
    [[ "$LEAD" =~ ^[0-9]+$ ]] || { log "REFUSING: SCHEDULE_MIN_LEAD_MINUTES='$LEAD': a whole number of minutes"; exit 1; }
    # a minute's allowance: a plan reads "now" and "now + 30 minutes" in two calls of date
    if [ $((END - $(date +%s))) -lt $((LEAD * 60 - 60)) ]; then
        log "REFUSING: the end, $(date -d "@$END" '+%F %H:%M %Z'), is less than $LEAD minutes away, too close for the work before the workers start; nothing started"; exit 1
    fi
    log "SCHEDULE OK: ${#STEPS[@]} steps; the workers stop at $(date -d "@$END" '+%F %H:%M %Z'). Checked only: nothing started"
    exit 0
fi
log "schedule started for project $PROJECT, $WORKERS workers per GPU, steps: ${STEPS[*]}"
for i in "${!STEPS[@]}"; do
    when="${STEPS[$i]%%:*}"; what="${STEPS[$i]#*:}"
    case "$when" in ''|*[!0-9]*) log "step ${STEPS[$i]}: the time must be seconds since 1970 (date +%s)"; exit 1 ;; esac
    next=""
    if [ $((i + 1)) -lt ${#STEPS[@]} ]; then next="${STEPS[$((i + 1))]%%:*}"; fi
    now=$(date +%s)
    if [ -n "$next" ] && [ "$now" -ge "$next" ]; then log "skipped $what of $(date -u -d "@$when" +%FT%TZ): its time and the next step's have passed"; continue; fi
    if [ "$now" -lt "$when" ]; then sleep $((when - now)); fi
    if slurm_active; then log "Slurm's slurmd became active: writing queue/STOP and ending"; touch "$Q/STOP"; exit 1; fi
    case "$what" in
        end) touch "$Q/STOP"; rm -f "${Q:?}"/STOP_gpu*; log "end: queue/STOP written; every worker stops after its current job"; break ;;
        none) run_on "" ;;
        *) case "$what" in *[!0-9,]*) log "step $what: give GPU indices like 0,1,2,3, or none, or end"; exit 1 ;; esac
           run_on "$what" ;;
    esac
done
log "schedule finished"
