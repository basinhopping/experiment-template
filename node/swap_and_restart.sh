#!/usr/bin/env bash
# Install new code under running workers: stop every worker of the project and wait until none is left, install a
# code archive, check its tree digest, smoke-test and enqueue sweeps while the GPUs are free of the project's workers,
# then start the workers again until <end>. For workers started by hand or by a schedule (<project>_schedule); a
# round keeps the code it was prepared with, so to change a round's code, stop the round and prepare a new one.
# Run on the node in a tmux session named <project>_swap, from another tree that holds the new archive's code (for
# example ~/<project>_next: bash ~/<project>_incoming/unpack.sh <archive> ~/<project>_next), since ~/<project>'s
# files are replaced underneath it. tmux gives a new session the tmux server's environment, not the caller's, so the
# workers' environment and disk floor go into the command:
#     tmux new -d -s <project>_swap "ENVIRONMENT=main MIN_FREE_GB=15 bash ~/<project>_next/node/swap_and_restart.sh \
#         <archive> <digest> <workers per GPU> <gpus> $(date -d '2026-10-29 06:30' +%s) [<sweep file> ...]"
# 0. Before anything changes, it refuses unless: ENVIRONMENT and MIN_FREE_GB are set; WAIT_MINUTES,
#    SWAP_POLL_SECONDS and SWAP_LEAD_MINUTES are whole numbers; the archive's SHA-256 holds; the tree it runs from
#    holds <archive> and has <digest>; the code installed now came from an archive still in ~/<project>_incoming, and
#    is still the tree unpack.sh recorded then, so it can be put back as it is; the workers' schedule after the swap
#    (<gpus> from now until <end>, a unix time) passes schedule.sh --check (the environment exists, the floor is a
#    number, the end at most 7 days ahead and, when sweeps are given, at least 30 minutes, for their smoke tests);
#    each sweep's jobs, as the new code's `check` expands them, are ones GPU workers take (fit.device cuda or auto);
#    no CPU workers run (the swap starts GPU workers only); no session of the project runs but workers, a schedule
#    and this one (a round's chain, a test or a build uses the code); and no Python of the project runs outside the
#    workers (a notebook, a test). A refusal adds a line to queue/SWAP_FAILED. When the swap goes ahead, an earlier
#    queue/SWAP_FAILED is moved into logs/.
# 1. queue/STOP: each worker finishes the job it is on, then exits. Waits until no worker session of the project is
#    left and no Python of the project runs, at most WAIT_MINUTES (default 90). On a timeout, or when another session
#    of the project has started meanwhile, it installs nothing and puts nothing back in the queue, and goes to step 4
#    with the workers still running.
# 2. unpack.sh installs the archive (beside this session and a schedule, which run only bash); the tree digest must
#    be <digest>. If it is not, the code installed before is installed again, unless unpack changed nothing, and
#    nothing is enqueued.
# 3. The schedule is checked again (the wait may have taken long): when it would now be refused, nothing is enqueued
#    and the workers stay stopped. Otherwise smoke_enqueue.sh with the sweep files given, on a free GPU of <gpus>:
#    every sweep is enqueued, or none.
# 4. requeue-running when every worker has stopped (a job left in queue/running was cut short), the stop files
#    removed, and the session <project>_schedule replaced by one running schedule.sh with <gpus> from now until
#    <end>: one schedule per project, so a later swap replaces it again. This session then ends. A schedule with
#    several steps is started again by hand after the swap.
# If anything goes wrong, queue/SWAP_FAILED says what, and status prints it first.
# A Python of the project is one whose command names ~/venvs/<project>-..., whose environment has VIRTUAL_ENV there or
# PYTHONPATH in ~/<project>/src, or whose folder is in ~/<project>; one started any other way is not seen.
# Log: logs/swap_<utc>.txt. For tests: SWAP_HOME replaces $HOME and TMUX_CMD replaces tmux; SWAP_POLL_SECONDS
# (default 20) is the wait between checks, SWAP_LEAD_MINUTES (default 30) the lead the smoke tests need.
set -uo pipefail
BASE="${SWAP_HOME:-$HOME}"
TMUX_CMD="${TMUX_CMD:-tmux}"
POLL="${SWAP_POLL_SECONDS:-20}"
WAIT_MINUTES="${WAIT_MINUTES:-90}"
SELF="$(cd "$(dirname "$0")" && pwd)"
USAGE="usage: ENVIRONMENT=<environment> MIN_FREE_GB=<GB> swap_and_restart.sh <archive> <digest> <workers per GPU> <gpus> <end, unix time> [<sweep file> ...]"
ARCHIVE="$(basename "${1:-none}")"
case "$ARCHIVE" in
    [a-z]*_code_*.tar.gz) ;;
    *) echo "REFUSED before any change: '${1:-}' is not a code archive's name, <project>_code_<utc>.tar.gz. $USAGE"; exit 1 ;;
esac
DIGEST="${2:-}"
WORKERS="${3:-}"
GPUS="${4:-}"
END="${5:-}"
shift $(( $# < 5 ? $# : 5 ))
SWEEPS=("$@")
PROJECT="${ARCHIVE%%_code_*}"
ROOT="$BASE/$PROJECT"
IN="$BASE/${PROJECT}_incoming"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [ ! -d "$ROOT" ]; then echo "REFUSED before any change: no $ROOT"; exit 1; fi
mkdir -p "$ROOT/logs" "$ROOT/queue"
LOG="$ROOT/logs/swap_$STAMP.txt"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
failed() { echo "$(date -u +%FT%TZ) $* (log: logs/swap_$STAMP.txt)" >> "$ROOT/queue/SWAP_FAILED"; say "SWAP_FAILED: $*"; }
refused() { failed "REFUSED before any change: $*"; exit 1; }
sessions() { "$TMUX_CMD" ls -F '#{session_name}' 2>/dev/null | grep -E "^${PROJECT}_" || true; }
worker_sessions() { sessions | grep -E "^${PROJECT}_(gpu[0-9]+|cpu)$" || true; }
workers_running() { worker_sessions | grep -c . ; }
# sessions that may keep running while the code is replaced: workers (stopped first), a schedule and this one
others() { sessions | grep -v -E "^${PROJECT}_(gpu[0-9]+|cpu|swap|schedule)$" | paste -sd' ' -; }
project_pythons() {
    local pid exe
    for pid in $(pgrep -u "$(id -u)" 2>/dev/null); do
        exe=$(readlink "/proc/$pid/exe" 2>/dev/null) || continue
        case "${exe##*/}" in python*) ;; *) continue ;; esac
        if tr '\0' '\n' < "/proc/$pid/cmdline" 2>/dev/null | grep -q -F "$BASE/venvs/${PROJECT}-" \
                || tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
                   | grep -q -E "^(VIRTUAL_ENV=$BASE/venvs/${PROJECT}-|PYTHONPATH=(.*:)?$ROOT/src/?(:|$))" \
                || case "$(readlink "/proc/$pid/cwd" 2>/dev/null)/" in "$ROOT"/*) true ;; *) false ;; esac; then
            echo "$pid"
        fi
    done
}
python_running() { project_pythons | grep -c . ; }
worker_tree() {   # every process the worker sessions started
    local roots="" s
    for s in $(worker_sessions); do roots="$roots $("$TMUX_CMD" list-panes -s -t "=$s" -F '#{pane_pid}' 2>/dev/null)"; done
    ps -u "$(id -u)" -o pid=,ppid= | awk -v roots="$roots" '
        BEGIN { n = split(roots, r, " "); for (i = 1; i <= n; i++) keep[r[i]] = 1 }
        { pid[NR] = $1; parent[NR] = $2 }
        END { do { changed = 0; for (i = 1; i <= NR; i++) if (!(pid[i] in keep) && (parent[i] in keep)) { keep[pid[i]] = 1; changed = 1 } } while (changed)
              for (p in keep) print p }'
}
outside_pythons() {
    local tree p
    tree=" $(worker_tree | tr '\n' ' ') "
    for p in $(project_pythons); do
        case "$tree" in *" $p "*) ;; *) echo "$p ($(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | cut -c1-80))" ;; esac
    done
}
schedule_check() { HOME="$BASE" SCHEDULE_MIN_LEAD_MINUTES="$LEAD" bash "$SELF/schedule.sh" --check "$WORKERS" "$(date +%s):$GPUS" "$END:end" 2>&1; }
start_schedule() {
    if "$TMUX_CMD" has-session -t "=${PROJECT}_schedule" 2>/dev/null; then
        "$TMUX_CMD" kill-session -t "=${PROJECT}_schedule"
        say "the running ${PROJECT}_schedule is ended: the workers' schedule after the swap replaces it"
    fi
    "$TMUX_CMD" new -d -s "${PROJECT}_schedule" "ENVIRONMENT=$ENVIRONMENT MIN_FREE_GB=$MIN_FREE_GB MAX_FAILS=${MAX_FAILS:-5} THREADS=${THREADS:-8} SCHEDULE_MAX_DAYS=${SCHEDULE_MAX_DAYS:-7} bash $ROOT/node/schedule.sh $WORKERS $(date +%s):$GPUS $END:end >> $LOG 2>&1"
    local waited=0
    until grep -q "schedule started for project" "$LOG" || [ "$waited" -ge 30 ]; do sleep 1; waited=$((waited + 1)); done
    if ! grep -q "schedule started for project" "$LOG"; then
        touch "$ROOT/queue/STOP"
        failed "the workers' schedule did not start (see this log); queue/STOP is written, so every worker still running stops after its job"
        return 1
    fi
    say "workers on GPUs $GPUS until $(date -d "@$END" '+%F %H:%M %Z'), in tmux session ${PROJECT}_schedule (pending $(ls "$ROOT/queue/pending" 2>/dev/null | wc -l))"
}
restart() {   # $1: yes when every worker has stopped, so a job left in queue/running was cut short and goes back
    if [ "$1" = yes ] && ! (cd "$ROOT" && bash node/run.sh requeue-running --all-workers-stopped yes) >> "$LOG" 2>&1; then
        failed "requeue-running failed (see this log): jobs cut short stay in queue/running"
    fi
    rm -f "${ROOT:?}/queue/STOP" "${ROOT:?}/queue"/STOP_gpu* "${ROOT:?}/queue/STOP_cpu"
    start_schedule
}

# 0. checks before anything changes
if [ -z "$DIGEST" ] || [ -z "$WORKERS" ] || [ -z "$GPUS" ] || [ -z "$END" ]; then
    refused "an argument is missing or empty (an end that date -d could not read arrives empty): $USAGE"
fi
for setting in "WAIT_MINUTES=$WAIT_MINUTES" "SWAP_POLL_SECONDS=$POLL" "SWAP_LEAD_MINUTES=${SWAP_LEAD_MINUTES:-30}"; do
    if ! [[ "${setting#*=}" =~ ^[0-9]+$ ]]; then refused "$setting: a whole number"; fi
done
if [ "$POLL" -lt 1 ]; then refused "SWAP_POLL_SECONDS=$POLL: at least 1"; fi
# the smoke tests and the enqueue come between the schedule's check and its start
LEAD=0
if [ "${#SWEEPS[@]}" -gt 0 ]; then LEAD="${SWAP_LEAD_MINUTES:-30}"; fi
if [ -z "${ENVIRONMENT:-}" ] || [ -z "${MIN_FREE_GB:-}" ]; then
    refused "set ENVIRONMENT and MIN_FREE_GB, the workers' environment and disk floor, in the tmux command ($USAGE)"
fi
export ENVIRONMENT MIN_FREE_GB
if [ ! -f "$IN/$ARCHIVE" ] || [ ! -f "$IN/$ARCHIVE.sha256" ]; then refused "no $IN/$ARCHIVE or its .sha256"; fi
if ! (cd "$IN" && sha256sum -c --quiet "$ARCHIVE.sha256") >> "$LOG" 2>&1; then refused "$ARCHIVE does not match its SHA-256: copy it again"; fi
if [ "$(bash "$SELF/tree_digest.sh" "$SELF/..")" != "$DIGEST" ]; then
    refused "the tree this swap runs from ($SELF/..) is not $DIGEST: run it from a tree that holds the new archive's code (unpack.sh <archive> ~/${PROJECT}_next)"
fi
NEXT=$(tail -1 "$SELF/../CODE_VERSIONS.log" 2>/dev/null | cut -f3)
if [ -n "$NEXT" ] && [ "$NEXT" != "$ARCHIVE" ]; then
    refused "the tree this swap runs from was installed from $NEXT, not $ARCHIVE: name the archive it holds"
fi
PREVIOUS=$(tail -1 "$ROOT/CODE_VERSIONS.log" 2>/dev/null | cut -f3)
RECORDED=$(tail -1 "$ROOT/CODE_VERSIONS.log" 2>/dev/null | cut -f4)
BEFORE=$(bash "$ROOT/node/tree_digest.sh" "$ROOT")
if [ -z "$PREVIOUS" ] || [ ! -f "$IN/$PREVIOUS" ] || [ -z "$RECORDED" ]; then
    refused "the code installed now did not come from an archive still in $IN with its tree digest recorded (CODE_VERSIONS.log's last line: '$(tail -1 "$ROOT/CODE_VERSIONS.log" 2>/dev/null)'), so it could not be put back: install it with unpack.sh first"
fi
if [ "$BEFORE" != "$RECORDED" ]; then
    refused "the code in place is tree $BEFORE, not the $RECORDED that $PREVIOUS installed (a file changed or added on the node?), so it could not be put back as it is: install it again with unpack.sh first"
fi
if ! checked=$(schedule_check); then
    say "$checked"
    refused "the workers' schedule after the swap would be refused: $(echo "$checked" | tail -1)"
fi
for f in ${SWEEPS[@]+"${SWEEPS[@]}"}; do
    file="$f"
    case "$f" in /*) ;; *) file="$SELF/../$f" ;; esac
    if [ ! -f "$file" ]; then refused "no sweep file $f in the new code"; fi
    # each configuration's fit.device, from the sweep expanded by the new code
    if ! expanded=$(cd "$SELF/.." && HOME="$BASE" PYTHONPATH="$SELF/../src" "$BASE/venvs/${PROJECT}-${ENVIRONMENT}/bin/python" -m sweepkit.cli check --sweep-file "$file" 2>&1); then
        say "$expanded"
        refused "check of $f failed: $(echo "$expanded" | tail -1)"
    fi
    if echo "$expanded" | grep -q -E '"device": "(cpu|mps)"'; then
        refused "$f has jobs on cpu or mps, which the GPU workers do not take: set fit.device to auto or cuda"
    fi
done
if [ -n "$(sessions | grep -x "${PROJECT}_cpu")" ]; then
    refused "CPU workers are running (${PROJECT}_cpu): the swap starts GPU workers only, so they would not come back. Stop them first (touch queue/STOP_cpu), and start them again by hand after the swap"
fi
if [ -n "$(others)" ]; then
    refused "sessions of the project that use the code are running: $(others). A swap installs beside workers, a schedule and its own session (named ${PROJECT}_swap) only; a round keeps the code it was prepared with (to change it: touch queue/STOP, end the round's session once its workers are gone, and prepare a new round)"
fi
outside=$(outside_pythons | paste -sd';' -)
if [ -n "$outside" ]; then
    refused "a Python of the project runs outside the workers: $outside. Stop it first"
fi
if [ -f "$ROOT/queue/SWAP_FAILED" ]; then
    mv "$ROOT/queue/SWAP_FAILED" "$ROOT/logs/swap_failed_before_$STAMP.txt"
    say "an earlier queue/SWAP_FAILED is moved to logs/swap_failed_before_$STAMP.txt"
fi
say "swap: $ROOT holds $PREVIOUS (tree $BEFORE); installing $ARCHIVE (tree $DIGEST); environment $ENVIRONMENT, $MIN_FREE_GB GB kept free"

# 1. stop the workers
touch "$ROOT/queue/STOP"
say "queue/STOP written; waiting for every worker to end and no Python of the project to run (at most $WAIT_MINUTES minutes)"
waited=0
while [ "$(workers_running)" != 0 ] || [ "$(python_running)" != 0 ]; do
    if [ "$waited" -ge $((WAIT_MINUTES * 60)) ]; then
        failed "timeout: $(workers_running) worker sessions and $(python_running) Python processes of the project still running after $WAIT_MINUTES minutes; nothing installed, nothing put back in the queue"
        restart no
        exit 1
    fi
    sleep "$POLL"
    waited=$((waited + POLL))
done
say "every worker has stopped (after ${waited}s); jobs in queue/running: $(ls "$ROOT/queue/running" 2>/dev/null | wc -l)"
if [ -n "$(others)" ]; then
    failed "sessions of the project that use the code started during the swap: $(others); nothing installed"
    restart yes
    exit 1
fi

# 2. install, or put the code before back
ok=0
if HOME="$BASE" UNPACK_ALLOW_SESSIONS="swap|schedule" bash "$IN/unpack.sh" "$ARCHIVE" >> "$LOG" 2>&1; then
    NOW=$(bash "$ROOT/node/tree_digest.sh" "$ROOT")
    if [ "$NOW" = "$DIGEST" ]; then ok=1; say "TREE OK $NOW"; else say "TREE DIFFERS $NOW (expected $DIGEST)"; fi
else
    say "unpack.sh FAILED for $ARCHIVE (see this log)"
fi
if [ "$ok" != 1 ]; then
    if [ "$(bash "$ROOT/node/tree_digest.sh" "$ROOT")" = "$BEFORE" ]; then
        failed "the new code was not installed (see this log); the code before ($PREVIOUS) is in place"
    elif HOME="$BASE" UNPACK_ALLOW_SESSIONS="swap|schedule" bash "$IN/unpack.sh" "$PREVIOUS" >> "$LOG" 2>&1 \
            && [ "$(bash "$ROOT/node/tree_digest.sh" "$ROOT")" = "$BEFORE" ]; then
        failed "the new code did not install cleanly; the code before ($PREVIOUS) is installed again"
    else
        failed "REINSTALL FAILED: $ROOT is tree $(bash "$ROOT/node/tree_digest.sh" "$ROOT"); the workers stay stopped; the code moved aside is in $IN/replaced/"
        exit 1
    fi
fi

# 3. the schedule again, then the new sweeps while the GPUs are free of the project's workers (they would make the
# GPUs look busy)
if ! checked=$(schedule_check); then
    say "$checked"
    failed "after the install, the workers' schedule would be refused: $(echo "$checked" | tail -1). Nothing enqueued; the workers stay stopped (queue/STOP)"
    exit 1
fi
sweeps="no sweeps given"
if [ "$ok" = 1 ] && [ "${#SWEEPS[@]}" -gt 0 ]; then
    (cd "$ROOT" && SMOKE_DEVICE=cuda SMOKE_GPUS="$GPUS" bash node/smoke_enqueue.sh "${SWEEPS[@]}") 2>&1 | tee -a "$LOG"
    if [ "${PIPESTATUS[0]}" = 0 ]; then
        sweeps="sweeps enqueued"
    else
        sweeps="SWEEPS NOT ENQUEUED"
        failed "smoke_enqueue.sh: the new sweeps were not enqueued (see this log); the workers start on the queue as it was"
    fi
fi

# 4. the workers again, until the end
say "swap finished: $([ "$ok" = 1 ] && echo "new code, $sweeps" || echo "OLD CODE, nothing enqueued"), pending $(ls "$ROOT/queue/pending" 2>/dev/null | wc -l)"
restart yes
