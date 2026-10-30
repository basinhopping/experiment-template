#!/usr/bin/env bash
# The testing matrix of TESTING.md for the environments named, on one free GPU, within a time limit, in a tmux session
# that ends by itself:
#     tmux new -d -s <project>_testall "bash ~/<project>/node/test_all.sh <gpu index> <hours> main rapids"
# <hours> bounds the whole run (a number, for example 3 or 1.5). It prints the time it will end by, runs each step
# under `timeout` with the time left, and once the time is up records the steps it did not run instead of running
# them. Agree the window with the person first (a weekend, or one GPU on a chosen evening) and end it before the
# node's own cutoff.
# The environments are main, rapids and automl, each built first by node/setup_env.sh (~/venvs/<project>-<name>);
# a named one that is not built is recorded and skipped. For each one, with only that GPU visible, it runs:
#   env-check   the installed versions against the pins of the profiles it was built from, and the devices seen
#   tests       every unit test, with SWEEPKIT_REQUIRE naming the environment's packages and the GPU, so a missing
#               package or an invisible GPU fails instead of skipping
#   agreement   the tests that compare the CPU and GPU versions of a method, one line each (where cuML, cuGraph or
#               XGBoost is installed)
#   smoke cuda  smoke-all of every component, offering the GPU (each run records the device that ran)
#   smoke cpu   smoke-all of every component on the CPU
# and, in main, the large-data test at LARGE_ROWS rows (default 1000000), which prints each step's time and peak
# memory. A step that fails or runs out of time does not stop the others.
#
# Disk: it keeps at least MIN_FREE_GB (default 10) free in the home folder for everything else that runs from it.
# It refuses to start unless MIN_FREE_GB plus 2 GB (its own files at a million rows) are free, checks again before
# each step, and records the steps it did not run once the free space is below MIN_FREE_GB. Each environment's size
# (du -sh) goes into summary.txt, so the sizes TESTING.md estimates can be replaced by measured ones.
#
# Everything goes to artifacts/test_all/<utc>/: one log per step, summary.csv (step, environment, status, exit code,
# seconds, last line of its log; status is ok, failed, timed out, not run: time, not run: disk, or not built) and
# summary.txt. The folder is <utc>.tmp until the end and is renamed then, so it can be packed only once complete:
# bash ~/<project>/node/pack.sh test_all/<utc>. Temporary files go to a folder inside it (TMPDIR), never to a folder
# other users share, and are removed at the end. Its own output, refusals included, is also kept in
# logs/test_all_<utc>.log, so it can be read after the session ends. It refuses to start when the GPU is in use by
# another user (node/free_gpus.sh) or when Slurm's slurmd is active. SWEEPKIT_NETWORK=1 before the command lets the
# tests that download a small pretrained model run (into ~/cache/huggingface); without it they skip. It ends with
# TEST-ALL DONE and the counts. MIN_STEP_SECONDS (default 60) is the least time a step is started with.
set -uo pipefail
source "$(dirname "$0")/common.sh"
LOG="$ROOT/logs/test_all_$(stamp).log"
exec > >(tee -a "$LOG") 2>&1
USAGE="usage: test_all.sh <gpu index> <hours> <environment> [<environment> ...]; environments: main, rapids, automl"
GPU="${1:?$USAGE}"
HOURS="${2:?$USAGE}"
shift 2
[ "$#" -ge 1 ] || { echo "$USAGE"; exit 1; }
case "$GPU" in *[!0-9]*) echo "the GPU index is a number, for example 0"; exit 1 ;; esac
case "$HOURS" in ''|.|*[!0-9.]*|*.*.*) echo "the hours are a number, for example 3 or 1.5"; exit 1 ;; esac
for environment in "$@"; do
    case "$environment" in main|rapids|automl) ;; *) echo "unknown environment $environment: $USAGE"; exit 1 ;; esac
done
ALLOWED=$(awk -v h="$HOURS" 'BEGIN { printf "%d", h * 3600 }')
LARGE_ROWS="${LARGE_ROWS:-1000000}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
MIN_STEP_SECONDS="${MIN_STEP_SECONDS:-60}"
OWN_GB=2
if [ "$ALLOWED" -lt "$MIN_STEP_SECONDS" ]; then echo "$HOURS hours leave no time for a step"; exit 1; fi
if slurm_active; then echo "Slurm's slurmd is active: not starting"; exit 1; fi
busy=$(gpu_busy_reason "$GPU")
if [ -n "$busy" ]; then echo "GPU $GPU is not free ($busy): stop here and paste this"; exit 1; fi
bash "$ROOT/node/disk_check.sh" "$((MIN_FREE_GB + OWN_GB))" || exit 1
START=$(date +%s)
DEADLINE=$((START + ALLOWED))
export CUDA_VISIBLE_DEVICES="$GPU"
STAMP=$(stamp)
while [ -e "$ROOT/artifacts/test_all/$STAMP" ] || [ -e "$ROOT/artifacts/test_all/$STAMP.tmp" ]; do
    sleep 1; STAMP=$(stamp)                     # one folder per run, even for two runs started in the same second
done
FINAL="$ROOT/artifacts/test_all/$STAMP"
OUT="$FINAL.tmp"
mkdir -p "$OUT/tmp"
export TMPDIR="$OUT/tmp"
cd "$ROOT" || exit 1
echo "step,environment,status,exit_code,seconds,last_line" > "$OUT/summary.csv"
{
    echo "test_all on GPU $GPU for $*, started $(date -d "@$START" '+%F %H:%M %Z'), ends by" \
         "$(date -d "@$DEADLINE" '+%F %H:%M %Z') ($(date -u -d "@$DEADLINE" +%FT%TZ))"
    echo "keeps $MIN_FREE_GB GB free in the home folder; large-data test at $LARGE_ROWS rows"
    echo "code: $(tree_digest)"
    "${NVIDIA_SMI_CMD:-nvidia-smi}" --query-gpu=index,name,driver_version,memory.total --format=csv,noheader \
        -i "$GPU" 2>/dev/null
} | tee "$OUT/summary.txt"
n=0
declare -A count=([ok]=0 [failed]=0 [timed_out]=0 [not_run]=0)
not_built=""

free_gb() { df -Pk "$HOME" | awk 'NR == 2 {printf "%d", $4 / 1024 / 1024}'; }

record() {
    # record <name> <environment> <status> <exit code> <seconds> <last line>
    echo "$1,$2,$3,$4,$5,$(echo "$6" | tr ',' ';' | cut -c1-200)" >> "$OUT/summary.csv"
}

step() {
    # step <name> <environment> <command ...>: run it with the time left, log it, record it
    local name="$1" environment="$2" started rc last log left status
    shift 2
    n=$((n + 1))
    log="$OUT/$(printf '%02d' "$n")_${name}.log"
    left=$((DEADLINE - $(date +%s)))
    if [ "$left" -lt "$MIN_STEP_SECONDS" ]; then
        record "$name" "$environment" "not run: time" "" 0 "the time limit was reached"
        echo "step $n: $name ($environment) not run: the time limit was reached" | tee -a "$OUT/summary.txt"
        count[not_run]=$((count[not_run] + 1))
        return
    fi
    if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then
        record "$name" "$environment" "not run: disk" "" 0 "$(free_gb) GB free; MIN_FREE_GB is $MIN_FREE_GB"
        echo "step $n: $name ($environment) not run: $(free_gb) GB free, below $MIN_FREE_GB" | tee -a "$OUT/summary.txt"
        count[not_run]=$((count[not_run] + 1))
        return
    fi
    echo "$(date -u +%FT%TZ) step $n: $name ($environment), at most ${left} s" | tee -a "$OUT/summary.txt"
    started=$(date +%s)
    timeout --kill-after=60 "$left" "$@" > "$log" 2>&1
    rc=$?
    case "$rc" in
        0) status=ok ;;
        124|137) status="timed out" ;;
        *) status=failed ;;
    esac
    last=$(grep -v '^[[:space:]]*$' "$log" | tail -1)
    record "$name" "$environment" "$status" "$rc" "$(($(date +%s) - started))" "$last"
    echo "    $status (exit $rc): $(echo "$last" | cut -c1-200)" | tee -a "$OUT/summary.txt"
    case "$status" in ok) count[ok]=$((count[ok] + 1)) ;; failed) count[failed]=$((count[failed] + 1)) ;;
        *) count[timed_out]=$((count[timed_out] + 1)) ;; esac
}

profiles_of() {
    # the profiles an environment was built from: the first line of its newest freeze file from setup_env.sh
    local file
    file=$(ls -1 "$ROOT/artifacts/env/${1}_"*.txt 2>/dev/null | sort | tail -1)
    [ -n "$file" ] && sed -n '1s/^# profiles: //p' "$file"
}

for environment in "$@"; do
    venv="$HOME/venvs/${PROJECT}-${environment}"
    python="$venv/bin/python"
    if [ ! -x "$python" ]; then
        echo "environment $environment: not built (node/setup_env.sh builds it), skipped" | tee -a "$OUT/summary.txt"
        record "environment" "$environment" "not built" "" 0 "no $python"
        not_built="$not_built $environment"
        continue
    fi
    profiles=$(profiles_of "$environment")
    if [ -z "$profiles" ]; then
        echo "environment $environment: no record of its profiles in artifacts/env/, skipped" | tee -a "$OUT/summary.txt"
        record "environment" "$environment" "not built" "" 0 "no artifacts/env/${environment}_<utc>.txt"
        not_built="$not_built $environment"
        continue
    fi
    size=$(timeout 300 du -sh "$venv" 2>/dev/null | cut -f1)
    # shellcheck disable=SC2086
    require="$("$python" -m sweepkit.cli required-modules --profiles $profiles),gpu"
    echo "environment $environment: profiles $profiles; ${size:-size not measured} in $venv;" \
         "SWEEPKIT_REQUIRE=$require" | tee -a "$OUT/summary.txt"
    # shellcheck disable=SC2086
    step "env_check_$environment" "$environment" "$python" -m sweepkit.cli env-check --profiles $profiles
    step "tests_$environment" "$environment" env SWEEPKIT_REQUIRE="$require" \
        "$python" -m unittest discover -s tests -t .
    if "$python" -c "import importlib.util as u, sys; sys.exit(0 if any(u.find_spec(m) for m in ('cuml', 'cugraph', 'xgboost')) else 1)"; then
        step "agreement_$environment" "$environment" "$python" -m unittest -v tests.test_gpu_agreement \
            tests.test_cluster_models.TestGraphOnTheGPU tests.test_embed_models.TestCuML \
            tests.test_cluster_models.TestPackagesFromExtrasAndSingleCell
    fi
    step "smoke_cuda_$environment" "$environment" "$python" -m sweepkit.cli smoke-all --profile all --device cuda \
        --rows 100
    step "smoke_cpu_$environment" "$environment" "$python" -m sweepkit.cli smoke-all --profile all --device cpu \
        --rows 100
    if [ "$environment" = main ]; then
        step "large_data_$environment" "$environment" env SWEEPKIT_LARGE_ROWS="$LARGE_ROWS" \
            "$python" -m unittest -v tests.test_large_data
    fi
done

# the smoke-all tables this run wrote, beside the logs
for log in "$OUT"/*_smoke_*.log; do
    [ -f "$log" ] || continue
    folder=$(sed -n 's/.*saved \(.*smoke_all_[^ ]*\)$/\1/p' "$log" | tail -1)
    if [ -n "$folder" ] && [ -f "$folder/table.csv" ]; then
        cp "$folder/table.csv" "${log%.log}_table.csv"
    fi
done
rm -rf "$OUT/tmp"
echo "$(date -u +%FT%TZ) TEST-ALL DONE: $n steps: ${count[ok]} ok, ${count[failed]} failed, ${count[timed_out]}" \
     "timed out, ${count[not_run]} not run${not_built:+; not built:$not_built}; $(free_gb) GB free" \
    | tee -a "$OUT/summary.txt"
mv -T "$OUT" "$FINAL"
echo "results: $FINAL (pack with: bash $ROOT/node/pack.sh test_all/$STAMP)"
