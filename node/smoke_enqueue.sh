#!/usr/bin/env bash
# Smoke-test, then enqueue, sweep files in the order given. Run on the node, usually inside a round's tmux session:
#     SMOKE_DEVICE=cuda SMOKE_GPUS=2,3 bash ~/<project>/node/smoke_enqueue.sh sweeps/screen1.json sweeps/screen2.json
# A sweep already enqueued (artifacts/sweeps/<sweep_id> exists) is skipped. Each other sweep is smoke-tested
# (run.sh smoke, SMOKE_ROWS rows per role) on SMOKE_DEVICE, cpu or cuda: what the workers will offer, so a job whose
# fit.device is cuda is smoke-tested on a GPU. With cuda, the smoke tests run on the first GPU of SMOKE_GPUS (the GPUs
# the project may use; SMOKE_GPU names one) that no other user is on, and nothing runs when none is free. Every sweep
# is smoke-tested first; they are enqueued only when every one ends with SMOKE OK, all in one enqueue, which checks
# them all before it queues any. A failure is reported with its log, nothing is enqueued, and the script exits 1, so
# a round's chain stops before its workers start and no job of the round is left waiting in the queue. Ends with
# ALL DONE and the number of pending jobs.
set -uo pipefail
source "$(dirname "$0")/common.sh"
cd "$ROOT" || exit 1
SMOKE_DEVICE="${SMOKE_DEVICE:-cpu}"
SMOKE_ROWS="${SMOKE_ROWS:-200}"
[ "$#" -ge 1 ] || { echo "usage: smoke_enqueue.sh <sweep file> [<sweep file> ...]"; exit 1; }
case "$SMOKE_DEVICE" in
    cpu) export CUDA_VISIBLE_DEVICES="" ;;
    cuda)
        CHOICES="${SMOKE_GPUS:-${SMOKE_GPU:-}}"
        [ -n "$CHOICES" ] || { echo "SMOKE_DEVICE=cuda needs SMOKE_GPUS, the GPUs this project may use"; exit 1; }
        FREE=$(bash "$ROOT/node/free_gpus.sh" 2>/dev/null || true)
        PICK=""
        for g in $(echo "$CHOICES" | tr ',' ' '); do
            if printf '%s\n' "$FREE" | grep -qx "$g"; then PICK="$g"; break; fi
        done
        [ -n "$PICK" ] || { echo "no free GPU among $CHOICES for the smoke tests: stop here and paste this"; exit 1; }
        export CUDA_VISIBLE_DEVICES="$PICK"
        echo "smoke tests on GPU $PICK" ;;
    *) echo "SMOKE_DEVICE is cpu or cuda"; exit 1 ;;
esac
missed=0
ready=()
# the devices the workers will take: GPU workers take jobs on cuda or auto, CPU workers jobs on cpu or auto
case "$SMOKE_DEVICE" in cuda) OTHERS="cpu|mps" ;; *) OTHERS="cuda|mps" ;; esac
for f in "$@"; do
    if [ ! -f "$f" ]; then echo "NOT ENQUEUED: $f, no such file"; missed=$((missed + 1)); continue; fi
    # each configuration's fit.device, as check expands the sweep
    if ! expanded=$(py -m sweepkit.cli check --sweep-file "$f" 2>&1); then
        echo "NOT ENQUEUED: $f, check failed: $(echo "$expanded" | tail -1)"; missed=$((missed + 1)); continue
    fi
    if echo "$expanded" | grep -q -E "\"device\": \"($OTHERS)\""; then
        echo "NOT ENQUEUED: $f has jobs on $OTHERS, which workers on $SMOKE_DEVICE do not take (fit.device auto runs anywhere)"
        missed=$((missed + 1)); continue
    fi
    id=$(sed -n 's/.*"sweep_id"[[:space:]]*:[[:space:]]*"\([A-Za-z0-9_-]*\)".*/\1/p' "$f" | head -1)
    if [ -z "$id" ]; then echo "NOT ENQUEUED: $f has no sweep_id"; missed=$((missed + 1)); continue; fi
    if [ -d "artifacts/sweeps/$id" ]; then echo "already enqueued: $id"; continue; fi
    L="$ROOT/logs/smoke_${id}_$(stamp).log"
    py -m sweepkit.cli smoke --sweep-file "$f" --device "$SMOKE_DEVICE" --rows "$SMOKE_ROWS" > "$L" 2>&1
    rc=$?
    grep -E "^sweep |FAILED|^NOTE|^SMOKE" "$L"
    if [ "$rc" = 0 ] && grep -q "^SMOKE OK" "$L"; then
        ready+=("$f")
    else
        echo "NOT ENQUEUED: $id, see $L"
        missed=$((missed + 1))
    fi
done
if [ "$missed" -gt 0 ]; then
    echo "NOT ENQUEUED: $missed of $# sweeps failed their smoke test, so none is enqueued: stop here and paste this"
    exit 1
fi
# one call for every sweep: enqueue checks them all (each sweep_id once, none enqueued before, the sweeps they follow
# recorded) before it records or queues any, so a refusal leaves the queue as it was
if [ "${#ready[@]}" -gt 0 ]; then
    L="$ROOT/logs/enqueue_$(stamp).log"
    py -m sweepkit.cli enqueue --sweep-file "${ready[@]}" > "$L" 2>&1
    rc=$?
    grep -E '"(sweep_id|configs|jobs)"|WARNING|ERROR|Error' "$L"
    if [ "$rc" != 0 ]; then
        echo "NOT ENQUEUED: enqueue refused the sweeps (see $L), so none is enqueued: stop here and paste this"
        exit 1
    fi
fi
echo "ALL DONE $(date -u +%FT%TZ), pending $(ls queue/pending 2>/dev/null | wc -l)"
