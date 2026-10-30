#!/usr/bin/env bash
# Workers on one GPU, or on the CPU. Started by schedule.sh, or by hand in a tmux session that ends by itself:
#     tmux new -d -s <project>_gpu0 "bash ~/<project>/node/worker.sh gpu 0 3"
#     tmux new -d -s <project>_cpu  "bash ~/<project>/node/worker.sh cpu 0 8"
# gpu <g> <n>: n worker processes on GPU g (CUDA_VISIBLE_DEVICES=g, so each sees only that card as cuda:0); their IDs
#              are gpu<g>_1 ... gpu<g>_n, and queue/STOP_gpu<g> stops them.
# cpu 0 <n>:   n worker processes on the CPU (no GPU visible to them); IDs cpu_1 ... cpu_n; queue/STOP_cpu stops them.
# Each process takes the jobs its device can run from ~/<project>/queue until nothing is left for it, writes
# logs/worker_<id>_<utc>_<pid>.log, prints JOB DONE, JOB FAILED or JOB SKIPPED for every job, and stops after
# MAX_FAILS failed jobs in a row (a problem that fails every job cannot empty the queue overnight) or when the disk
# has less than MIN_FREE_GB free. queue/STOP stops every worker after its current job. The session ends when its
# processes have finished: that is the normal end.
set -euo pipefail
source "$(dirname "$0")/common.sh"
KIND="${1:?usage: worker.sh <gpu|cpu> <gpu index, 0 for cpu> <processes>}"
INDEX="${2:?usage: worker.sh <gpu|cpu> <gpu index, 0 for cpu> <processes>}"
PROCESSES="${3:?usage: worker.sh <gpu|cpu> <gpu index, 0 for cpu> <processes>}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
MAX_FAILS="${MAX_FAILS:-5}"
case "$KIND" in
    gpu) export CUDA_VISIBLE_DEVICES="$INDEX"; GROUP="gpu$INDEX"; DEVICE=cuda ;;
    cpu) export CUDA_VISIBLE_DEVICES=""; GROUP="cpu"; DEVICE=cpu ;;
    *) echo "worker.sh: the first argument is gpu or cpu"; exit 1 ;;
esac
cd "$ROOT"
STAMP=$(stamp)
for n in $(seq 1 "$PROCESSES"); do
    LOG="$ROOT/logs/worker_${GROUP}_${n}_${STAMP}_$$.log"
    py -m sweepkit.cli worker --worker-id "${GROUP}_${n}" --device "$DEVICE" --min-free-gb "$MIN_FREE_GB" \
        --max-consecutive-failures "$MAX_FAILS" > "$LOG" 2>&1 &
    sleep 5
done
wait
echo "all $GROUP workers finished at $(date -u +%FT%TZ)"
