#!/usr/bin/env bash
# The check before any long run on the node, in one environment:
#     bash ~/<project>/node/check.sh main core torch unsupervised
#     bash ~/<project>/node/check.sh rapids rapids
#     CHECK_GPUS=2,3 bash ~/<project>/node/check.sh main core torch unsupervised
# 1. every test, with SWEEPKIT_REQUIRE naming what this environment must have, so a test that would skip for a missing
#    package or GPU fails instead (core: numpy, scipy, sklearn; torch: torch; unsupervised: hdbscan, umap, ...; and
#    gpu whenever a GPU is visible)
# 2. env-check: installed versions against the profiles' pins, and the GPUs this process sees
# The tests that use a GPU run on one GPU no other user is on. A test's device is the first GPU it sees, and on a
# shared node GPU 0 is often someone else's, so CUDA_VISIBLE_DEVICES is set to the first GPU of CHECK_GPUS (the GPUs
# the project may use; every GPU when unset) that free_gpus.sh reports free, unless CUDA_VISIBLE_DEVICES was set
# before this script (empty: no GPU). Stops at the first failure. The log is logs/check_<environment>_<utc>.log.
set -euo pipefail
ENVIRONMENT="${1:?usage: check.sh <environment> <profile> [<profile> ...]}"
shift
[ "$#" -ge 1 ] || { echo "usage: check.sh <environment> <profile> [<profile> ...]"; exit 1; }
export ENVIRONMENT
source "$(dirname "$0")/common.sh"
LOG="$ROOT/logs/check_${ENVIRONMENT}_$(stamp).log"
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"
REQUIRE=$(py -m sweepkit.cli required-modules --profiles "$@")
if command -v "${NVIDIA_SMI_CMD:-nvidia-smi}" >/dev/null 2>&1 && "${NVIDIA_SMI_CMD:-nvidia-smi}" -L >/dev/null 2>&1; then
    if [ -z "${CUDA_VISIBLE_DEVICES+set}" ]; then
        FREE=$(bash "$ROOT/node/free_gpus.sh" 2>/dev/null || true)
        PICK=""
        for g in $(echo "${CHECK_GPUS:-$FREE}" | tr ',' ' '); do
            if printf '%s\n' "$FREE" | grep -qx "$g"; then PICK="$g"; break; fi
        done
        if [ -z "$PICK" ]; then
            echo "no free GPU among ${CHECK_GPUS:-all GPUs} for the tests: stop here and paste this"
            exit 1
        fi
        export CUDA_VISIBLE_DEVICES="$PICK"
    fi
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then REQUIRE="$REQUIRE,gpu"; fi
    echo "tests on GPU ${CUDA_VISIBLE_DEVICES:-none} (CUDA_VISIBLE_DEVICES)"
fi
echo "=== 1/2 tests (SWEEPKIT_REQUIRE=$REQUIRE)"
SWEEPKIT_REQUIRE="$REQUIRE" py -m unittest discover -s tests -t .
echo "=== 2/2 env-check"
py -m sweepkit.cli env-check --profiles "$@"
echo "check: both passed. Log: $LOG"
