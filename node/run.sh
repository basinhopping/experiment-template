#!/usr/bin/env bash
# Run any sweepkit command on the node with the project's environment, from any folder:
#     bash ~/<project>/node/run.sh status --sweep screen1 --top 20 --since all
#     ENVIRONMENT=rapids bash ~/<project>/node/run.sh analyze --sweep screen1 --confidence 0.95 ...
# Does the cd and the environment (common.sh) itself, and keeps a copy of the output in logs/<command>_<utc>.log.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$ROOT"
COMMAND="${1:?usage: run.sh <command> [arguments], for example: run.sh status --sweep screen1 --top 20 --since all}"
set +e
py -m sweepkit.cli "$@" 2>&1 | tee -a "$ROOT/logs/${COMMAND}_$(stamp)_$$.log"
code=${PIPESTATUS[0]}
exit "$code"
