#!/usr/bin/env bash
# Build one of the project's Python environments on the node, once:
#     bash ~/<project>/node/setup_env.sh main core torch unsupervised
#     bash ~/<project>/node/setup_env.sh rapids rapids
#     bash ~/<project>/node/setup_env.sh automl automl
# The first argument names the environment (~/venvs/<project>-<environment>), the others the profiles installed into
# it, in order, each from env/requirements-<profile>.txt with exact pins (docs/PROFILES.md says which profiles a
# problem needs and why). Refuses if the environment exists (move it aside to rebuild). Records the exact versions
# in artifacts/env/<environment>_<utc>.txt. Pip keeps no download cache, to spare the home quota. Uses python3.12,
# or python3.13 when there is no python3.12, unless PYTHON names the interpreter (3.12 or 3.13).
set -euo pipefail
ENVIRONMENT="${1:?usage: setup_env.sh <environment> <profile> [<profile> ...]}"
shift
[ "$#" -ge 1 ] || { echo "usage: setup_env.sh <environment> <profile> [<profile> ...]"; exit 1; }
export ENVIRONMENT
source "$(dirname "$0")/common.sh"
LOG="$ROOT/logs/setup_env_${ENVIRONMENT}_$(stamp).log"
exec > >(tee -a "$LOG") 2>&1
if [ -z "${PYTHON:-}" ]; then
    for candidate in python3.12 python3.13; do
        if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
    done
fi
[ -n "${PYTHON:-}" ] || { echo "no python3.12 or python3.13 found; set PYTHON to one"; exit 1; }
echo "setup_env $(date -u +%FT%TZ): environment $VENV, profiles $*, python $PYTHON"
if [ -e "$VENV" ]; then echo "REFUSING: $VENV exists. Move it aside first if you mean to rebuild it."; exit 1; fi
for profile in "$@"; do
    test -f "$ROOT/env/requirements-$profile.txt" || { echo "no env/requirements-$profile.txt"; exit 1; }
done
echo "--- free space before"; df -h "$HOME" | tail -1
mkdir -p "$(dirname "$VENV")" "$HF_HOME"
"$PYTHON" -m venv "$VENV"
py -m pip install --no-cache-dir --upgrade pip
for profile in "$@"; do
    echo "--- profile $profile"
    py -m pip install --no-cache-dir -r "$ROOT/env/requirements-$profile.txt"
done
mkdir -p "$ROOT/artifacts/env"
FREEZE="$ROOT/artifacts/env/${ENVIRONMENT}_$(stamp).txt"
{ echo "# profiles: $*"; echo "# python: $(py -c 'import sys; print(sys.version.split()[0])')"; py -m pip freeze; } > "$FREEZE"
echo "--- exact versions written to $FREEZE"
du -sh "$VENV"
echo "--- free space after"; df -h "$HOME" | tail -1
echo "setup_env: done. Next: bash $ROOT/node/check.sh $ENVIRONMENT $*"
