#!/usr/bin/env bash
# Build an environment on a Linux or Mac computer, from the repository folder:
#     bash env/setup.sh main core torch unsupervised
#     TORCH_INDEX=cu126 bash env/setup.sh main core torch        (Linux with a GPU older than Turing)
#     bash env/setup.sh automl core automl
# The first argument names the environment (.venv-<environment> in the repository folder, ignored by git), the others
# the profiles installed into it, in order, each from env/requirements-<profile>.txt with exact pins. docs/PROFILES.md
# says which profiles a problem needs. Refuses if the environment exists (move it aside to rebuild). The exact
# versions installed go to artifacts/env/<environment>_<utc>.txt.
# PYTHON names the interpreter (python3.12 or python3.13; the first found is used otherwise). TORCH_INDEX (cu130,
# cu126 or cpu) installs torch (and torchvision with the pretrained profile) from the PyTorch index instead of PyPI.
# The environment's Python finds the package without PYTHONPATH: a .pth file in it names this repository's src/
# (move the repository and that file needs the new path). On the shared node use node/setup_env.sh, which puts
# environments in ~/venvs/<project>-<environment>.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENVIRONMENT="${1:?usage: setup.sh <environment> <profile> [<profile> ...]}"
shift
[ "$#" -ge 1 ] || { echo "usage: setup.sh <environment> <profile> [<profile> ...]"; exit 1; }
VENV="${SWEEPKIT_VENV:-$REPO/.venv-$ENVIRONMENT}"
if [ -z "${PYTHON:-}" ]; then
    for candidate in python3.12 python3.13; do
        if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
    done
fi
[ -n "${PYTHON:-}" ] || { echo "no python3.12 or python3.13 found; install one, or set PYTHON"; exit 1; }
for profile in "$@"; do
    test -f "$REPO/env/requirements-$profile.txt" || { echo "no env/requirements-$profile.txt"; exit 1; }
done
if [ -e "$VENV" ]; then echo "REFUSING: $VENV exists. Move it aside first if you mean to rebuild it."; exit 1; fi
echo "building $VENV with $("$PYTHON" --version) from profiles: $*"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
# the environment's Python finds the package in src/ without PYTHONPATH (a .pth file naming this repository's src)
# (written by Python in the encoding it reads .pth files with: the locale's before 3.13, UTF-8 from 3.13)
"$VENV/bin/python" -c 'import sys, sysconfig, pathlib; pathlib.Path(sysconfig.get_paths()["purelib"], "sweepkit_src.pth").write_text(sys.argv[1] + chr(10), encoding="utf-8" if sys.version_info >= (3, 13) else "locale")' "$REPO/src"
for profile in "$@"; do
    echo "--- profile $profile"
    if [ "$profile" = torch ] && [ -n "${TORCH_INDEX:-}" ]; then
        "$VENV/bin/python" -m pip install --index-url "https://download.pytorch.org/whl/$TORCH_INDEX" \
            "$(grep -E '^torch==' "$REPO/env/requirements-torch.txt")"
    elif [ "$profile" = pretrained ] && [ -n "${TORCH_INDEX:-}" ]; then
        "$VENV/bin/python" -m pip install --index-url "https://download.pytorch.org/whl/$TORCH_INDEX" \
            "$(grep -E '^torchvision==' "$REPO/env/requirements-pretrained.txt")"
        grep -v -E '^torchvision==' "$REPO/env/requirements-pretrained.txt" > "$VENV/requirements-pretrained-rest.txt"
        "$VENV/bin/python" -m pip install -r "$VENV/requirements-pretrained-rest.txt"
    else
        "$VENV/bin/python" -m pip install -r "$REPO/env/requirements-$profile.txt"
    fi
done
mkdir -p "$REPO/artifacts/env"
FREEZE="$REPO/artifacts/env/${ENVIRONMENT}_$(date -u +%Y%m%dT%H%M%SZ).txt"
{ echo "# profiles: $*"; "$VENV/bin/python" -m pip freeze; } > "$FREEZE"
echo "exact versions written to $FREEZE"
echo "next: $VENV/bin/python -m sweepkit.cli env-check --profiles $*"
echo "(or activate it: source $VENV/bin/activate, then python -m sweepkit.cli ...)"
