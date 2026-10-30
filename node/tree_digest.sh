#!/usr/bin/env bash
# The code-tree digest of a repository folder: one SHA-256 over the code items (the same list as
# sweepkit.hashing.CODE_ITEMS and the push scripts), so a person can check that a copy arrived complete:
#     bash node/tree_digest.sh ~/<project>
# The same number comes from `python -m sweepkit.cli tree-digest --root .` on any machine. LC_ALL=C makes the sort
# order bytewise everywhere (without it, two machines once sorted the same files differently).
set -euo pipefail
cd "${1:?usage: tree_digest.sh <repository folder>}"
ITEMS="src tests sweeps node env docs templates notebooks README.md START_HERE.md CLAUDE.md TESTING.md LICENSE
       project.json .gitignore .gitattributes"
present=()
for item in $ITEMS; do
    if [ -e "$item" ]; then present+=("./$item"); fi
done
if [ "${#present[@]}" -eq 0 ]; then printf '' | sha256sum | cut -c1-64; exit 0; fi
# what tools leave (sweepkit.hashing.TOOL_LEFTOVERS) is not code: a notebook's checkpoints, a file browser's files
find "${present[@]}" \( -name __pycache__ -o -name '*.pyc' -o -name .ipynb_checkpoints -o -name .pytest_cache \
    -o -name .mypy_cache -o -name .ruff_cache -o -name .DS_Store -o -name Thumbs.db -o -name '*.egg-info' \) -prune \
    -o -type f -print0 \
    | LC_ALL=C sort -z | xargs -0 -r sha256sum | cut -c1-16,65- | sha256sum | cut -c1-64
