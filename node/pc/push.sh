#!/usr/bin/env bash
# Send the code (and, with --data, a folder of data/) to the node. Run on a Mac or Linux computer, from the repository
# folder:
#     bash node/pc/push.sh
#     bash node/pc/push.sh --data received_batch1
# The same as push.ps1 on Windows: packs the code items (the list in sweepkit.hashing.CODE_ITEMS) into
# <project>_code_<utc>.tar.gz with its SHA-256, copies it with unpack.sh to ~/<project>_incoming on the node in one scp,
# and prints the line to run there and the tree digest. The node's login (user@host) is read from node.local.txt
# beside README.md: one line, kept out of git. PUSH_DRY_RUN=1 packs and prints without copying (for tests).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA=""
if [ "${1:-}" = "--data" ]; then DATA="${2:?usage: push.sh [--data <folder in data/>]}"; fi
PROJECT="$(sed -n 's/.*"project"[[:space:]]*:[[:space:]]*"\([a-z][a-z0-9_]*\)".*/\1/p' "$REPO/project.json" | head -1)"
[ -n "$PROJECT" ] || { echo "project.json has no valid project name"; exit 1; }
if [ "${PUSH_DRY_RUN:-0}" != 1 ]; then
    [ -f "$REPO/node.local.txt" ] || { echo "node.local.txt is missing beside README.md: write the node's login (user@host) in it, one line"; exit 1; }
    NODE="$(head -1 "$REPO/node.local.txt" | tr -d '[:space:]')"
fi
if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/${PROJECT}_push_${STAMP}_XXXX")"
INC="$STAGE/${PROJECT}_incoming"
mkdir -p "$INC"
# this computer's repository folder, so that node/pack.sh can print the whole line that pulls results back here
printf 'posix %s\n' "$REPO" > "$INC/pc_folder.txt"
ITEMS=()
for item in src tests sweeps node env docs templates notebooks README.md START_HERE.md CLAUDE.md TESTING.md LICENSE \
            project.json .gitignore .gitattributes; do
    if [ -e "$REPO/$item" ]; then ITEMS+=("$item"); fi
done
CODE="${PROJECT}_code_${STAMP}.tar.gz"
# COPYFILE_DISABLE keeps a Mac's tar from adding ._ resource files
COPYFILE_DISABLE=1 tar -czf "$INC/$CODE" --exclude=__pycache__ --exclude='*.pyc' -C "$REPO" "${ITEMS[@]}"
(cd "$INC" && $SHA "$CODE" > "$CODE.sha256")
cp "$REPO/node/unpack.sh" "$INC/"
echo "code archive $CODE  sha256 $(cut -d' ' -f1 "$INC/$CODE.sha256")"
DATA_NAME=""
if [ -n "$DATA" ]; then
    case "$DATA" in *[!A-Za-z0-9_.-]*) echo "--data takes the name of a folder in data/"; exit 1 ;; esac
    [ -d "$REPO/data/$DATA" ] || { echo "no folder data/$DATA"; exit 1; }
    DATA_NAME="${PROJECT}_data_${DATA}_${STAMP}.tar"
    COPYFILE_DISABLE=1 tar -cf "$INC/$DATA_NAME" -C "$REPO/data" "$DATA"
    (cd "$INC" && $SHA "$DATA_NAME" > "$DATA_NAME.sha256")
    echo "data archive $DATA_NAME  sha256 $(cut -d' ' -f1 "$INC/$DATA_NAME.sha256")"
fi
if [ "${PUSH_DRY_RUN:-0}" = 1 ]; then
    echo "dry run: nothing copied; the archive is in $INC"
else
    scp -r "$INC" "${NODE}:"
fi
DIGEST="$(bash "$REPO/node/tree_digest.sh" "$REPO" 2>/dev/null || true)"
echo ""
echo "Copied to ~/${PROJECT}_incoming on the node. There, run:"
echo "    bash ~/${PROJECT}_incoming/unpack.sh $CODE"
if [ -n "$DATA_NAME" ]; then echo "    bash ~/${PROJECT}_incoming/unpack.sh $DATA_NAME"; fi
echo "Tree digest of the code sent: $DIGEST"
echo "Local copy of what was sent: $STAGE"
