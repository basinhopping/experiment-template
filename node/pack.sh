#!/usr/bin/env bash
# Pack finished results for the user's computer. Run on the node:
#     bash ~/<project>/node/pack.sh runs/screen1 sweeps/screen1
#     bash ~/<project>/node/pack.sh --after <earlier archive name> runs/screen1
#     bash ~/<project>/node/pack.sh --after <first> --after <second> runs/screen1
# The paths are under artifacts/ (runs/<sweep>, sweeps/<sweep>, analysis, selections, exports/<name>, checks, ...).
# Writes ~/<project>_outgoing/<name>.tar.gz and <name>.tar.gz.sha256, where <name> is printed at the end; the pull
# script on the computer (node/pc/pull.ps1 or pull.sh) fetches and checks them. The last lines are the pull line to
# paste on the computer, whole when the last push recorded the computer's repository folder
# (~/<project>_incoming/pc_folder.txt), otherwise to run in that folder. The archive holds a list of every
# file's SHA-256, which pull checks after unpacking. Refuses while any folder under the given paths is unfinished
# (*.tmp). With several paths the name is the first one's plus "and_<n>_more".
# --after is for a sweep pulled before it finished: the earlier archive (still in ~/<project>_outgoing) lists what
# the computer already has, so only the rest is packed (pull refuses to overwrite). A file an earlier archive holds
# with other content stops it. --after may be given several times.
set -euo pipefail
source "$(dirname "$0")/common.sh"
USAGE="usage: pack.sh [--after <earlier archive name>]... <path under artifacts/> [more paths]"
AFTER=()
NA=0
while [ "${1:-}" = "--after" ]; do
    AFTER+=("${2:?$USAGE}")
    NA=$((NA + 1))
    shift 2
done
[ "$#" -ge 1 ] || { echo "$USAGE"; exit 1; }
cd "$ROOT/artifacts"
for SUB in "$@"; do
    test -d "$SUB" || { echo "no artifacts/$SUB"; exit 1; }
    if [ -n "$(find "$SUB" -type d -name "*.tmp" -print -quit)" ]; then
        echo "REFUSING: unfinished folders (*.tmp) under artifacts/$SUB"; exit 1
    fi
done
mkdir -p "$OUTGOING"
for A in ${AFTER[@]+"${AFTER[@]}"}; do
    if [ ! -f "$OUTGOING/$A.tar.gz" ]; then echo "no earlier archive $OUTGOING/$A.tar.gz"; exit 1; fi
done
LABEL="$(echo "$1" | tr '/' '_')"
if [ "$#" -gt 1 ]; then LABEL="${LABEL}_and_$(($# - 1))_more"; fi
if [ "$NA" -gt 0 ]; then LABEL="${LABEL}_new"; fi
NAME="${PROJECT}_results_${LABEL}_$(stamp)"
STAGE="${OUTGOING:?}/${NAME:?}.stage"
DONE=0
cleanup() {
    rm -rf "${STAGE:?}"
    if [ "$DONE" != 1 ]; then rm -f "${OUTGOING:?}/${NAME:?}.tar.gz" "${OUTGOING:?}/${NAME:?}.tar.gz.sha256"; fi
}
trap cleanup EXIT
mkdir -p "$STAGE/_pulled"
LIST="$STAGE/_pulled/$NAME.files.sha256"
if [ "$NA" -eq 0 ]; then
    find "$@" -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$LIST"
    tar -czf "$OUTGOING/$NAME.tar.gz" -C "$ROOT/artifacts" "$@" -C "$STAGE" _pulled
else
    : > "$STAGE/earlier.sha256"
    for A in "${AFTER[@]}"; do
        tar -xzOf "$OUTGOING/$A.tar.gz" "_pulled/$A.files.sha256" >> "$STAGE/earlier.sha256" \
            || { echo "no list of files in $A.tar.gz"; exit 1; }
    done
    find "$@" -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum > "$STAGE/now.sha256"
    # a sha256sum line is 64 hex digits, two characters, then the path (from column 67)
    awk -v changed="$STAGE/changed.txt" '
        FILENAME == ARGV[1] { earlier[substr($0, 67)] = substr($0, 1, 64); next }
        { path = substr($0, 67)
          if (!(path in earlier)) print
          else if (earlier[path] != substr($0, 1, 64)) print path > changed }' \
        "$STAGE/earlier.sha256" "$STAGE/now.sha256" > "$LIST"
    if [ -s "$STAGE/changed.txt" ]; then
        echo "REFUSING: these files changed since ${AFTER[*]} (the computer has the earlier content):"
        head -20 "$STAGE/changed.txt"; exit 1
    fi
    if [ ! -s "$LIST" ]; then echo "nothing new under $* since ${AFTER[*]}"; exit 1; fi
    cut -c67- "$LIST" > "$STAGE/names.txt"
    tar -czf "$OUTGOING/$NAME.tar.gz" -C "$ROOT/artifacts" -T "$STAGE/names.txt" -C "$STAGE" _pulled
fi
(cd "$OUTGOING" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
DONE=1
echo "packed $(tar -tzf "$OUTGOING/$NAME.tar.gz" | wc -l) entries from $# paths: $NAME"
# The computer's repository folder, as the last push recorded it (pc_folder.txt: "windows <folder>" or
# "posix <folder>"), makes the line that pulls the archive whole, ready to paste there.
KIND=""
FOLDER=""
if [ -f "$INCOMING/pc_folder.txt" ]; then
    read -r KIND FOLDER < "$INCOMING/pc_folder.txt" || true
    FOLDER="${FOLDER%$'\r'}"
fi
if [ "$KIND" = windows ] && [ -n "$FOLDER" ]; then
    echo "paste this on your computer (Command Prompt):"
    echo "cd /d \"$FOLDER\" && powershell -ExecutionPolicy Bypass -File node\\pc\\pull.ps1 $NAME"
elif [ "$KIND" = posix ] && [ -n "$FOLDER" ]; then
    echo "paste this on your computer (a terminal):"
    echo "cd '${FOLDER//\'/\'\\\'\'}' && bash node/pc/pull.sh $NAME"
else
    echo "paste this on your computer, in the repository folder:"
    echo "    Windows:     powershell -ExecutionPolicy Bypass -File node\\pc\\pull.ps1 $NAME"
    echo "    Mac, Linux:  bash node/pc/pull.sh $NAME"
fi
