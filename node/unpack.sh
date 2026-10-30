#!/usr/bin/env bash
# Install an archive the push script sent. Run on the node, from anywhere:
#     bash ~/<project>_incoming/unpack.sh <project>_code_<utc>.tar.gz
#     bash ~/<project>_incoming/unpack.sh <project>_data_<name>_<utc>.tar
#     bash ~/<project>_incoming/unpack.sh <project>_code_<utc>.tar.gz ~/<project>_next
# The project's name is the archive's prefix. Every archive's SHA-256 is checked first.
# Code: the code items (src, tests, sweeps, node, env, docs, templates, notebooks and the top-level files; the same
# list as sweepkit.hashing.CODE_ITEMS) in ~/<project> are replaced wholesale, so a file removed at home cannot survive
# here. data/, artifacts/, queue/, logs/ and reports/ are never touched. The old code is moved aside into
# ~/<project>_incoming/replaced/<utc>_<folder>/, not deleted: on an NFS home a file a running process holds open
# cannot be deleted, while a rename works (copies moved aside earlier are deleted at the next install when nothing
# holds them). If an old item cannot be moved aside, the items already moved go back and nothing changes. Each
# install is recorded in <root>/CODE_VERSIONS.log: the time, the archive's SHA-256, its name and the tree digest. Installing into ~/<project> is refused while any tmux session of
# the project runs: a running worker keeps the code it started with, and a module it imports later would come from
# the new files. swap_and_restart.sh, which stops the workers first and waits until no Python of the project runs,
# sets UNPACK_ALLOW_SESSIONS="swap|schedule": its own session and a schedule run only bash, and the workers a
# schedule starts later run the new code.
# Data: unpacked into ~/<project>/data/<name>/, refused if that folder exists.
# The optional second argument installs code into another folder, for testing new code while workers run.
set -euo pipefail
NAME="${1:?usage: unpack.sh <archive name in the incoming folder> [code folder]}"
case "$NAME" in
    *_code_*.tar.gz) PROJECT="${NAME%%_code_*}" ;;
    *_data_*.tar) PROJECT="${NAME%%_data_*}" ;;
    *) echo "unknown archive name: $NAME (expected <project>_code_<utc>.tar.gz or <project>_data_<name>_<utc>.tar)"; exit 1 ;;
esac
IN="$HOME/${PROJECT}_incoming"
ROOT="${2:-$HOME/$PROJECT}"
ROOT="${ROOT%/}"
ALLOW="${UNPACK_ALLOW_SESSIONS:-}"
busy=$(tmux ls -F '#{session_name}' 2>/dev/null | grep -E "^${PROJECT}_" \
       | { if [ -n "$ALLOW" ]; then grep -v -E "^${PROJECT}_(${ALLOW})\$"; else cat; fi; } || true)
if [ "$ROOT" = "$HOME/$PROJECT" ] && [ -n "$busy" ]; then
    echo "REFUSING: tmux sessions of $PROJECT are running ($(echo $busy)). touch $ROOT/queue/STOP, wait until none is left, then run this again."
    exit 1
fi
cd "$IN"
sha256sum -c "$NAME.sha256"
STAGE="${IN:?}/unpacking_${NAME:?}"
mkdir "$STAGE"
cleanup() { rm -rf "${STAGE:?}"; }
trap cleanup EXIT
case "$NAME" in
*_code_*.tar.gz)
    tar -xzf "$NAME" -C "$STAGE"
    test -f "$STAGE/src/sweepkit/__init__.py" || { echo "archive has no src/sweepkit"; exit 1; }
    mkdir -p "$ROOT"
    ITEMS="src tests sweeps node env docs templates notebooks README.md START_HERE.md CLAUDE.md TESTING.md LICENSE
           project.json .gitignore .gitattributes"
    OLD="$IN/replaced"
    mkdir -p "$OLD"
    for previous in "$OLD"/*; do
        if [ -e "$previous" ]; then
            rm -rf "${previous:?}" 2>/dev/null || echo "note: $previous is still held open somewhere; left for a later install"
        fi
    done
    ASIDE=$(mktemp -d "$OLD/$(date -u +%Y%m%dT%H%M%SZ)_$(basename "$ROOT")_XXXX")
    moved=""
    for item in $ITEMS; do
        if [ -e "$ROOT/$item" ] || [ -L "$ROOT/$item" ]; then
            if mv "$ROOT/$item" "$ASIDE/$item"; then
                moved="$moved $item"
            else
                for back in $moved; do mv "$ASIDE/$back" "$ROOT/$back"; done
                echo "REFUSING: $ROOT/$item could not be moved aside; the code in $ROOT is unchanged"
                exit 1
            fi
        fi
    done
    for item in $ITEMS; do
        if [ -e "$STAGE/$item" ]; then mv "$STAGE/$item" "$ROOT/$item"; fi
    done
    find "$ROOT/node" -name "*.sh" -exec sed -i 's/\r$//' {} +
    left=$(find "$STAGE" -mindepth 1 -maxdepth 1)
    if [ -n "$left" ]; then echo "WARNING, not installed: $left"; fi
    INSTALLED=$(bash "$ROOT/node/tree_digest.sh" "$ROOT")
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$(cut -d' ' -f1 "$NAME.sha256")" "$NAME" "$INSTALLED" >> "$ROOT/CODE_VERSIONS.log"
    echo "code in $ROOT replaced from $NAME; tree digest $INSTALLED"
    ;;
*_data_*.tar)
    DATA_NAME=$(echo "$NAME" | sed -E "s/^${PROJECT}_data_([A-Za-z0-9_.-]+)_[0-9TZ]+\.tar$/\1/")
    DEST="$ROOT/data/$DATA_NAME"
    if [ -e "$DEST" ]; then echo "REFUSING: $DEST exists. Move it aside first."; exit 1; fi
    tar -xf "$NAME" -C "$STAGE"
    test -d "$STAGE/$DATA_NAME" || { echo "archive has no folder $DATA_NAME"; exit 1; }
    mkdir -p "$ROOT/data"
    mv "$STAGE/$DATA_NAME" "$DEST"
    echo "data in $DEST. Check it with profile-data, or verify-data when it came from ingest"
    ;;
esac
