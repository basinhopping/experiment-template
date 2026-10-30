#!/usr/bin/env bash
# Fetch results packed on the node by node/pack.sh, check them, and unpack them into artifacts/. Run on a Mac or Linux
# computer, from the repository folder, with the name pack.sh printed:
#     bash node/pc/pull.sh <project>_results_<...>
# Checks the archive's SHA-256, refuses to overwrite anything already in artifacts/, unpacks, then checks every file
# against the list of hashes made on the node. The node's login is read from node.local.txt beside README.md.
# PULL_FROM=<folder> takes the archive from a local folder instead of the node (for tests).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
NAME="${1:?usage: pull.sh <archive name pack.sh printed>}"
PROJECT="$(sed -n 's/.*"project"[[:space:]]*:[[:space:]]*"\([a-z][a-z0-9_]*\)".*/\1/p' "$REPO/project.json" | head -1)"
case "$NAME" in "${PROJECT}_results_"*) ;; *) echo "$NAME is not an archive name of project $PROJECT"; exit 1 ;; esac
case "$NAME" in *[!A-Za-z0-9_.-]*) echo "unexpected characters in $NAME"; exit 1 ;; esac
if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi
ART="$REPO/artifacts"
IN="$ART/_incoming"
mkdir -p "$IN"
if [ -n "${PULL_FROM:-}" ]; then
    cp "$PULL_FROM/$NAME.tar.gz" "$PULL_FROM/$NAME.tar.gz.sha256" "$IN/"
else
    [ -f "$REPO/node.local.txt" ] || { echo "node.local.txt is missing beside README.md"; exit 1; }
    NODE="$(head -1 "$REPO/node.local.txt" | tr -d '[:space:]')"
    scp "${NODE}:${PROJECT}_outgoing/$NAME.tar.gz*" "$IN/"
fi
[ -f "$IN/$NAME.tar.gz.sha256" ] || { echo "the archive's .sha256 did not arrive; run the same command again"; exit 1; }
EXPECTED="$(cut -d' ' -f1 "$IN/$NAME.tar.gz.sha256")"
ACTUAL="$($SHA "$IN/$NAME.tar.gz" | cut -d' ' -f1)"
[ "$EXPECTED" = "$ACTUAL" ] || { echo "archive SHA-256 mismatch: expected $EXPECTED, got $ACTUAL"; exit 1; }
while IFS= read -r entry; do
    case "$entry" in */) continue ;; esac
    if [ -e "$ART/$entry" ]; then echo "REFUSING: artifacts/$entry already exists"; exit 1; fi
done < <(tar -tzf "$IN/$NAME.tar.gz")
tar -xzf "$IN/$NAME.tar.gz" -C "$ART"
n=0
while IFS= read -r line; do
    [ -n "$line" ] || continue
    hash="${line%% *}"
    path="${line:66}"
    got="$($SHA "$ART/$path" | cut -d' ' -f1)"
    [ "$got" = "$hash" ] || { echo "hash mismatch after unpacking: $path"; exit 1; }
    n=$((n + 1))
done < "$ART/_pulled/$NAME.files.sha256"
echo "pulled $NAME: $n files unpacked into artifacts/ and checked against the node's hashes"
