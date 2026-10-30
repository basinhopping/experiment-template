#!/usr/bin/env bash
# Run the steps of a plan file in order, in one tmux session that ends by itself (this replaces writing a new chain
# script for every night):
#     tmux new -d -s <project>_<round> "bash ~/<project>/node/chain.sh sweeps/rounds/<round>_prep_<date>/plan.txt"
# round-new writes a round's plan there, under sweeps/, so that it travels to the node with the code and the tree
# digest covers it. A plan file holds one command per line, run from the repository folder with bash; lines starting
# with # and empty lines are skipped, and a carriage return at a line's end (a file saved on Windows) is dropped. Each
# step's output goes to logs/<round>/<n>_<name>.txt, named after the script the step runs (2_smoke_enqueue.txt for
# node/smoke_enqueue.sh), or else its first word (the round is the plan file's folder name without its
# _prep_<date>), and the chain stops at the first step that exits with anything but 0, printing CHAIN STOPPED.
# It ends with CHAIN DONE. It refuses to start while Slurm's slurmd is active.
set -uo pipefail
source "$(dirname "$0")/common.sh"
PLAN="${1:?usage: chain.sh <plan file>}"
cd "$ROOT" || exit 1
[ -f "$PLAN" ] || { echo "no plan file $PLAN"; exit 1; }
if slurm_active; then echo "Slurm's slurmd is active: not starting"; exit 1; fi
ROUND=$(basename "$(dirname "$PLAN")")
ROUND="${ROUND%_prep_*}"
OUT="$ROOT/logs/$ROUND"
mkdir -p "$OUT"
n=0
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in ''|'#'*) continue ;; esac
    n=$((n + 1))
    word=$(echo "$line" | awk '{for (i = 1; i <= NF; i++) if ($i ~ /\.sh$/) {n = split($i, part, "/"); sub(/\.sh$/, "", part[n]); print part[n]; exit} print $1}' | tr -cd 'A-Za-z0-9_.-')
    echo "$(date -u +%FT%TZ) step $n: $line" | tee -a "$OUT/chain.txt"
    bash -c "$line" > "$OUT/${n}_${word}.txt" 2>&1
    rc=$?
    tail -3 "$OUT/${n}_${word}.txt" | tee -a "$OUT/chain.txt"
    if [ "$rc" != 0 ]; then echo "$(date -u +%FT%TZ) CHAIN STOPPED at step $n (exit $rc)" | tee -a "$OUT/chain.txt"; exit "$rc"; fi
done < "$PLAN"
echo "$(date -u +%FT%TZ) CHAIN DONE, $n steps" | tee -a "$OUT/chain.txt"
