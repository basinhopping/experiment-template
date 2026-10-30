#!/usr/bin/env bash
# Run someone's own Python script unchanged on the GPU through cuML's zero-code-change mode, with the log that shows
# where every call ran, in the rapids environment:
#     bash ~/<project>/node/rapids_run.sh cuml <script.py> [script arguments]
#     bash ~/<project>/node/rapids_run.sh cuml+cudf <script.py> [script arguments]
# cuml runs `python -m cuml.accel -v <script>` (scikit-learn, UMAP and HDBSCAN calls on the GPU where cuML supports
# them); cuml+cudf also accelerates pandas: `python -m cudf.pandas -m cuml.accel <script>`, the form cuML documents for
# the two together, with CUML_ACCEL_LOG_LEVEL=info in place of -v, which cuDF's runner refuses. Log level info prints
# for each call whether it ran on the GPU or fell back to the CPU, and why (cuML 26.10 documentation, Logging and
# Profiling). The whole output is kept in logs/rapids_<script>_<utc>.log and copied into
# artifacts/rapids/<script>_<utc>/run.log beside a summary.txt with the counts of calls that ran on the GPU, ran on
# the CPU and fell back to the CPU, and every distinct fallback reason. A fallback is not an error: it is where the
# script did not get the GPU, and the summary is how to see it.
set -uo pipefail
ENVIRONMENT=rapids
source "$(dirname "$0")/common.sh"
MODE="${1:?usage: rapids_run.sh <cuml|cuml+cudf> <script.py> [arguments]}"
SCRIPT="${2:?usage: rapids_run.sh <cuml|cuml+cudf> <script.py> [arguments]}"
shift 2
[ -f "$SCRIPT" ] || { echo "no script $SCRIPT"; exit 1; }
NAME=$(basename "$SCRIPT" .py)
STAMP=$(stamp)
LOG="$ROOT/logs/rapids_${NAME}_${STAMP}.log"
case "$MODE" in
    cuml) set -- -m cuml.accel -v "$SCRIPT" "$@" ;;
    cuml+cudf)
        # cuDF's runner takes the module and the script but not cuml.accel's -v ("unrecognized arguments: -v" on the
        # node, 2026-10-27), so the form cuML documents for the two together is used, with the log level set by
        # cuML's environment variable instead (cuML 26.10, Logging and Profiling: CUML_ACCEL_LOG_LEVEL)
        export CUML_ACCEL_LOG_LEVEL=info
        set -- -m cudf.pandas -m cuml.accel "$SCRIPT" "$@" ;;
    *) echo "the first argument is cuml or cuml+cudf"; exit 1 ;;
esac
echo "$(date -u +%FT%TZ) python $*" | tee "$LOG"
py "$@" 2>&1 | tee -a "$LOG"
code=${PIPESTATUS[0]}
OUT="$ROOT/artifacts/rapids/${NAME}_${STAMP}"
mkdir -p "$OUT"
cp "$LOG" "$OUT/run.log"
{
    echo "script: $SCRIPT ($MODE), exit code $code"
    echo "calls that ran on the GPU: $(grep -c 'ran on GPU' "$LOG")"
    echo "calls that ran on the CPU: $(grep -c 'ran on CPU' "$LOG")"
    echo "fell back to the CPU: $(grep -c 'falling back to CPU' "$LOG")"
    echo "reasons:"
    grep -o 'falling back to CPU.*' "$LOG" | sort | uniq -c | sort -rn
    if ! grep -q 'cuml.accel' "$LOG"; then
        echo "note: cuml.accel printed no line about any call (its log level did not apply), so these counts measure nothing"
    fi
} > "$OUT/summary.txt"
cat "$OUT/summary.txt"
echo "saved $OUT"
exit "$code"
