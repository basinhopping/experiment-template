# Settings shared by every node script. Sourced by the others, never run on its own.
#
# Everything is named after the project (project.json "project"), so one person can run two projects from the
# template on the same node without collisions, as long as neither name is the other's followed by _ (the sessions
# of myproject_b would look like myproject's):
#     the code           ~/<project>                 (this repository, installed by unpack.sh)
#     incoming, outgoing ~/<project>_incoming, ~/<project>_outgoing
#     environments       ~/venvs/<project>-<environment>   (main, rapids, automl; node/setup_env.sh builds them)
#     tmux sessions      <project>_gpu<g>, <project>_cpu, <project>_schedule, <project>_<round>
# Nothing here uses /scratch or any shared folder of other users.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$ROOT/project.json" ]; then echo "no project.json in $ROOT"; exit 1; fi
PROJECT="$(sed -n 's/.*"project"[[:space:]]*:[[:space:]]*"\([a-z][a-z0-9_]*\)".*/\1/p' "$ROOT/project.json" | head -1)"
if [ -z "$PROJECT" ]; then echo "project.json has no valid \"project\" name"; exit 1; fi
ENVIRONMENT="${ENVIRONMENT:-main}"
VENV="$HOME/venvs/${PROJECT}-${ENVIRONMENT}"
INCOMING="$HOME/${PROJECT}_incoming"
OUTGOING="$HOME/${PROJECT}_outgoing"

# Model weights downloaded by the pretrained profile are cached in the home folder.
export HF_HOME="${HF_HOME:-$HOME/cache/huggingface}"

# The CUDA MPS fix: another user's MPS daemon can make CUDA unavailable to this user's processes unless these are
# empty before Python starts. sweepkit clears them again on import when SWEEPKIT_CLEAR_MPS is 1.
export CUDA_MPS_PIPE_DIRECTORY=""
export CUDA_MPS_LOG_DIRECTORY=""
export SWEEPKIT_CLEAR_MPS=1

# GPU numbers: nvidia-smi numbers the GPUs in PCI bus order, CUDA by default fastest first. On a node with GPUs of
# different models the two orders can differ, and CUDA_VISIBLE_DEVICES=<g> would then select another card than
# nvidia-smi's GPU g, possibly one another user is on. PCI bus order makes them the same.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Thread caps per process. Several workers share the CPU, so each gets a share; set THREADS to suit the node.
THREADS="${THREADS:-8}"
export OPENBLAS_NUM_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" NUMBA_NUM_THREADS="$THREADS"

export PYTHONPATH="$ROOT/src"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/logs"

py() { "$VENV/bin/python" "$@"; }
stamp() { date -u +%Y%m%dT%H%M%SZ; }
tree_digest() { bash "$ROOT/node/tree_digest.sh" "${1:-$ROOT}"; }
sessions_of_project() { tmux ls -F '#{session_name}' 2>/dev/null | grep -E "^${PROJECT}_" || true; }

slurm_active() {
    # Slurm hands GPUs to batch jobs without seeing workers started by hand, so nothing starts while slurmd runs.
    systemctl is-active --quiet slurmd 2>/dev/null
}

gpu_busy_reason() {
    # Why GPU $1 counts as busy ("" when it is free for this project's workers): a process of another user (an MPS
    # server does not count: it is a daemon the MPS fix bypasses), or more than BUSY_MIB of memory in use, which also
    # catches work run through someone's MPS server, whose clients nvidia-smi may not list.
    local g="$1" uuid pids n=0 pid args mem
    local smi="${NVIDIA_SMI_CMD:-nvidia-smi}" limit="${BUSY_MIB:-2000}"
    uuid=$("$smi" --query-gpu=index,uuid --format=csv,noheader | awk -F', *' -v g="$g" '$1 == g {print $2}')
    if [ -z "$uuid" ]; then echo "GPU $g not found"; return; fi
    pids=$("$smi" --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', *' -v u="$uuid" '$1 == u {print $2}')
    for pid in $pids; do
        if [ "$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')" = "$(id -u)" ]; then continue; fi
        args=$(ps -o args= -p "$pid" 2>/dev/null)
        case "$args" in nvidia-cuda-mps-server*|*/nvidia-cuda-mps-server*) continue ;; esac
        n=$((n + 1))
    done
    mem=$("$smi" --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g="$g" '$1 == g {print $2}')
    if [ "$n" -gt 0 ]; then echo "processes of other users: $n"
    elif [ -z "$mem" ] || [ "$mem" -gt "$limit" ]; then echo "memory in use: ${mem:-unknown} MiB"
    fi
}
