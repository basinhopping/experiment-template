#!/usr/bin/env bash
# The GPUs no other user is on, one index per line (the reasons for the others go to stderr):
#     bash ~/<project>/node/free_gpus.sh
# Busy means a process of another user (an MPS server does not count) or more than BUSY_MIB (default 2000) MiB of
# memory in use. Exits 1 when no GPU is free.
set -uo pipefail
source "$(dirname "$0")/common.sh"
found=0
for g in $("${NVIDIA_SMI_CMD:-nvidia-smi}" --query-gpu=index --format=csv,noheader | tr -d ' '); do
    busy=$(gpu_busy_reason "$g")
    if [ -z "$busy" ]; then echo "$g"; found=1; else echo "GPU $g busy: $busy" >&2; fi
done
[ "$found" = 1 ]
