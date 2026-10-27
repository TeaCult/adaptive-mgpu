#!/usr/bin/env bash
# Optional Linux MPS launcher. Run benchmarks manually on the actual GPUs.
set -euo pipefail
export CUDA_MPS_PIPE_DIRECTORY="/tmp/adaptive-mgpu-mps-${UID}/pipe"
export CUDA_MPS_LOG_DIRECTORY="/tmp/adaptive-mgpu-mps-${UID}/log"

if [[ "${1:-}" == --stop ]]; then
    printf 'quit\n' | nvidia-cuda-mps-control
    exit
fi
if [[ $# -lt 2 || ! $1 =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Usage: bash scripts/with_mps.sh GPU_IDS COMMAND [ARGS...]" >&2
    echo "Example: bash scripts/with_mps.sh 0,1 python -m adaptive_mgpu.server" >&2
    echo "Stop this launcher's daemon: bash scripts/with_mps.sh --stop" >&2
    exit 2
fi
command -v nvidia-smi >/dev/null
command -v nvidia-cuda-mps-control >/dev/null
gpu_ids=$1
shift
uuids=()
IFS=',' read -ra indices <<< "$gpu_ids"
for index in "${indices[@]}"; do
    uuid=$(nvidia-smi -i "$index" --query-gpu=uuid --format=csv,noheader)
    [[ $uuid == GPU-* ]] || { echo "Could not resolve GPU $index" >&2; exit 1; }
    uuids+=("$uuid")
done
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${uuids[*]}")
percent=${MPS_THREAD_PERCENTAGE:-100}
if [[ ! $percent =~ ^[0-9]+$ ]] || (( percent < 1 || percent > 100 )); then
    echo "MPS_THREAD_PERCENTAGE must be between 1 and 100" >&2
    exit 2
fi
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$percent
umask 077
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
selection="${CUDA_MPS_PIPE_DIRECTORY}/selected-gpus"
if printf 'get_server_list\n' | nvidia-cuda-mps-control >/dev/null 2>&1; then
    if [[ ! -f $selection || $(cat "$selection") != "$CUDA_VISIBLE_DEVICES" ]]; then
        echo "Existing MPS daemon uses another GPU selection. Stop it before changing GPUs." >&2
        exit 1
    fi
else
    nvidia-cuda-mps-control -d
    printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$selection"
fi
echo "MPS GPUs: $CUDA_VISIBLE_DEVICES; client thread ceiling: ${percent}%" >&2
exec "$@"
