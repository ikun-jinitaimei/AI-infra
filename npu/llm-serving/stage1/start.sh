#!/usr/bin/env bash
set -euo pipefail

# Apptainer rejects an externally supplied HOME override. Set it inside the
# container to the writable home directory mounted by the official runner.
export HOME=/home/local
export XDG_CACHE_HOME="$HOME/.cache"
mkdir -p "$XDG_CACHE_HOME"
# Triton's temporary shared libraries can leave NFS silly-rename files during
# dlopen/unlink. Use the container's local shared-memory filesystem instead.
export TMPDIR=/dev/shm

: "${HELLOHPC_MODEL_PATH:?HELLOHPC_MODEL_PATH is set by the platform}"
: "${HELLOHPC_SERVICE_BASE_URL:?HELLOHPC_SERVICE_BASE_URL is set by the platform}"
: "${HELLOHPC_MODEL_ID:?HELLOHPC_MODEL_ID is set by the platform}"
: "${HELLOHPC_CASE_ID:?HELLOHPC_CASE_ID is set by the platform}"
: "${HELLOHPC_STAGE:?HELLOHPC_STAGE is set by the platform}"

: "${HELLOHPC_SERVICE_LOG:?service log path required}"
: "${HELLOHPC_ASSIGNED_NPU_COUNT:?assigned device count required}"

# Initial Stage1 candidate. Stage2 requires a separately validated model stack.
if [[ "$HELLOHPC_STAGE" != stage1 || "$HELLOHPC_CASE_ID" != qwen-performance || "$HELLOHPC_ASSIGNED_NPU_COUNT" != 1 ]]; then
    echo "This untested candidate targets Stage1 only; Stage2 is not implemented yet" >&2
    exit 2
fi

endpoint=$(python3 - <<'PY'
import ipaddress
import os
from urllib.parse import urlsplit
url = urlsplit(os.environ['HELLOHPC_SERVICE_BASE_URL'])
host = url.hostname
if url.scheme != 'http' or not host or not url.port:
    raise ValueError('expected explicit HTTP host and port')
if host != 'localhost' and not ipaddress.ip_address(host).is_loopback:
    raise ValueError('service must listen on loopback')
print(host, url.port)
PY
)
read -r host port <<< "$endpoint"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Stage1 has exactly one cgroup-visible device. Its runtime logical ID is 0;
# the evaluator uses the physical ID for health checks and device mounts.
export ASCEND_RT_VISIBLE_DEVICES=0
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$HELLOHPC_MODEL_PATH" \
    --served-model-name "$HELLOHPC_MODEL_ID" \
    --host "$host" --port "$port" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 \
    >"$HELLOHPC_SERVICE_LOG" 2>&1 < /dev/null &
