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

# Stage2: aligned with the pinned image A2 DeepSeek-V4 W8A8 deployment guide.
if [[ "$HELLOHPC_STAGE" != stage2 || "$HELLOHPC_CASE_ID" != dpsk-stage2 || "$HELLOHPC_ASSIGNED_NPU_COUNT" != 8 ]]; then
    echo "Stage2 candidate requires the official eight-device allocation" >&2
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
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Stage2 has eight cgroup-visible devices with logical IDs 0 through 7;
# the evaluator uses the physical ID for health checks and device mounts.
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$HELLOHPC_MODEL_PATH" \
    --served-model-name "$HELLOHPC_MODEL_ID" \
    --host "$host" --port "$port" \
    --tensor-parallel-size 8 \
    --data-parallel-size 1 \
    --enable-expert-parallel \
    --quantization ascend \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --safetensors-load-strategy prefetch \
    --model-loader-extra-config '{"enable_multithread_load":"true","num_threads":16}' \
    --speculative-config '{"num_speculative_tokens":1,"method":"mtp","enforce_eager":true}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --async-scheduling \
    --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":false},"enable_cpu_binding":true,"enable_dsa_cp":true,"multistream_overlap_shared_expert":true}' \
    --dtype bfloat16 \
    --max-model-len 102400 \
    --max-num-seqs 20 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 \
    >"$HELLOHPC_SERVICE_LOG" 2>&1 < /dev/null &
