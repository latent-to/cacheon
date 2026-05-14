#!/bin/bash
set -e
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
GPU_COUNT=${GPU_COUNT:-1}
[ "$GPU_COUNT" -lt 1 ] && GPU_COUNT=1
exec python -m vllm.entrypoints.openai.api_server \
  --model /models \
  --served-model-name Qwen2.5-72B-Instruct \
  --generation-config vllm \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size "$GPU_COUNT"
