#!/usr/bin/env bash
  set -euo pipefail

  MODEL="bartowski/Qwen2.5-1.5B-Instruct-AWQ"
  PORT="${PORT:-8001}"

  vllm serve "$MODEL" \
    --quantization awq \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    "$@"
