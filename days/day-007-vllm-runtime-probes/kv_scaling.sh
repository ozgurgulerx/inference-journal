#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-microsoft/Phi-3-mini-4k-instruct}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
LENS_STR="${LENS:-512 1024 2048 4096}"
CSV="${CSV:-kv_cache_scaling.csv}"

read -r -a LENS_ARR <<< "$LENS_STR"

echo "max_model_len,gpu_mem_used_mb,delta_mb_from_prev,bytes_per_token_est,notes" > "$CSV"

prev_mem=""
prev_len=""

for L in "${LENS_ARR[@]}"; do
  echo "[*] starting server max-model-len=$L"

  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype auto \
    --max-model-len "$L" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --port "$PORT" \
    > "server_${L}.log" 2>&1 &

  PID=$!

  deadline=$((SECONDS + 180))
  until curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null; do
    if (( SECONDS >= deadline )); then
      echo "[!] server did not become ready for max-model-len=$L" >&2
      kill "$PID" || true
      wait "$PID" || true
      exit 1
    fi
    sleep 1
  done

  mem_used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)

  delta_mb=""
  bytes_per_token=""
  if [[ -n "$prev_mem" ]]; then
    delta_mb=$((mem_used_mb - prev_mem))
    delta_tokens=$((L - prev_len))
    bytes_per_token=$(python3 - <<EOF
import math

delta_mb = int(${delta_mb})
delta_tokens = int(${delta_tokens})
print(int(delta_mb * 1024 * 1024 / max(delta_tokens, 1)))
EOF
)
  fi

  echo "${L},${mem_used_mb},${delta_mb},${bytes_per_token}," >> "$CSV"

  echo "[*] stopping server max-model-len=$L"
  kill "$PID" || true
  wait "$PID" || true
  sleep 2

  prev_mem=$mem_used_mb
  prev_len=$L

done
