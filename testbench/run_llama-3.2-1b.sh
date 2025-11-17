#!/usr/bin/env bash
set -euo pipefail

# Ensure we run under bash even if invoked via `sh`
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

# Optional: ensure your env is active
# conda activate llama-env

# Run from repo root (keep hardcoded path for this workspace)
cd /home/frank23026407/custom_llama

TS=$(date +%Y%m%d-%H%M%S)
BASE_LOG_DIR="testbench/log/llama-3.2-1b"
mkdir -p "$BASE_LOG_DIR"

# Find all model directories under model/llama-3.2-1b (any configuration)
# e.g., Llama-3.2-1B, Llama-3.2-1B-awq-quantized-fix-precision-b32-m2, etc.
mapfile -d '' -t MODEL_DIRS < <(find model/llama-3.2-1b -mindepth 1 -maxdepth 1 -type d -print0)

if (( ${#MODEL_DIRS[@]} == 0 )); then
  echo "No model directories found under ./model/llama-3.2-1b" >&2
  exit 1
fi

run_task() {
  local label="$1"; shift
  local log_dir="$1"; shift
  
  # Ensure log directory exists
  mkdir -p "$log_dir"
  local log_file="${log_dir}/${label}.${TS}.log"
  
  # Check if task already completed (look for most recent log with accuracy)
  local latest_log=$(find "${log_dir}" -name "${label}.*.log" -type f 2>/dev/null | sort -r | head -1)
  if [[ -n "$latest_log" ]] && grep -q "Accuracy" "$latest_log" 2>/dev/null; then
    echo "[SKIP] ${label} - already completed (found in ${latest_log##*/})"
    return 0
  fi
  
  echo "Running ${label}..."
  # Use tee while preserving exit code of python
  "$@" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}
  if (( rc != 0 )); then
    echo "Warning: ${label} failed with exit code ${rc}" >&2
  fi
}

echo "Discovered ${#MODEL_DIRS[@]} Llama-3.2-1B model(s)."
for MODEL_PATH in "${MODEL_DIRS[@]}"; do
  # Build a readable, unique log subfolder based on the model path
  rel_path=${MODEL_PATH#model/}
  safe_name=$(echo "$rel_path" | tr '/ ' '__')
  LOG_DIR="${BASE_LOG_DIR}/${safe_name}"
  mkdir -p "$LOG_DIR"

  printf "\n=== Evaluating model: %s ===\n" "${MODEL_PATH}"

  run_task "hellaswag"     "$LOG_DIR" python testbench/task_script_hellaswag.py  --backend custom --model_path "$MODEL_PATH"
  run_task "arc_challenge" "$LOG_DIR" python testbench/task_script_arc_c.py      --model_path "$MODEL_PATH"
  run_task "arc_easy"      "$LOG_DIR" python testbench/task_script_arc_e.py      --model_path "$MODEL_PATH"
  run_task "boolq"         "$LOG_DIR" python testbench/task_script_boolq.py      --model_path "$MODEL_PATH"
  run_task "obqa"          "$LOG_DIR" python testbench/task_script_obqa.py       --model_path "$MODEL_PATH"
  run_task "piqa"          "$LOG_DIR" python testbench/task_script_piqa.py       --model_path "$MODEL_PATH"
  run_task "winogrande"    "$LOG_DIR" python testbench/task_script_winogrande.py --model_path "$MODEL_PATH"

  echo "Completed evaluations for: ${MODEL_PATH}. Logs in ${LOG_DIR}"
done

echo "All runs completed. Logs in ${BASE_LOG_DIR}"