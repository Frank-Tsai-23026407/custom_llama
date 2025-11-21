#!/usr/bin/env bash
set -euo pipefail

# Function to check if a valid log exists in a directory
has_valid_log() {
  local log_dir="$1"
  
  # Check if directory exists
  if [[ ! -d "$log_dir" ]]; then
    return 1
  fi
  
  # Look for any .log files in the directory
  local log_files=("$log_dir"/*.log)
  
  # Check if any log files exist
  if [[ ! -e "${log_files[0]}" ]]; then
    return 1
  fi
  
  # Check each log file for valid results
  for log_file in "${log_files[@]}"; do
    if grep -q -e "--- HellaSwag Evaluation Results ---" "$log_file" && \
       grep -q -e "Accuracy (acc_norm):" "$log_file"; then
      echo "  ✓ Valid log found: $(basename "$log_file")"
      return 0
    fi
  done
  
  return 1
}

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama-env

cd /home/frank23026407/custom_llama

TS=$(date +%Y%m%d-%H%M%S)
BASE_LOG_DIR="testbench/log/tinyllama_hellaswag_only"
mkdir -p "$BASE_LOG_DIR"

# Only search block shape configs under TinyLlama_1.1v-2d-comprehensive
# MODEL_ROOT="model/tinyllama/TinyLlama_1.1v-2d-comprehensive"
# MODEL_ROOT="model/tinyllama/TinyLlama_1.1v-quantized"
MODEL_ROOT="model/tinyllama/TinyLlama_1.1v-comprehensive"

mapfile -d '' -t MODEL_DIRS < <(find "$MODEL_ROOT" -mindepth 2 -maxdepth 2 -type d -print0)

if (( ${#MODEL_DIRS[@]} == 0 )); then
  echo "No model directories found under $MODEL_ROOT" >&2
  exit 1
fi

echo "Discovered ${#MODEL_DIRS[@]} block shape configs."
for MODEL_PATH in "${MODEL_DIRS[@]}"; do
  rel_path=${MODEL_PATH#model/}
  safe_name=$(echo "$rel_path" | tr '/ ' '__')
  LOG_DIR="$BASE_LOG_DIR/$safe_name"
  mkdir -p "$LOG_DIR"
  
  echo "Checking $MODEL_PATH..."
  
  # Check if valid log already exists
  if has_valid_log "$LOG_DIR"; then
    echo "  → Skipping (valid results already exist)"
    continue
  fi
  
  log_file="$LOG_DIR/hellaswag.$TS.log"
  echo "  → Running HellaSwag evaluation..."
  python testbench/task_script_hellaswag.py --backend custom --model_path "$MODEL_PATH" 2>&1 | tee "$log_file"
done

echo "All HellaSwag runs completed. Logs in $BASE_LOG_DIR"
