#!/usr/bin/env bash
set -euo pipefail


# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama-env

cd /home/frank23026407/custom_llama

TS=$(date +%Y%m%d-%H%M%S)
BASE_LOG_DIR="testbench/log/tinyllama_hellaswag_only"
mkdir -p "$BASE_LOG_DIR"

# Only search block shape configs under TinyLlama_1.1v-2d-comprehensive
# MODEL_ROOT="model/tinyllama/TinyLlama_1.1v-2d-comprehensive"
MODEL_ROOT="model/tinyllama/TinyLlama_1.1v-quantized"

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
  log_file="$LOG_DIR/hellaswag.$TS.log"
  echo "Running HellaSwag for $MODEL_PATH..."
  python testbench/task_script_hellaswag.py --backend custom --model_path "$MODEL_PATH" 2>&1 | tee "$log_file"
done

echo "All HellaSwag runs completed. Logs in $BASE_LOG_DIR"
