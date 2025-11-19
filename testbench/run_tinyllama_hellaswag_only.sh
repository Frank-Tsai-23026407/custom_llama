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
MODEL_ROOT="model/tinyllama/TinyLlama_1.1v-2d-comprehensive"

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
  
  # Check if a successful completion log already exists
  if find "$LOG_DIR" -name "hellaswag.*.log" -type f -print0 2>/dev/null | xargs -0 grep -l "Accuracy (acc_norm):" 2>/dev/null | grep -q .; then
    echo "✓ Skipping $MODEL_PATH - successful completion log already exists"
    continue
  fi
  
  log_file="$LOG_DIR/hellaswag.$TS.log"
  echo "Running HellaSwag for $MODEL_PATH..."
  python testbench/task_script_hellaswag.py --backend custom --model_path "$MODEL_PATH" 2>&1 | tee "$log_file"
done

echo "Static (pre-quantized) HellaSwag runs completed. Logs in $BASE_LOG_DIR"

# -----------------------------------------------------------------------------
# Runtime BFP 2D evaluation (no disk save). Matches AWQ 2D shapes (mantissa 4/5)
# -----------------------------------------------------------------------------
RUNTIME_DIR="$BASE_LOG_DIR/runtime_bfp_2d"
mkdir -p "$RUNTIME_DIR"

declare -a BLOCK_SHAPES=(
  "128 1"
  "16 8"
  "1 128"
  "2 64"
  "32 4"
  "4 32"
  "64 2"
  "8 16"
)
declare -a MANTISSA_BITS=(4 5)

for mb in "${MANTISSA_BITS[@]}"; do
  for shape in "${BLOCK_SHAPES[@]}"; do
    bh=$(echo $shape | cut -d' ' -f1)
    bw=$(echo $shape | cut -d' ' -f2)
    label="bfp_runtime_${bh}x${bw}_m${mb}"
    log_dir="$RUNTIME_DIR/$label"
    mkdir -p "$log_dir"
    
    # Check if a successful completion log already exists
    if find "$log_dir" -name "hellaswag.*.log" -type f -print0 2>/dev/null | xargs -0 grep -l "Accuracy (acc_norm):" 2>/dev/null | grep -q .; then
      echo "✓ Skipping RUNTIME BFP 2D block ${bh}x${bw}, mantissa $mb - successful completion log already exists"
      continue
    fi
    
    log_file="$log_dir/hellaswag.$TS.log"
    echo "Running RUNTIME BFP 2D HellaSwag: block ${bh}x${bw}, mantissa $mb..."
    python testbench/task_script_hellaswag_bfp2d_runtime.py \
      --model tinyllama \
      --block-height $bh \
      --block-width $bw \
      --mantissa-bits $mb \
      2>&1 | tee "$log_file"
  done
done

echo "All HellaSwag runs completed (static + runtime). Logs in $BASE_LOG_DIR"
