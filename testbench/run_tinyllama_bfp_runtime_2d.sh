#!/usr/bin/env bash
set -euo pipefail

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama-env

cd /home/frank23026407/custom_llama

TS=$(date +%Y%m%d-%H%M%S)
BASE_LOG_DIR="testbench/log/tinyllama_bfp_runtime"
mkdir -p "$BASE_LOG_DIR"

# List of 2D block shapes and mantissa bits (match AWQ settings)
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
    label="bfp_${bh}x${bw}_m${mb}"
    log_dir="$BASE_LOG_DIR/$label"
    mkdir -p "$log_dir"
    log_file="$log_dir/hellaswag.$TS.log"
    echo "Running BFP runtime quantization: block ${bh}x${bw}, mantissa $mb..."
    python quantize_model_script/evaluate_with_runtime_quantization.py \
      --model tinyllama \
      --method bfp \
      --block-height $bh \
      --block-width $bw \
      --mantissa-bits $mb \
      > "$log_file" 2>&1
  done
done

echo "All BFP runtime quantization runs completed. Logs in $BASE_LOG_DIR"
