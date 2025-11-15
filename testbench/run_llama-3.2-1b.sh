#!/usr/bin/env bash
set -euo pipefail

# Optional: ensure your env is active
# conda activate llama-env

# Run from repo root
cd /home/frank23026407/custom_llama

MODEL_PATH="model/llama-3.2-1b/Llama-3.2-1B"
LOG_DIR="testbench/log/llama-3.2-1b"
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOG_DIR"

echo "Running HellaSwag..."
python testbench/tinyllama_my_bfp_hellaswag.py \
  --backend custom \
  --model_path "$MODEL_PATH" \
  | tee "$LOG_DIR/hellaswag.$TS.log"

echo "Running ARC-Challenge..."
python testbench/tinyllama_my_bfp_arc_c.py \
  --model_path "$MODEL_PATH" \
  | tee "$LOG_DIR/arc_challenge.$TS.log"

echo "Running ARC-Easy..."
python testbench/tinyllama_my_bfp_arc_e.py \
  --model_path "$MODEL_PATH" \
  | tee "$LOG_DIR/arc_easy.$TS.log"

echo "Running BoolQ..."
python testbench/tinyllama_my_bfp_boolq.py \
  --model_path "$MODEL_PATH" \
  | tee "$LOG_DIR/boolq.$TS.log"

echo "Running OpenBookQA..."
python testbench/tinyllama_my_bfp_obqa.py \
  --model_path "$MODEL_PATH" \
  | tee "$LOG_DIR/obqa.$TS.log"

echo "Running PIQA..."
python testbench/tinyllama_my_bfp_piqa.py \
  --model_path "$MODEL_PATH" \
  | tee "$LOG_DIR/piqa.$TS.log"

echo "Running WinoGrande..."
python testbench/tinyllama_my_bfp_winogrande.py \
  --model_path "$MODEL_PATH" \
  | tee "$LOG_DIR/winogrande.$TS.log"

echo "All runs completed. Logs in $LOG_DIR"