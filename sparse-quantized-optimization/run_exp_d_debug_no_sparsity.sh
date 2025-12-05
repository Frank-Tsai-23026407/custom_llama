#!/bin/bash

# Debug Experiment D: Mixed Precision (bit5/bit8) WITHOUT Pruning

MODEL_PATH="TinyLlama/TinyLlama_v1.1"
SPARSITY_RATIO=0.0
DEFAULT_BITS=5
HIGH_BITS=8
LOG_DIR="/home/frank23026407/custom_llama/sparse-quantized-optimization/log"
mkdir -p $LOG_DIR

echo "Running Debug Experiment D: Mixed Precision NO PRUNING"
echo "Model: $MODEL_PATH"
echo "Sparsity: $SPARSITY_RATIO"
echo "Bits: Default=$DEFAULT_BITS, High=$HIGH_BITS"

python /home/frank23026407/custom_llama/sparse-quantized-optimization/task_script_exp_d.py \
    --model_path "$MODEL_PATH" \
    --sparsity_ratio $SPARSITY_RATIO \
    --default_bits $DEFAULT_BITS \
    --high_bits $HIGH_BITS \
    --limit 100 \
    | tee "$LOG_DIR/debug_exp_d_no_sparsity.log"

echo "Done. Log saved to $LOG_DIR/debug_exp_d_no_sparsity.log"
