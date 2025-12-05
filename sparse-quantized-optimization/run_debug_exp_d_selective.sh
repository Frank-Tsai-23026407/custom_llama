#!/bin/bash
MODEL_PATH="TinyLlama/TinyLlama_v1.1"
SPARSITY_RATIO=0.5
DEFAULT_BITS=5
HIGH_BITS=8
LOG_DIR="/home/frank23026407/custom_llama/sparse-quantized-optimization/log"
mkdir -p $LOG_DIR

echo "Running Debug Exp D (Selective) - Limit 100"
python /home/frank23026407/custom_llama/sparse-quantized-optimization/task_script_exp_d.py \
    --model_path "$MODEL_PATH" \
    --sparsity_ratio $SPARSITY_RATIO \
    --default_bits $DEFAULT_BITS \
    --high_bits $HIGH_BITS \
    --limit 100 \
    | tee "$LOG_DIR/debug_exp_d_selective.log"
