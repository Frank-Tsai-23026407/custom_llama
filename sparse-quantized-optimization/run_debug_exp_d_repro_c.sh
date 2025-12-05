#!/bin/bash
MODEL_PATH="TinyLlama/TinyLlama_v1.1"
SPARSITY_RATIO=0.5
DEFAULT_BITS=5
HIGH_BITS=5 # Force high bits to 5 to mimic Exp C (all 5-bit, all pruned)
LOG_DIR="/home/frank23026407/custom_llama/sparse-quantized-optimization/log"
mkdir -p $LOG_DIR

echo "Running Debug Exp D (Repro Exp C) - Limit 100"
python /home/frank23026407/custom_llama/sparse-quantized-optimization/task_script_exp_d.py \
    --model_path "$MODEL_PATH" \
    --sparsity_ratio $SPARSITY_RATIO \
    --default_bits $DEFAULT_BITS \
    --high_bits $HIGH_BITS \
    --limit 100 \
    | tee "$LOG_DIR/debug_exp_d_repro_c.log"
