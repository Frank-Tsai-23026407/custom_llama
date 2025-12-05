#!/bin/bash
MODEL_PATH="TinyLlama/TinyLlama_v1.1"
LOG_DIR="/home/frank23026407/custom_llama/sparse-quantized-optimization/log"
mkdir -p $LOG_DIR

# Run baseline (FP16) on 100 samples
python /home/frank23026407/custom_llama/sparse-quantized-optimization/task_script_exp_b.py \
    --model_path "$MODEL_PATH" \
    --sparsity_ratio 0.0 \
    --limit 100 \
    | tee "$LOG_DIR/debug_baseline_100.log"
