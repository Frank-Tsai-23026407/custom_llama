#!/bin/bash

# Experiment D (Selective): Mixed Precision + Wanda (Skip Sensitive Layers)

MODEL_PATH="TinyLlama/TinyLlama_v1.1"
SPARSITY_RATIO=0.5
DEFAULT_BITS=5
HIGH_BITS=8
LOG_DIR="/home/frank23026407/custom_llama/sparse-quantized-optimization/log"
mkdir -p $LOG_DIR

echo "Running Experiment D (Selective): Mixed Precision + Wanda (Skip Sensitive)"
echo "Model: $MODEL_PATH"
echo "Sparsity: $SPARSITY_RATIO (applied only to default precision layers)"
echo "Bits: Default=$DEFAULT_BITS, High=$HIGH_BITS"

python /home/frank23026407/custom_llama/sparse-quantized-optimization/task_script_exp_d.py \
    --model_path "$MODEL_PATH" \
    --sparsity_ratio $SPARSITY_RATIO \
    --default_bits $DEFAULT_BITS \
    --high_bits $HIGH_BITS \
    | tee "$LOG_DIR/exp_d_mixed_wanda_selective_sparsity_${SPARSITY_RATIO}.log"

echo "Done. Log saved to $LOG_DIR/exp_d_mixed_wanda_selective_sparsity_${SPARSITY_RATIO}.log"
