#!/bin/bash

# Experiment C: BFP bit5 + Activation-Aware (Wanda) 50% Pruning

MODEL_PATH="/home/frank23026407/custom_llama/model/tinyllama/TinyLlama_1.1v-comprehensive/bfp/block_1x128_mantissa_5"
SPARSITY_RATIO=0.5
LOG_DIR="/home/frank23026407/custom_llama/sparse-quantized-optimization/log"
mkdir -p $LOG_DIR

echo "Running Experiment C: Wanda Pruning (Sparsity: $SPARSITY_RATIO)"
echo "Model: $MODEL_PATH"

python /home/frank23026407/custom_llama/sparse-quantized-optimization/task_script_exp_c.py \
    --model_path "$MODEL_PATH" \
    --sparsity_ratio $SPARSITY_RATIO \
    | tee "$LOG_DIR/exp_c_wanda_sparsity_${SPARSITY_RATIO}.log"

echo "Done. Log saved to $LOG_DIR/exp_c_wanda_sparsity_${SPARSITY_RATIO}.log"
