#!/bin/bash

# Top 4 configurations from summary.md
# 1. 2x64, Mantissa 5
# 2. 1x128, Mantissa 5
# 3. 4x32, Mantissa 5
# 4. 128x1, Mantissa 5

# Set dynamic mix ratio
RATIO=0.01

LOG_DIR="testbench/log/dynamic_mix_hellaswag"
mkdir -p $LOG_DIR

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama-env

echo "Starting Dynamic Mix Precision Evaluation with Ratio $RATIO"

# Config 1: 2x64, Mantissa 5
echo "Running Config 1: 2x64, Mantissa 5"
python dynamic-mix-precision-awq/task_script_hellaswag_dynamic.py \
    --block_size "2x64" \
    --mantissa_bits 5 \
    --dynamic_mix_ratio $RATIO \
    --disable_tqdm \
    2>&1 | tee "$LOG_DIR/2x64_m5_r${RATIO}.log"

# Config 2: 1x128, Mantissa 5
echo "Running Config 2: 1x128, Mantissa 5"
python dynamic-mix-precision-awq/task_script_hellaswag_dynamic.py \
    --block_size "1x128" \
    --mantissa_bits 5 \
    --dynamic_mix_ratio $RATIO \
    --disable_tqdm \
    2>&1 | tee "$LOG_DIR/1x128_m5_r${RATIO}.log"

# Config 3: 4x32, Mantissa 5
echo "Running Config 3: 4x32, Mantissa 5"
python dynamic-mix-precision-awq/task_script_hellaswag_dynamic.py \
    --block_size "4x32" \
    --mantissa_bits 5 \
    --dynamic_mix_ratio $RATIO \
    --disable_tqdm \
    2>&1 | tee "$LOG_DIR/4x32_m5_r${RATIO}.log"

# Config 4: 128x1, Mantissa 5
echo "Running Config 4: 128x1, Mantissa 5"
python dynamic-mix-precision-awq/task_script_hellaswag_dynamic.py \
    --block_size "128x1" \
    --mantissa_bits 5 \
    --dynamic_mix_ratio $RATIO \
    --disable_tqdm \
    2>&1 | tee "$LOG_DIR/128x1_m5_r${RATIO}.log"

echo "Evaluation Complete. Logs saved to $LOG_DIR"
