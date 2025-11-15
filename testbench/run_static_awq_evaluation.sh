#!/bin/bash

# This script runs the HellaSwag evaluation on all previously generated
# fix-precision AWQ quantized models.

# Define project root and log directory
PROJECT_ROOT="/home/frank23026407/TinyLlama"
LOG_DIR="$PROJECT_ROOT/awq/log"
mkdir -p "$LOG_DIR"

# Set common parameters for model paths
B_SIZE=128
BASE_MODEL_PATH="$PROJECT_ROOT/model/tinyllama/TinyLlmam_1.1v"

# Run evaluation for different mantissa bits
for M_BIT in 2 3 4 5
do
    # Construct the path to the fix-precision quantized model directory
    MODEL_DIR="${BASE_MODEL_PATH}-awq-quantized-fix-precision-b${B_SIZE}-m${M_BIT}"

    # Check if the model directory exists before running evaluation
    if [ ! -d "$MODEL_DIR" ]; then
        echo "Error: Model directory not found: $MODEL_DIR"
        continue # Skip to the next iteration
    fi

    echo "----------------------------------------------------------------"
    echo "Evaluating Fix-Precision AWQ model: $MODEL_DIR"
    echo "----------------------------------------------------------------"
    
    python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" --model_path "$MODEL_DIR" | tee "$LOG_DIR/fix-precision_awq_eval_m${M_BIT}_b${B_SIZE}_hellaswag.log"
done

echo "All fix-precision AWQ evaluation experiments are complete."
