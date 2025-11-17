#!/bin/bash

# Set the project root and log directory
PROJECT_ROOT="/home/frank23026407/TinyLlama"
LOG_DIR="$PROJECT_ROOT/awq/log"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Set common parameters

# runt in full precision
python "$PROJECT_ROOT/awq/task_script_hellaswag.py" | tee "$LOG_DIR/full_precision_hellaswag.log"

for B_SIZE in 128 64 32
do
    # Run for block floating point inference
    for M_BIT in 5 4 3 2
    do
        echo "----------------------------------------------------------------"
        echo "Running BFP inference with Mantissa Bits = $M_BIT, Block Size = $B_SIZE"
        echo "----------------------------------------------------------------"
        
        # Run the 'all' mode: quantize the model and then evaluate it on HellaSwag.
        python "$PROJECT_ROOT/awq/task_script_hellaswag.py" --bft --m_bit "$M_BIT" --b_size "$B_SIZE" | tee "$LOG_DIR/bft_m${M_BIT}_b${B_SIZE}_hellaswag.log"
    done

    echo "All block floating point inference are complete."

    # Run for different mantissa bits
    for M_BIT in 5 4 3 2
    do
        echo "----------------------------------------------------------------"
        echo "Running Mix-Precision AWQ with Mantissa Bits = $M_BIT, Block Size = $B_SIZE"
        echo "----------------------------------------------------------------"
        
        # Run the 'all' mode: quantize the model and then evaluate it on HellaSwag.
        python "$PROJECT_ROOT/tinyllama_mix-precision_awq.py" --mode all --m_bit "$M_BIT" --b_size "$B_SIZE" | tee "$LOG_DIR/mix-precision_awq_m${M_BIT}_b${B_SIZE}_hellaswag.log"
    done

    echo "All mix-precision AWQ experiments are complete."

    # Set common parameters for model paths
    G_SIZE=128
    BASE_MODEL_PATH="$PROJECT_ROOT/model/tinyllama/TinyLlmam_1.1v"

    # Run evaluation for different mantissa bits
    for M_BIT in 2 3 4 5
    do
        # Construct the path to the fix-precision quantized model directory
        MODEL_DIR="${BASE_MODEL_PATH}-awq-quantized-fix-precision-b${B_SIZE}-m${M_BIT}"

        echo "----------------------------------------------------------------"
        echo "Evaluating Fix-Precision AWQ model: $MODEL_DIR"
        echo "----------------------------------------------------------------"
        
        python "$PROJECT_ROOT/tinyllama_mix-precision_awq.py" --mode evaluate --model_path "$MODEL_DIR" | tee "$LOG_DIR/fix-precision_awq_eval_m${M_BIT}_b${B_SIZE}_g${G_SIZE}_hellaswag.log"
    done
done

echo "All fix-precision AWQ evaluation experiments are complete."
