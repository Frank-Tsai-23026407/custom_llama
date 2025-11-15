#!/bin/bash

# Set the project root and log directory
PROJECT_ROOT="/home/frank23026407/TinyLlama"
LOG_DIR="$PROJECT_ROOT/awq/log"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Set common parameters
B_SIZE=32

# Run for block floating point inference
for M_BIT in 5 4 3 2
do
    echo "----------------------------------------------------------------"
    echo "Running BFP inference with Mantissa Bits = $M_BIT, Block Size = $B_SIZE"
    echo "----------------------------------------------------------------"
    
    # Run the 'all' mode: quantize the model and then evaluate it on HellaSwag.
    python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" --bft --m_bit "$M_BIT" --b_size "$B_SIZE" | tee "$LOG_DIR/bft_m${M_BIT}_b${B_SIZE}_hellaswag.log"
done

echo "All block floating point inference are complete."
