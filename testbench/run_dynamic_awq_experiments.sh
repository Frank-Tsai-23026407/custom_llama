#!/bin/bash

# This script runs the mix-precision AWQ quantization and HellaSwag evaluation
# for different mantissa bit settings. The block size is fixed at 128.

# Ensure the log directory exists
LOG_DIR="/home/frank23026407/TinyLlama/awq/log"
mkdir -p "$LOG_DIR"

# Set common parameters
B_SIZE=128

# Run for different mantissa bits
for M_BIT in 5 4 3 2
do
    echo "----------------------------------------------------------------"
    echo "Running Mix-Precision AWQ with Mantissa Bits = $M_BIT, Block Size = $B_SIZE"
    echo "----------------------------------------------------------------"
    
    # Run the 'all' mode: quantize the model and then evaluate it on HellaSwag.
    python /home/frank23026407/TinyLlama/tinyllama_mix-precision_awq.py --mode all --m_bit "$M_BIT" --b_size "$B_SIZE" | tee "$LOG_DIR/mix-precision_awq_m${M_BIT}_b${B_SIZE}_hellaswag.log"
done

echo "All mix-precision AWQ experiments are complete."
