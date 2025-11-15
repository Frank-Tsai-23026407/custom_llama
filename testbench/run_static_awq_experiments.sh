#!/bin/bash

# This script runs the fix-precision AWQ quantization.
# The Python script 'quantize_tinyllama.py' is configured to loop through
# mantissa bits from 2 to 5 with a fixed block size of 128.

# Define project root and log directory
PROJECT_ROOT="/home/frank23026407/TinyLlama"
LOG_DIR="$PROJECT_ROOT/awq/log"
mkdir -p "$LOG_DIR"

# Define the log file for the entire fix-precision quantization run
LOG_FILE="$LOG_DIR/fix-precision_awq_b128_m2-5.log"

echo "----------------------------------------------------------------"
echo "Starting Fix-Precision AWQ Quantization for Mantissa Bits 2 through 5"
echo "Block Size is fixed at 128 inside the Python script."
echo "Output will be logged to: $LOG_FILE"
echo "----------------------------------------------------------------"

# Change to the project root directory to handle Python's relative imports
cd "$PROJECT_ROOT" || exit

# Run the quantization script and pipe the output to tee
python -m block_quantization.quantize_tinyllama | tee "$LOG_FILE"

echo "All fix-precision AWQ quantization tasks are complete. Log saved to: $LOG_FILE"
