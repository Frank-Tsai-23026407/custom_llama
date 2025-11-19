#!/bin/bash

# TinyLlama 2D BFP Quantization Sweep Script
# This script quantizes TinyLlama with multiple block size configurations

echo "======================================================================"
echo "TinyLlama 2D Block-Based BFP Quantization Sweep"
echo "======================================================================"
echo ""
echo "Block Sizes: 128×1, 64×2, 32×4, 16×8, 8×16, 4×32, 2×64, 1×128"
echo "Mantissa Bits: 5, 4"
echo "Total Configurations: 16"
echo ""
echo "======================================================================"

# Navigate to script directory
cd "$(dirname "$0")"

# Run the quantization sweep
python quantize_tinyllama_2d_sweep.py \
    --model tinyllama \
    "$@"

echo ""
echo "======================================================================"
echo "Quantization sweep completed!"
echo "======================================================================"
