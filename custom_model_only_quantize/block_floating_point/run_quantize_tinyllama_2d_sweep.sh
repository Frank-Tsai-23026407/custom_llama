#!/bin/bash

# TinyLlama 2D BFP Quantization Sweep Script
# This script uses a shell loop to call bfp_quantize_2d.py with multiple block configurations

set -e

MODEL="tinyllama"
MANTISSA_BITS="5 4"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "TinyLlama 2D Block-Based BFP Quantization Sweep (Shell Version)"
echo "======================================================================"
echo ""
echo "Model: ${MODEL}"
echo "Block Sizes: 128x1, 64x2, 32x4, 16x8, 8x16, 4x32, 2x64, 1x128"
echo "Mantissa Bits: ${MANTISSA_BITS}"
echo ""
echo "======================================================================"

# Define the block height and width combinations
BLOCK_HEIGHTS=(128 64 32 16 8 4 2 1)
BLOCK_WIDTHS=(1 2 4 8 16 32 64 128)

# Loop through the combinations
for i in "${!BLOCK_HEIGHTS[@]}"; do
    H=${BLOCK_HEIGHTS[$i]}
    W=${BLOCK_WIDTHS[$i]}
    
    echo ""
    echo "[Config $((i+1))/8] Running BFP 2D: Block ${H}x${W}"
    echo "----------------------------------------------------------------------"
    
    python "${SCRIPT_DIR}/bfp_quantize_2d.py" \
        --model "${MODEL}" \
        --block-height "${H}" \
        --block-width "${W}" \
        --mantissa-bits ${MANTISSA_BITS} \
        "$@"
done

echo ""
echo "======================================================================"
echo "Quantization sweep completed!"
echo "======================================================================"
