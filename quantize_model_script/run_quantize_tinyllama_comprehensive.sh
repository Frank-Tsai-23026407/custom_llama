#!/bin/bash

# Comprehensive TinyLlama 2D Quantization Script
# Uses unified model_quantization.py
# Supports: BFP, AWQ-Fix, AWQ-Mix
# Block Sizes: 128×1, 64×2, 32×4, 16×8, 8×16, 4×32, 2×64, 1×128
# Mantissa Bits: 5, 4, 3

echo "======================================================================"
echo "TinyLlama Comprehensive 2D Quantization (Unified Script)"
echo "======================================================================"
echo ""
echo "Methods: BFP, AWQ-Fix, AWQ-Mix"
echo "Block Sizes: 128×1, 64×2, 32×4, 16×8, 8×16, 4×32, 2×64, 1×128"
echo "Mantissa Bits: 5, 4, 3"
echo ""
echo "======================================================================"

# Navigate to script directory
cd "$(dirname "$0")"

# Run unified quantization script
# You can override with: --method bfp, --method awq-fix, --method awq-mix
python model_quantization.py \
    --model tinyllama \
    --method all \
    --top-k 16 \
    --skip-generation-test \
    --alpha-search-steps 20 \
    "$@"

echo ""
echo "======================================================================"
echo "Quantization complete!"
echo "======================================================================"
