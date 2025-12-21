#!/bin/bash

# Comprehensive TinyLlama 2D Quantization Script
# Supports BFP, AWQ (fix-precision), and AWQ (mix-precision)

echo "======================================================================"
echo "TinyLlama Comprehensive 2D Quantization"
echo "======================================================================"
echo ""
echo "Methods: BFP, AWQ-Fix, AWQ-Mix"
echo "Block Sizes: 128×1, 64×2, 32×4, 16×8, 8×16, 4×32, 2×64, 1×128"
echo "Mantissa Bits: 5, 4"
echo ""
echo "======================================================================"

# Navigate to script directory
cd "$(dirname "$0")"

# Default: run all methods
# You can specify --method bfp, --method awq-fix, --method awq-mix, or --method all

python quantize_tinyllama_comprehensive_2d.py \
    --model tinyllama \
    --method all \
    --top-k 16 \
    "$@"

echo ""
echo "======================================================================"
echo "Quantization complete!"
echo "======================================================================"
