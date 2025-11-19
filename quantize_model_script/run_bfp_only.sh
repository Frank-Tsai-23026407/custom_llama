#!/bin/bash

# Quick BFP quantization only (fastest, 16 configurations)

echo "======================================================================"
echo "TinyLlama BFP 2D Quantization (Fast Mode)"
echo "======================================================================"

cd "$(dirname "$0")"

python quantize_tinyllama_comprehensive_2d.py \
    --model tinyllama \
    --method bfp \
    --skip-generation-test \
    "$@"

echo ""
echo "BFP quantization complete!"
