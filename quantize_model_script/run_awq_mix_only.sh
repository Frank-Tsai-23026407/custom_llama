#!/bin/bash

# AWQ Mix-Precision quantization (16 configurations)

echo "======================================================================"
echo "TinyLlama AWQ Mix-Precision 2D Quantization"
echo "======================================================================"

cd "$(dirname "$0")"

python quantize_tinyllama_comprehensive_2d.py \
    --model tinyllama \
    --method awq-mix \
    --top-k 16 \
    "$@"

echo ""
echo "AWQ Mix-Precision quantization complete!"
