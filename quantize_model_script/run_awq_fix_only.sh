#!/bin/bash

# AWQ Fix-Precision quantization (16 configurations)

echo "======================================================================"
echo "TinyLlama AWQ Fix-Precision 2D Quantization"
echo "======================================================================"

cd "$(dirname "$0")"

python quantize_tinyllama_comprehensive_2d.py \
    --model tinyllama \
    --method awq-fix \
    --top-k 16 \
    "$@"

echo ""
echo "AWQ Fix-Precision quantization complete!"
