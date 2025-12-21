#!/bin/bash

# Define block size pairs as (height width)
BLOCK_SIZES=(
    # block size 128
    "128 1"
    "64 2"
    "32 4"
    "16 8"
    "8 16"
    "4 32"
    "2 64"
    "1 128"
    # # block size 64
    # "64 1"
    # "32 2"
    # "16 4"
    # "8 8"
    # "4 16"
    # "2 32"
    # "1 64"
    # # block size 32
    # "32 1"
    # "16 2"
    # "8 4"
    # "4 8"
    # "2 16"
    # "1 32"
)

# Mantissa bits to try
M_BITS="5 4 3"

# Generate strings for display
DISPLAY_SIZES=""
for size in "${BLOCK_SIZES[@]}"; do
    H=$(echo $size | cut -d' ' -f1)
    W=$(echo $size | cut -d' ' -f2)
    DISPLAY_SIZES="${DISPLAY_SIZES}${H}x${W}, "
done
DISPLAY_SIZES=${DISPLAY_SIZES%, } # Remove trailing comma

echo "======================================================================"
echo "TinyLlama AWQ Fix-Precision Quantization"
echo "======================================================================"
echo ""
echo "Methods: AWQ (BFP-based)"
echo "Block Sizes: ${DISPLAY_SIZES}"
echo "Evaluate on wikitext-103 PPL and HellaSwag accuracy"
echo "Mantissa Bits (include sign bit): ${M_BITS// /, }"
echo "Only dry run quantization (no save model)"
echo ""
echo "======================================================================"

# Navigate to script directory
cd "$(dirname "$0")"

for size in "${BLOCK_SIZES[@]}"; do
    H=$(echo $size | cut -d' ' -f1)
    W=$(echo $size | cut -d' ' -f2)
    
    echo "Running 2D AWQ with Block Size ${H}x${W}..."
    
    python fix_precision_awq.py \
        --model "TinyLlama/TinyLlama_v1.1" \
        --block-height $H \
        --block-width $W \
        --mantissa-bits $M_BITS \
        --eval-ppl \
        --eval-hellaswag \
        --dry-run \
        "$@"
done

echo ""
echo "======================================================================"
echo "Evaluation complete!"
echo "======================================================================"
