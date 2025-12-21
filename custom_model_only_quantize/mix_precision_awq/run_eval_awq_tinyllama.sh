#!/bin/bash

# Define block size pairs as (height width)
BLOCK_SIZES=(
    # # block size 128
    # "128 1"
    # "64 2"
    # "32 4"
    # "16 8"
    # "8 16"
    # "4 32"
    # "2 64"
    # "1 128"
    # block size 64
    "64 1"
    "32 2"
    "16 4"
    "8 8"
    "4 16"
    "2 32"
    "1 64"
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

# Top-k ratio for salient weight protection (e.g., 0.01 for 1%)
TOP_K_RATIO=0.0625

# Generate strings for display
DISPLAY_SIZES=""
for size in "${BLOCK_SIZES[@]}"; do
    H=$(echo $size | cut -d' ' -f1)
    W=$(echo $size | cut -d' ' -f2)
    DISPLAY_SIZES="${DISPLAY_SIZES}${H}x${W}, "
done
DISPLAY_SIZES=${DISPLAY_SIZES%, } # Remove trailing comma

echo "======================================================================"
echo "TinyLlama AWQ Mixed-Precision Quantization"
echo "======================================================================"
echo ""
echo "Methods: Mixed-Precision AWQ (BFP-based 2D)"
echo "Block Sizes: ${DISPLAY_SIZES}"
echo "Top-K Ratio: ${TOP_K_RATIO}"
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
    
    echo "Running 2D Mixed-Precision AWQ with Block Size ${H}x${W} (Ratio=${TOP_K_RATIO})..."
    
    python mix_precision_awq.py \
        --model "TinyLlama/TinyLlama_v1.1" \
        --block-height $H \
        --block-width $W \
        --mantissa-bits $M_BITS \
        --top-k-ratio $TOP_K_RATIO \
        --eval-ppl \
        --eval-hellaswag \
        --dry-run \
        "$@"
done

echo ""
echo "======================================================================"
echo "Evaluation complete!"
echo "======================================================================"
