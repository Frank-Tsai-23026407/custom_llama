#!/usr/bin/env bash
set -euo pipefail

# Quantize Llama-3.2-1B using unified model_quantization.py script
# Supports: BFP, Fix-Precision AWQ, Mix-Precision AWQ
# Block sizes: 128×1, 64×2, 32×4, 16×8, 8×16, 4×32, 2×64, 1×128
# Mantissa bits: 5, 4, 3, 2

MODEL="llama-3.2-1b"
NUM_SAMPLES=128
DATASET="Salesforce/wikitext"
DATASET_CONFIG="wikitext-103-raw-v1"

echo "[INFO] Starting comprehensive quantization for ${MODEL}"
echo "======================================================================"

# Navigate to script directory
cd "$(dirname "$0")"

# Run unified quantization script with all methods
python model_quantization.py \
    --model "${MODEL}" \
    --method all \
    --dataset "${DATASET}" \
    --dataset-config "${DATASET_CONFIG}" \
    --num-samples "${NUM_SAMPLES}" \
    --mantissa-bits 5 4 3 2 \
    --top-k 16 \
    --skip-generation-test \
    "$@"

echo ""
echo "======================================================================"
echo "[INFO] All quantization configurations completed for ${MODEL}"
echo "======================================================================"
