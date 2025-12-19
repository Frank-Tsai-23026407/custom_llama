#!/usr/bin/env bash
set -euo pipefail

# Quantize Llama-3.2-1B across modes and settings
# Modes: BFP, Fix-Precision AWQ, Mix-Precision AWQ
# Block sizes: 128, 64, 32
# Mantissa bits: 5, 4, 3, 2

MODEL="llama-3.2-1b"
DEVICE="cuda"           # auto|cpu|cuda (used by AWQ scripts)
NUM_SAMPLES=128          # calibration samples for AWQ
DATASET="Salesforce/wikitext"
DATASET_CONFIG="wikitext-103-raw-v1"

BLOCK_SIZES=(128 64 32)
MANTISSAS=(5 4 3 2)

echo "[INFO] Starting quantization sweep for ${MODEL}"

# 1) Pure BFP quantization (no activations)
for b in "${BLOCK_SIZES[@]}"; do
  echo "[BFP] Block size=${b}"
  python "$(dirname "$0")/bfp_quantize.py" \
    --model "${MODEL}" \
    --block-size "${b}" \
    --mantissa-bits "${MANTISSAS[@]}"
  echo "[BFP] Completed block size=${b}"
  echo
done

# 2) Fix-Precision AWQ (activation aware)
for b in "${BLOCK_SIZES[@]}"; do
  echo "[FIX-AWQ] Block size=${b}"
  python "$(dirname "$0")/fix_precision_awq.py" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --dataset-config "${DATASET_CONFIG}" \
    --num-samples "${NUM_SAMPLES}" \
    --block-size "${b}" \
    --mantissa-bits "${MANTISSAS[@]}" \
    --device "${DEVICE}"
  echo "[FIX-AWQ] Completed block size=${b}"
  echo
done

# 3) Mix-Precision AWQ (activation aware)
for b in "${BLOCK_SIZES[@]}"; do
  echo "[MIX-AWQ] Block size=${b}"
  python "$(dirname "$0")/mix_precision_awq.py" \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --dataset-config "${DATASET_CONFIG}" \
    --num-samples "${NUM_SAMPLES}" \
    --block-size "${b}" \
    --mantissa-bits "${MANTISSAS[@]}" \
    --device "${DEVICE}"
  echo "[MIX-AWQ] Completed block size=${b}"
  echo
done

echo "[INFO] All quantization sweeps completed."
