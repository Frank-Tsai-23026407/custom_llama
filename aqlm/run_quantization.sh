#!/bin/bash
set -e

# Define paths
MODEL_ROOT="/home/frank23026407/custom_llama/model"
OUTPUT_ROOT="/home/frank23026407/custom_llama/aqlm/output"

# Create output directories
mkdir -p "$OUTPUT_ROOT/tinyllama"
mkdir -p "$OUTPUT_ROOT/llama-3.2-1b"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "Starting quantization for TinyLlama..."
python3 /home/frank23026407/custom_llama/aqlm/main.py \
    "$MODEL_ROOT/tinyllama/TinyLlama_1.1v" \
    c4 \
    --nsamples=128 \
    --model_seqlen=2048 \
    --num_codebooks=2 \
    --nbits_per_codebook=8 \
    --codebook_value_nbits=8 \
    --in_group_size=8 \
    --out_group_size=1 \
    --max_epochs=10 \
    --steps_per_epoch=100 \
    --finetune_max_epochs=2 \
    --init_max_iter=20 \
    --save "$OUTPUT_ROOT/tinyllama" \
    --devices cuda:0

echo "Finished quantization for TinyLlama."

echo "Starting quantization for Llama 3.2 1B..."
python3 /home/frank23026407/custom_llama/aqlm/main.py \
    "$MODEL_ROOT/llama-3.2-1b/Llama-3.2-1B" \
    c4 \
    --nsamples=128 \
    --model_seqlen=2048 \
    --num_codebooks=2 \
    --nbits_per_codebook=8 \
    --codebook_value_nbits=8 \
    --in_group_size=8 \
    --out_group_size=1 \
    --max_epochs=10 \
    --steps_per_epoch=100 \
    --finetune_max_epochs=2 \
    --init_max_iter=20 \
    --save "$OUTPUT_ROOT/llama-3.2-1b" \
    --devices cuda:0

echo "Finished quantization for Llama 3.2 1B."
