#!/bin/bash

# Quick precision sweep on a small subset of HellaSwag
# Useful for rapid testing and debugging

PROJECT_ROOT="/home/frank23026407/TinyLlama"
LOG_DIR="$PROJECT_ROOT/awq/log/precision_sweep_quick"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "Quick HellaSwag Precision Sweep (First 100 samples)"
echo "========================================================"

# Test configurations with small sample size for quick comparison
SAMPLE_SIZE=100

# Reference: HuggingFace
echo ""
echo "================================================================"
echo "Testing: HuggingFace Reference (Ground Truth)"
echo "================================================================"
python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
    --backend huggingface \
    --max_samples "$SAMPLE_SIZE" \
    | tee "$LOG_DIR/hf_reference.log"

# Clone with best config (bf16 + bf16 + fp32_softmax)
echo ""
echo "================================================================"
echo "Testing: Clone - BF16 Compute + BF16 RoPE + FP32 Softmax"
echo "================================================================"
python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
    --backend clone \
    --compute_dtype bf16 \
    --rope_cache_dtype bf16 \
    --softmax_fp32 \
    --max_samples "$SAMPLE_SIZE" \
    | tee "$LOG_DIR/clone_bf16_bf16_fp32soft.log"

# Clone with full bf16
echo ""
echo "================================================================"
echo "Testing: Clone - Full BF16 (no FP32 softmax)"
echo "================================================================"
python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
    --backend clone \
    --compute_dtype bf16 \
    --rope_cache_dtype bf16 \
    --max_samples "$SAMPLE_SIZE" \
    | tee "$LOG_DIR/clone_full_bf16.log"

# Clone with full fp32
echo ""
echo "================================================================"
echo "Testing: Clone - Full FP32 (Maximum Precision)"
echo "================================================================"
python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
    --backend clone \
    --compute_dtype fp32 \
    --rope_cache_dtype fp32 \
    --softmax_fp32 \
    --max_samples "$SAMPLE_SIZE" \
    | tee "$LOG_DIR/clone_full_fp32.log"

# Custom backend with bf16 policy
echo ""
echo "================================================================"
echo "Testing: Custom Backend (bf16 policy)"
echo "================================================================"
python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
    --backend custom \
    --precision_policy bf16 \
    --max_samples "$SAMPLE_SIZE" \
    | tee "$LOG_DIR/custom_bf16.log"

# Quick AWQ quantized models (HF backend)
echo ""
echo "================================================================"
echo "Quick: AWQ Mix-Precision/Fix-Precision b128 m2-m5 (HF backend)"
echo "================================================================"

AWQ_MODELS=(
    "model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m2"
    "model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m3"
    "model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m4"
    "model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m5"
    "model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m2"
    "model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m3"
    "model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m4"
    "model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m5"
)

for MPATH in "${AWQ_MODELS[@]}"; do
    BASENAME=$(basename "$MPATH")
    LOG_FILE="$LOG_DIR/awq_${BASENAME}.log"
    echo ""
    echo "----------------------------------------------------------------"
    echo "Testing AWQ model: $BASENAME (backend=huggingface)"
    echo "----------------------------------------------------------------"
    python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
            --backend huggingface \
            --model_path "$MPATH" \
            --max_samples "$SAMPLE_SIZE" \
            | tee "$LOG_FILE"
done

echo ""
echo "========================================================"
echo "Quick sweep completed!"
echo "========================================================"
echo ""
echo "Results Summary:"
echo "----------------"
grep "Accuracy (acc_norm):" "$LOG_DIR"/*.log | sed 's|.*/||' | sed 's|\.log:| |'

echo ""
echo "Full logs are in: $LOG_DIR"
