#!/bin/bash

# Sweep different precision configurations on HellaSwag benchmark
# This script tests various dtype and precision combinations

PROJECT_ROOT="/home/frank23026407/TinyLlama"
LOG_DIR="$PROJECT_ROOT/awq/log/precision_sweep"

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# echo "========================================================"
# echo "HellaSwag Precision Configuration Sweep"
# echo "========================================================"

# # Configuration 1: Full BF16 (match_hf policy equivalent)
# echo ""
# echo "================================================================"
# echo "Config 1: BF16 Compute + BF16 RoPE + FP32 Softmax (Best Config)"
# echo "================================================================"
# python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
#     --backend clone \
#     --compute_dtype bf16 \
#     --rope_cache_dtype bf16 \
#     --softmax_fp32 \
#     | tee "$LOG_DIR/config1_bf16_bf16_fp32softmax.log"

# # Configuration 2: Full BF16 without FP32 Softmax
# echo ""
# echo "================================================================"
# echo "Config 2: BF16 Compute + BF16 RoPE + BF16 Softmax"
# echo "================================================================"
# python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
#     --backend clone \
#     --compute_dtype bf16 \
#     --rope_cache_dtype bf16 \
#     | tee "$LOG_DIR/config2_bf16_bf16_bf16softmax.log"

# # Configuration 3: FP32 Compute + BF16 RoPE + FP32 Softmax
# echo ""
# echo "================================================================"
# echo "Config 3: FP32 Compute + BF16 RoPE + FP32 Softmax"
# echo "================================================================"
# python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
#     --backend clone \
#     --compute_dtype fp32 \
#     --rope_cache_dtype bf16 \
#     --softmax_fp32 \
#     | tee "$LOG_DIR/config3_fp32_bf16_fp32softmax.log"

# # Configuration 4: BF16 Compute + FP32 RoPE + FP32 Softmax
# echo ""
# echo "================================================================"
# echo "Config 4: BF16 Compute + FP32 RoPE + FP32 Softmax"
# echo "================================================================"
# python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
#     --backend clone \
#     --compute_dtype bf16 \
#     --rope_cache_dtype fp32 \
#     --softmax_fp32 \
#     | tee "$LOG_DIR/config4_bf16_fp32_fp32softmax.log"

# # Configuration 5: Full FP32 (Maximum Precision)
# echo ""
# echo "================================================================"
# echo "Config 5: FP32 Compute + FP32 RoPE + FP32 Softmax (Max Precision)"
# echo "================================================================"
# python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
#     --backend clone \
#     --compute_dtype fp32 \
#     --rope_cache_dtype fp32 \
#     --softmax_fp32 \
#     | tee "$LOG_DIR/config5_fp32_fp32_fp32softmax.log"

# # Configuration 6: HuggingFace Reference (Ground Truth)
# echo ""
# echo "================================================================"
# echo "Config 6: HuggingFace Reference (Ground Truth)"
# echo "================================================================"
# python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
#     --backend huggingface \
#     | tee "$LOG_DIR/config6_huggingface_reference.log"

# # Configuration 7: Custom Backend with default policy
# echo ""
# echo "================================================================"
# echo "Config 7: Custom Backend (default policy)"
# echo "================================================================"
# python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
#     --backend custom \
#     --precision_policy default \
#     | tee "$LOG_DIR/config7_custom_default.log"

# # Configuration 8: Custom Backend with bf16 policy
# echo ""
# echo "================================================================"
# echo "Config 8: Custom Backend (bf16 policy)"
# echo "================================================================"
# python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
#     --backend custom \
#     --precision_policy bf16 \
#     | tee "$LOG_DIR/config8_custom_bf16.log"

# AWQ Quantized Models (HuggingFace forward on quantized checkpoints)
echo ""
echo "================================================================"
echo "AWQ: Mix-Precision b128 m2-m5 and Fix-Precision b128 m2-m5 (HF backend)"
echo "================================================================"

AWQ_MODELS=(
    "model/TinyLlama_1.1v-awq-quantized-mix-precision-b32-m2"
    "model/TinyLlama_1.1v-awq-quantized-mix-precision-b32-m3"
    "model/TinyLlama_1.1v-awq-quantized-mix-precision-b32-m4"
    "model/TinyLlama_1.1v-awq-quantized-mix-precision-b32-m5"
    "model/TinyLlama_1.1v-awq-quantized-fix-precision-b32-m2"
    "model/TinyLlama_1.1v-awq-quantized-fix-precision-b32-m3"
    "model/TinyLlama_1.1v-awq-quantized-fix-precision-b32-m4"
    "model/TinyLlama_1.1v-awq-quantized-fix-precision-b32-m5"
)

for MPATH in "${AWQ_MODELS[@]}"; do
    BASENAME=$(basename "$MPATH")
    LOG_FILE="$LOG_DIR/awq_${BASENAME}_custom.log"
    echo ""
    echo "----------------------------------------------------------------"
    echo "Evaluating AWQ model: $BASENAME (backend=huggingface)"
    echo "----------------------------------------------------------------"
    python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
            --backend custom \
            --model_path "$MPATH" \
            | tee "$LOG_FILE"
done

echo ""
echo "========================================================"
echo "All precision configurations completed!"
echo "========================================================"
echo ""
echo "Summary of logs:"
ls -lh "$LOG_DIR"/*.log

echo ""
echo "To extract accuracy results:"
echo "grep -h 'Accuracy (acc_norm):' $LOG_DIR/*.log"
