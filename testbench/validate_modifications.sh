#!/bin/bash

# Quick validation script to test if all modifications work correctly
# Tests each configuration with just 5 samples

PROJECT_ROOT="/home/frank23026407/TinyLlama"
TEST_SAMPLES=5

echo "========================================================"
echo "Quick Validation Test (5 samples per config)"
echo "========================================================"

echo ""
echo "Test 1: HuggingFace backend"
python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
    --backend huggingface \
    --max_samples "$TEST_SAMPLES" 2>&1 | tail -8

echo ""
echo "Test 2: Clone with precision parameters"
python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
    --backend clone \
    --compute_dtype bf16 \
    --rope_cache_dtype bf16 \
    --softmax_fp32 \
    --max_samples "$TEST_SAMPLES" 2>&1 | tail -8

echo ""
echo "Test 3: Custom with precision policy"
python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
    --backend custom \
    --precision_policy bf16 \
    --max_samples "$TEST_SAMPLES" 2>&1 | tail -8

echo ""
echo "========================================================"
echo "Validation Complete!"
echo "========================================================"
echo ""
echo "If all tests showed accuracy results, the modifications work correctly."
echo "You can now run the full sweep with:"
echo "  sh awq/run_precision_sweep_quick.sh      (5-10 minutes)"
echo "  sh awq/run_precision_sweep_hellaswag.sh  (2-4 hours)"
