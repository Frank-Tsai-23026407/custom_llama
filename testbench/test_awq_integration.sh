#!/bin/bash
# Quick test to verify AWQ models work with precision sweep

PROJECT_ROOT="/home/frank23026407/TinyLlama"
LOG_DIR="$PROJECT_ROOT/awq/log/awq_integration_test"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "AWQ Integration Test (5 samples each)"
echo "=========================================="

# Test one mix-precision and one fix-precision model
TEST_MODELS=(
  "model/tinyllama/TinyLlmam_1.1v-awq-quantized-mix-precision-b128-m2"
  "model/tinyllama/TinyLlmam_1.1v-awq-quantized-fix-precision-b128-m2"
)

for MPATH in "${TEST_MODELS[@]}"; do
  BASENAME=$(basename "$MPATH")
  LOG_FILE="$LOG_DIR/awq_${BASENAME}.log"
  
  echo ""
  echo "Testing: $BASENAME"
  conda run -n tinyllama-env python "$PROJECT_ROOT/awq/tinyllama_my_bfp_hellaswag.py" \
    --backend huggingface \
    --model_path "$PROJECT_ROOT/$MPATH" \
    --max_samples 5 \
    > "$LOG_FILE" 2>&1
  
  # Show result
  grep "Accuracy (acc_norm):" "$LOG_FILE" || echo "  ❌ Failed (check log)"
done

echo ""
echo "=========================================="
echo "Testing analyzer with AWQ logs"
echo "=========================================="
conda run -n tinyllama-env python "$PROJECT_ROOT/awq/analyze_hellaswag_results.py" "$LOG_DIR"

echo ""
echo "✅ Integration test complete!"
