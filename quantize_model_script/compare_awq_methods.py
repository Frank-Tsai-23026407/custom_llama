"""
Demonstration script comparing BFP, AWQ Fix-Precision, and AWQ Mix-Precision.

This script clearly shows the difference between:
1. BFP: Standard block quantization (no activation awareness)
2. Fix-Precision: Scaling-based AWQ (all weights in BFP, but scaled)
3. Mix-Precision: Hybrid AWQ (top-k in FP32, rest in BFP)
"""

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantize_model_script.block_quantization_2d import (
    block_floating_point_quantize_2d,
    awq_fix_precision_quantize_2d,
    awq_mix_precision_quantize_2d
)


def analyze_quantization_result(original, quantized, method_name):
    """Analyze and print quantization results."""
    mse = torch.mean((original - quantized) ** 2).item()
    max_error = torch.max(torch.abs(original - quantized)).item()
    
    # Count how many values are exactly preserved (for mix-precision)
    exact_matches = torch.sum(torch.isclose(original, quantized, rtol=1e-7, atol=1e-9)).item()
    total_elements = original.numel()
    
    print(f"\n{method_name}:")
    print(f"  MSE: {mse:.6e}")
    print(f"  Max Error: {max_error:.6e}")
    print(f"  Exact Matches: {exact_matches}/{total_elements} ({exact_matches/total_elements*100:.2f}%)")
    
    return mse


def main():
    print("=" * 80)
    print("Quantization Methods Comparison: BFP vs Fix-Precision vs Mix-Precision")
    print("=" * 80)
    
    # Create test weight matrix and activations
    torch.manual_seed(42)
    weight = torch.randn(128, 256) * 0.1
    activations = torch.randn(8, 10, 256) * 0.5
    
    BLOCK_HEIGHT = 16
    BLOCK_WIDTH = 16
    MANTISSA_BITS = 4
    TOP_K = 16
    
    print(f"\nConfiguration:")
    print(f"  Weight shape: {weight.shape}")
    print(f"  Activation shape: {activations.shape}")
    print(f"  Block size: {BLOCK_HEIGHT}×{BLOCK_WIDTH}")
    print(f"  Mantissa bits: {MANTISSA_BITS}")
    print(f"  Top-K (for mix-precision): {TOP_K}")
    
    print("\n" + "=" * 80)
    print("Running Quantization Methods...")
    print("=" * 80)
    
    # 1. BFP: Standard block quantization (baseline)
    print("\n[1] BFP (Block Floating Point) - Baseline")
    print("    - No activation awareness")
    print("    - All weights uniformly quantized to BFP")
    quantized_bfp = block_floating_point_quantize_2d(
        weight,
        block_height=BLOCK_HEIGHT,
        block_width=BLOCK_WIDTH,
        mantissa_bits=MANTISSA_BITS
    )
    mse_bfp = analyze_quantization_result(weight, quantized_bfp, "Result")
    
    # 2. AWQ Fix-Precision: Scaling-based, all weights in BFP
    print("\n" + "-" * 80)
    print("[2] AWQ Fix-Precision (Scaling-Based)")
    print("    - Uses activation magnitudes to compute scaling factors")
    print("    - Scale weights UP before quantization: W_scaled = W * s")
    print("    - Apply BFP quantization: Q(W_scaled)")
    print("    - Scale back DOWN: W_final = Q(W_scaled) / s")
    print("    - ALL weights represented in BFP (hardware-friendly)")
    quantized_fix = awq_fix_precision_quantize_2d(
        weight,
        activations,
        block_height=BLOCK_HEIGHT,
        block_width=BLOCK_WIDTH,
        mantissa_bits=MANTISSA_BITS
    )
    mse_fix = analyze_quantization_result(weight, quantized_fix, "Result")
    
    # 3. AWQ Mix-Precision: Top-K in FP32, rest in BFP
    print("\n" + "-" * 80)
    print(f"[3] AWQ Mix-Precision (Hybrid FP32+BFP)")
    print(f"    - Uses activation magnitudes to identify salient weights")
    print(f"    - Per block: find top-{TOP_K} most important ENTRIES (elements)")
    print(f"    - IMPORTANT: top-{TOP_K} means {TOP_K} individual entries per block")
    print(f"    - NOT per-row or per-column, but from the entire flattened block")
    print(f"    - For {BLOCK_HEIGHT}\u00d7{BLOCK_WIDTH} block: select {TOP_K} out of {BLOCK_HEIGHT*BLOCK_WIDTH} entries ({TOP_K/(BLOCK_HEIGHT*BLOCK_WIDTH)*100:.1f}%)")
    print(f"    - These {TOP_K} entries can be scattered across different rows/columns")
    print(f"    - Preserve selected entries in FULL FP32 precision (not quantized)")
    print(f"    - Quantize remaining entries to BFP")
    print(f"    - Result: Mixed FP32 + BFP (non-uniform format)")
    quantized_mix = awq_mix_precision_quantize_2d(
        weight,
        activations,
        block_height=BLOCK_HEIGHT,
        block_width=BLOCK_WIDTH,
        mantissa_bits=MANTISSA_BITS,
        top_k=TOP_K
    )
    mse_mix = analyze_quantization_result(weight, quantized_mix, "Result")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Method':<25} {'MSE':<15} {'vs BFP':<15} {'Format':<20}")
    print("-" * 80)
    print(f"{'BFP (baseline)':<25} {mse_bfp:<15.6e} {'—':<15} {'All BFP':<20}")
    
    improvement_fix = (mse_bfp - mse_fix) / mse_bfp * 100
    print(f"{'AWQ Fix-Precision':<25} {mse_fix:<15.6e} {improvement_fix:>+13.2f}% {'All BFP (scaled)':<20}")
    
    improvement_mix = (mse_bfp - mse_mix) / mse_bfp * 100
    print(f"{'AWQ Mix-Precision':<25} {mse_mix:<15.6e} {improvement_mix:>+13.2f}% {'FP32 + BFP':<20}")
    
    print("\n" + "=" * 80)
    print("KEY DIFFERENCES")
    print("=" * 80)
    print(f"""
Fix-Precision vs Mix-Precision:

Fix-Precision (Scaling-Based):
  ✓ All weights in BFP format (uniform representation)
  ✓ Uses scaling to protect salient weights during quantization
  ✓ Hardware-friendly, easy to deploy
  ✓ Better than baseline BFP, but not as accurate as mix-precision
  ✓ Process: scale → quantize → descale

Mix-Precision (Hybrid):
  ✓ Salient weight ENTRIES stored in full FP32 (no quantization)
  ✓ Non-salient entries in BFP
  ✓ Top-k selection: per block, {TOP_K} entries out of {BLOCK_HEIGHT*BLOCK_WIDTH} total
  ✓ Selected entries can be at any position (not restricted to same row/column)
  ✓ Best accuracy preservation
  ✗ Non-uniform format (mix of FP32 and BFP)
  ✗ Slightly larger storage overhead
  ✗ May require custom kernels for efficient inference

Top-K Selection Example ({BLOCK_HEIGHT}\u00d7{BLOCK_WIDTH} block, top_k={TOP_K}):
  - Total entries per block: {BLOCK_HEIGHT*BLOCK_WIDTH}
  - Selected for FP32: {TOP_K} entries ({TOP_K/(BLOCK_HEIGHT*BLOCK_WIDTH)*100:.1f}% of block)
  - Quantized to BFP: {BLOCK_HEIGHT*BLOCK_WIDTH - TOP_K} entries ({(1-TOP_K/(BLOCK_HEIGHT*BLOCK_WIDTH))*100:.1f}% of block)
  - Selection criterion: Highest salience = |weight \u00d7 importance_score|
  - Location: Can be scattered anywhere in the block (e.g., entry [0,5], [2,3], [7,1], etc.)
""")
    
    print("=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
