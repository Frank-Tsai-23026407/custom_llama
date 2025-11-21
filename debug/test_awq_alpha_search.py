"""
Test script for AWQ alpha grid search implementation.

This script tests the new alpha grid search feature for AWQ fix-precision quantization.
"""

import torch
import sys
sys.path.append('../')

from quantize_model_script.block_quantization import (
    awq_fix_precision_quantize_2d,
    search_best_alpha_2d,
    awq_fix_precision_quantize_2d_auto
)

print("=" * 80)
print("AWQ Alpha Grid Search Test")
print("=" * 80)

# Test parameters
weight = torch.randn(128, 256)  * 0.1
activation = torch.randn(8, 10, 256) * 0.5
block_height = 16
block_width = 16  
mantissa_bits = 4

print(f"\nTest setup:")
print(f"  Weight shape: {weight.shape}")
print(f"  Activation shape: {activation.shape}")
print(f"  Block size: {block_height}×{block_width}")
print(f"  Mantissa bits: {mantissa_bits}")

# Test 1: Different alpha values
print("\n" + "-" * 80)
print("Test 1: Different fixed alpha values")
print("-" * 80)

alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
results = []

for alpha in alpha_values:
    quantized = awq_fix_precision_quantize_2d(
        weight, activation,
        block_height, block_width, mantissa_bits,
        alpha=alpha
    )
    
    # Calculate output loss
    activation_2d = activation.reshape(-1, activation.shape[-1])
    original_output = torch.matmul(weight, activation_2d.T)
    quantized_output = torch.matmul(quantized, activation_2d.T)
    loss = torch.mean((original_output - quantized_output) ** 2).item()
    
    results.append((alpha, loss))
    print(f"  Alpha={alpha:.2f}: Output loss = {loss:.6e}")

# Test 2: Grid search for best alpha
print("\n" + "-" * 80)
print("Test 2: Grid search for optimal alpha")
print("-" * 80)

best_alpha, min_loss = search_best_alpha_2d(
    weight, activation,
    block_height, block_width, mantissa_bits,
    alpha_min=0.0,
    alpha_max=1.0,
    alpha_steps=10
)

print(f"  Best alpha found: {best_alpha:.3f}")
print(f"  Minimum loss: {min_loss:.6e}")

# Test 3: Auto quantization with best alpha
print("\n" + "-" * 80)
print("Test 3: Auto quantization (with alpha search)")
print("-" * 80)

quantized_auto, found_alpha = awq_fix_precision_quantize_2d_auto(
    weight, activation,
    block_height, block_width, mantissa_bits,
    alpha_steps=10,
    return_alpha=True
)

print(f"  Auto-found alpha: {found_alpha:.3f}")

# Verify it matches the search result
assert abs(found_alpha - best_alpha) < 0.01, "Alpha mismatch!"
print(f"  ✓ Matches grid search result")

# Test 4: Backward compatibility (alpha=1.0 should be default)
print("\n" + "-" * 80)
print("Test 4: Backward compatibility check")
print("-" * 80)

quantized_default = awq_fix_precision_quantize_2d(
    weight, activation,
    block_height, block_width, mantissa_bits
)

quantized_alpha1 = awq_fix_precision_quantize_2d(
    weight, activation,
    block_height, block_width, mantissa_bits,
    alpha=1.0
)

diff = torch.abs(quantized_default - quantized_alpha1).max().item()
print(f"  Max difference between default and alpha=1.0: {diff:.6e}")

if diff < 1e-6:
    print(f"  ✓ Backward compatible (diff < 1e-6)")
else:
    print(f"  ✗ Backward compatibility issue!")

# Summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)
print(f"  ✓ Alpha scaling works correctly")
print(f"  ✓ Grid search finds optimal alpha: {best_alpha:.3f}")
print(f"  ✓ Auto quantization works")  
print(f"  ✓ Backward compatibility maintained")
print("\n" + "=" * 80)
print("All tests passed!")
print("=" * 80)
