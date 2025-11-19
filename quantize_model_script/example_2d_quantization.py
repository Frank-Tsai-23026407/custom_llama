"""
Standalone example demonstrating 2D block quantization without model dependencies.
This script can be run without installing transformers or loading full models.
"""
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantize_model_script.block_quantization_2d import (
    block_floating_point_quantize_2d,
    awq_quantize_2d
)


def example_1_basic_bfp():
    """Example 1: Basic BFP quantization with 2D blocks."""
    print("=" * 70)
    print("Example 1: Basic BFP 2D Quantization")
    print("=" * 70)
    
    # Create a 2048x2048 weight matrix (as mentioned in problem statement)
    print("\nCreating a 2048x2048 weight matrix...")
    weight = torch.randn(2048, 2048) * 0.1
    
    # Define block size: 32x16 (as mentioned in problem statement)
    block_height = 32
    block_width = 16
    mantissa_bits = 4
    
    print(f"Weight shape: {weight.shape}")
    print(f"Block size: {block_height}x{block_width}")
    print(f"Mantissa bits: {mantissa_bits}")
    
    # Calculate number of blocks
    num_blocks_h = weight.shape[0] // block_height
    num_blocks_w = weight.shape[1] // block_width
    total_blocks = num_blocks_h * num_blocks_w
    
    print(f"Number of blocks: {num_blocks_h}x{num_blocks_w} = {total_blocks}")
    print(f"Each block shares 1 scaling factor")
    print(f"Total scaling factors: {total_blocks}")
    
    # Apply BFP quantization
    print("\nApplying BFP quantization...")
    quantized_weight = block_floating_point_quantize_2d(
        weight,
        block_height=block_height,
        block_width=block_width,
        mantissa_bits=mantissa_bits
    )
    
    # Calculate metrics
    mse = torch.mean((weight - quantized_weight) ** 2)
    max_error = torch.max(torch.abs(weight - quantized_weight))
    
    print(f"\nResults:")
    print(f"  Mean Squared Error: {mse.item():.6e}")
    print(f"  Max Absolute Error: {max_error.item():.6f}")
    print(f"  Quantized weight shape: {quantized_weight.shape}")
    
    # Show one block in detail
    print(f"\nDetailed view of block [0, 0]:")
    block_orig = weight[:block_height, :block_width]
    block_quant = quantized_weight[:block_height, :block_width]
    
    print(f"  Original block:")
    print(f"    Shape: {block_orig.shape}")
    print(f"    Min: {block_orig.min().item():.6f}")
    print(f"    Max: {block_orig.max().item():.6f}")
    print(f"    Mean: {block_orig.mean().item():.6f}")
    
    print(f"  Quantized block:")
    print(f"    Shape: {block_quant.shape}")
    print(f"    Min: {block_quant.min().item():.6f}")
    print(f"    Max: {block_quant.max().item():.6f}")
    print(f"    Mean: {block_quant.mean().item():.6f}")
    
    print(f"  Block MSE: {torch.mean((block_orig - block_quant) ** 2).item():.6e}")


def example_2_different_block_sizes():
    """Example 2: Comparing different block sizes."""
    print("\n" + "=" * 70)
    print("Example 2: Effect of Different Block Sizes")
    print("=" * 70)
    
    # Create a weight matrix
    weight = torch.randn(512, 512) * 0.1
    mantissa_bits = 4
    
    print(f"\nWeight shape: {weight.shape}")
    print(f"Mantissa bits: {mantissa_bits}")
    
    # Test different block sizes
    block_configs = [
        (16, 16, "Square blocks (16x16)"),
        (32, 16, "Rectangular blocks (32x16) - Problem statement example"),
        (8, 32, "Rectangular blocks (8x32)"),
        (64, 64, "Large square blocks (64x64)"),
    ]
    
    print("\nComparing block sizes:")
    print("-" * 70)
    
    results = []
    for block_h, block_w, description in block_configs:
        quantized = block_floating_point_quantize_2d(
            weight, block_h, block_w, mantissa_bits
        )
        
        mse = torch.mean((weight - quantized) ** 2)
        num_blocks = ((weight.shape[0] + block_h - 1) // block_h) * \
                     ((weight.shape[1] + block_w - 1) // block_w)
        
        results.append((description, block_h, block_w, num_blocks, mse.item()))
        
        print(f"{description}")
        print(f"  Block size: {block_h}x{block_w}")
        print(f"  Number of blocks: {num_blocks}")
        print(f"  MSE: {mse.item():.6e}")
        print()


def example_3_awq_with_salient_weights():
    """Example 3: AWQ with salient weight preservation."""
    print("=" * 70)
    print("Example 3: AWQ with Salient Weights (Top-K per Block)")
    print("=" * 70)
    
    # Create weight and activation matrices
    weight = torch.randn(128, 256) * 0.1
    activations = torch.randn(8, 10, 256) * 0.5  # (batch, seq_len, features)
    
    block_height = 16
    block_width = 16
    mantissa_bits = 3
    
    print(f"\nWeight shape: {weight.shape}")
    print(f"Activation shape: {activations.shape}")
    print(f"Block size: {block_height}x{block_width}")
    print(f"Block elements: {block_height * block_width}")
    print(f"Mantissa bits: {mantissa_bits}")
    
    # Test different top-K values
    top_k_values = [8, 16, 32, 64]
    
    print("\nTesting different top-K values:")
    print("-" * 70)
    
    # Baseline: BFP without AWQ
    bfp_only = block_floating_point_quantize_2d(
        weight, block_height, block_width, mantissa_bits
    )
    mse_bfp = torch.mean((weight - bfp_only) ** 2)
    print(f"Baseline (BFP only): MSE = {mse_bfp.item():.6e}")
    print()
    
    for top_k in top_k_values:
        quantized_awq = awq_quantize_2d(
            weight,
            activations,
            block_height=block_height,
            block_width=block_width,
            mantissa_bits=mantissa_bits,
            top_k=top_k
        )
        
        mse_awq = torch.mean((weight - quantized_awq) ** 2)
        improvement = ((mse_bfp - mse_awq) / mse_bfp * 100).item()
        
        print(f"AWQ with top-{top_k} salient weights per block:")
        print(f"  MSE: {mse_awq.item():.6e}")
        print(f"  Improvement over BFP: {improvement:.2f}%")
        print()


def example_4_mantissa_bits_effect():
    """Example 4: Effect of mantissa bits on quantization quality."""
    print("=" * 70)
    print("Example 4: Effect of Mantissa Bits")
    print("=" * 70)
    
    weight = torch.randn(256, 512) * 0.1
    block_height, block_width = 32, 16
    
    print(f"\nWeight shape: {weight.shape}")
    print(f"Block size: {block_height}x{block_width}")
    
    mantissa_bits_values = [2, 3, 4, 5, 6]
    
    print("\nComparing different mantissa bit counts:")
    print("-" * 70)
    
    for mantissa_bits in mantissa_bits_values:
        quantized = block_floating_point_quantize_2d(
            weight, block_height, block_width, mantissa_bits
        )
        
        mse = torch.mean((weight - quantized) ** 2)
        
        print(f"Mantissa bits: {mantissa_bits}")
        print(f"  MSE: {mse.item():.6e}")
        print(f"  Max error: {torch.max(torch.abs(weight - quantized)).item():.6f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("2D Block Quantization Examples")
    print("=" * 70)
    print("\nThis demonstrates the 2D block quantization implementation")
    print("for Block Floating Point (BFP) and Activation-Aware Weight")
    print("Quantization (AWQ) as requested in the problem statement.")
    print("\nKey features:")
    print("  - Weight matrices represented as 2D arrays")
    print("  - Divided into 2D blocks (e.g., 32x16)")
    print("  - Each block shares a scaling factor (BFP)")
    print("  - Top-K salient weights preserved per block (AWQ)")
    print("  - Block size is parameterized")
    
    try:
        example_1_basic_bfp()
        example_2_different_block_sizes()
        example_3_awq_with_salient_weights()
        example_4_mantissa_bits_effect()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully! ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
