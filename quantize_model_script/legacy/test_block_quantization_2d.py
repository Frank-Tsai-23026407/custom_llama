"""
Test suite for 2D block quantization functions.
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


def test_bfp_2d_basic():
    """Test basic BFP 2D quantization functionality."""
    print("Test 1: Basic BFP 2D Quantization")
    print("-" * 50)
    
    # Create a simple weight matrix
    weight = torch.randn(64, 128) * 0.1
    block_height, block_width = 8, 16
    mantissa_bits = 4
    
    # Apply quantization
    quantized = block_floating_point_quantize_2d(
        weight, block_height, block_width, mantissa_bits
    )
    
    # Check shape is preserved
    assert weight.shape == quantized.shape, "Shape mismatch!"
    
    # Check quantization error is reasonable
    mse = torch.mean((weight - quantized) ** 2)
    assert mse < 1e-2, f"MSE too high: {mse}"
    
    print(f"  Original shape: {weight.shape}")
    print(f"  Quantized shape: {quantized.shape}")
    print(f"  MSE: {mse.item():.6e}")
    print("  ✓ Test passed!\n")


def test_bfp_2d_parameterized_blocks():
    """Test BFP 2D with different block sizes."""
    print("Test 2: Parameterized Block Sizes")
    print("-" * 50)
    
    weight = torch.randn(256, 512) * 0.1
    
    test_cases = [
        (16, 16),  # Square blocks
        (32, 16),  # Rectangular blocks (example from problem statement)
        (8, 32),   # Different rectangular blocks
        (64, 64),  # Larger square blocks
    ]
    
    for block_h, block_w in test_cases:
        quantized = block_floating_point_quantize_2d(
            weight, block_h, block_w, mantissa_bits=4
        )
        
        assert weight.shape == quantized.shape, f"Shape mismatch for {block_h}x{block_w}!"
        mse = torch.mean((weight - quantized) ** 2)
        
        num_blocks_h = (weight.shape[0] + block_h - 1) // block_h
        num_blocks_w = (weight.shape[1] + block_w - 1) // block_w
        
        print(f"  Block size {block_h}x{block_w}:")
        print(f"    Number of blocks: {num_blocks_h}x{num_blocks_w}")
        print(f"    MSE: {mse.item():.6e}")
    
    print("  ✓ All block sizes passed!\n")


def test_bfp_2d_non_divisible():
    """Test BFP 2D with matrix dimensions not divisible by block size."""
    print("Test 3: Non-Divisible Matrix Dimensions")
    print("-" * 50)
    
    # Create weight with odd dimensions
    weight = torch.randn(100, 250) * 0.1
    block_height, block_width = 32, 16
    
    quantized = block_floating_point_quantize_2d(
        weight, block_height, block_width, mantissa_bits=4
    )
    
    # Check shape is preserved even with padding
    assert weight.shape == quantized.shape, "Shape mismatch with non-divisible dimensions!"
    
    mse = torch.mean((weight - quantized) ** 2)
    print(f"  Matrix size: {weight.shape}")
    print(f"  Block size: {block_height}x{block_width}")
    print(f"  MSE: {mse.item():.6e}")
    print("  ✓ Test passed!\n")


def test_awq_2d_basic():
    """Test basic AWQ 2D quantization functionality."""
    print("Test 4: Basic AWQ 2D Quantization")
    print("-" * 50)
    
    # Create weight and activation matrices
    weight = torch.randn(64, 128) * 0.1
    activations = torch.randn(4, 10, 128) * 0.5  # batch=4, seq_len=10, features=128
    
    block_height, block_width = 16, 16
    mantissa_bits = 3
    top_k = 16
    
    # Apply AWQ quantization
    quantized = awq_quantize_2d(
        weight, activations, block_height, block_width, mantissa_bits, top_k
    )
    
    # Check shape is preserved
    assert weight.shape == quantized.shape, "Shape mismatch in AWQ!"
    
    # AWQ should preserve important weights better than plain BFP
    mse_awq = torch.mean((weight - quantized) ** 2)
    
    # Compare with plain BFP
    quantized_bfp = block_floating_point_quantize_2d(
        weight, block_height, block_width, mantissa_bits
    )
    mse_bfp = torch.mean((weight - quantized_bfp) ** 2)
    
    print(f"  Weight shape: {weight.shape}")
    print(f"  Block size: {block_height}x{block_width}")
    print(f"  Top-K per block: {top_k}")
    print(f"  MSE (AWQ): {mse_awq.item():.6e}")
    print(f"  MSE (BFP): {mse_bfp.item():.6e}")
    print(f"  AWQ Improvement: {((mse_bfp - mse_awq) / mse_bfp * 100).item():.2f}%")
    print("  ✓ Test passed!\n")


def test_awq_2d_different_top_k():
    """Test AWQ 2D with different top-k values."""
    print("Test 5: AWQ with Different Top-K Values")
    print("-" * 50)
    
    weight = torch.randn(128, 256) * 0.1
    activations = torch.randn(4, 8, 256) * 0.5
    
    block_height, block_width = 16, 16
    mantissa_bits = 3
    
    top_k_values = [8, 16, 32, 64]
    
    for top_k in top_k_values:
        quantized = awq_quantize_2d(
            weight, activations, block_height, block_width, mantissa_bits, top_k
        )
        
        mse = torch.mean((weight - quantized) ** 2)
        print(f"  Top-K={top_k}: MSE={mse.item():.6e}")
    
    print("  ✓ All top-k values tested!\n")


def test_large_matrix():
    """Test with large matrix (2048x2048) as mentioned in problem statement."""
    print("Test 6: Large Matrix (2048x2048)")
    print("-" * 50)
    
    # Create a 2048x2048 matrix as mentioned in the problem
    weight = torch.randn(2048, 2048) * 0.1
    
    # Use 32x16 blocks as mentioned in the problem
    block_height, block_width = 32, 16
    mantissa_bits = 4
    
    quantized = block_floating_point_quantize_2d(
        weight, block_height, block_width, mantissa_bits
    )
    
    assert weight.shape == quantized.shape, "Shape mismatch!"
    
    num_blocks_h = 2048 // block_height
    num_blocks_w = 2048 // block_width
    
    mse = torch.mean((weight - quantized) ** 2)
    
    print(f"  Matrix size: {weight.shape}")
    print(f"  Block size: {block_height}x{block_width}")
    print(f"  Number of blocks: {num_blocks_h}x{num_blocks_w} = {num_blocks_h * num_blocks_w}")
    print(f"  MSE: {mse.item():.6e}")
    print("  ✓ Test passed!\n")


def test_mantissa_bits_effect():
    """Test the effect of different mantissa bit counts."""
    print("Test 7: Effect of Mantissa Bits")
    print("-" * 50)
    
    weight = torch.randn(128, 256) * 0.1
    block_height, block_width = 16, 16
    
    mantissa_bits_values = [2, 3, 4, 5, 6]
    
    print(f"  Matrix size: {weight.shape}")
    print(f"  Block size: {block_height}x{block_width}")
    
    for mantissa_bits in mantissa_bits_values:
        quantized = block_floating_point_quantize_2d(
            weight, block_height, block_width, mantissa_bits
        )
        
        mse = torch.mean((weight - quantized) ** 2)
        print(f"  Mantissa bits={mantissa_bits}: MSE={mse.item():.6e}")
    
    print("  ✓ Test passed!\n")


def run_all_tests():
    """Run all test cases."""
    print("=" * 70)
    print("Running 2D Block Quantization Test Suite")
    print("=" * 70)
    print()
    
    try:
        test_bfp_2d_basic()
        test_bfp_2d_parameterized_blocks()
        test_bfp_2d_non_divisible()
        test_awq_2d_basic()
        test_awq_2d_different_top_k()
        test_large_matrix()
        test_mantissa_bits_effect()
        
        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        return True
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
