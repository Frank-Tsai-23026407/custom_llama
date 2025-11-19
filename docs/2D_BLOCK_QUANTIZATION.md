# 2D Block Quantization for BFP and AWQ

This document describes the 2D block-based quantization implementation for Block Floating Point (BFP) and Activation-Aware Weight Quantization (AWQ).

## Overview

The 2D block quantization divides weight matrices into rectangular 2D blocks and applies quantization at the block level. This is more flexible and efficient than 1D flattening approaches, as it preserves the 2D structure of the weight matrices.

## Key Features

- **Parameterized Block Sizes**: Block dimensions (height × width) can be configured independently
- **2D Block Structure**: Maintains 2D structure instead of flattening to 1D
- **BFP Quantization**: Each block shares a scaling factor based on the maximum absolute value
- **AWQ Quantization**: Preserves top-K salient weights per block based on activation importance
- **Automatic Padding**: Handles non-divisible matrix dimensions automatically

## Usage

### 1. BFP Quantization with 2D Blocks

#### Python API

```python
from quantize_model_script.block_quantization_2d import block_floating_point_quantize_2d

# Create a weight matrix
weight = torch.randn(2048, 2048)

# Quantize with 32x16 blocks and 4 mantissa bits
quantized_weight = block_floating_point_quantize_2d(
    weight, 
    block_height=32, 
    block_width=16, 
    mantissa_bits=4
)
```

#### Command Line

```bash
# Quantize TinyLlama with 32x16 blocks
python quantize_model_script/bfp_quantize_2d.py \
    --model tinyllama \
    --block-height 32 \
    --block-width 16 \
    --mantissa-bits 3 4 5

# Dry run (don't save models)
python quantize_model_script/bfp_quantize_2d.py \
    --model tinyllama \
    --block-height 32 \
    --block-width 16 \
    --mantissa-bits 4 \
    --dry-run
```

### 2. AWQ Quantization with 2D Blocks

#### Python API

```python
from quantize_model_script.block_quantization_2d import awq_quantize_2d

# Create weight and activation matrices
weight = torch.randn(128, 256)
activations = torch.randn(4, 10, 256)  # (batch, seq_len, features)

# Quantize with AWQ
quantized_weight = awq_quantize_2d(
    weight,
    activations,
    block_height=16,
    block_width=16,
    mantissa_bits=3,
    top_k=16  # Preserve top 16 salient weights per block
)
```

#### Command Line

```bash
# Quantize with AWQ using calibration data
python quantize_model_script/awq_quantize_2d.py \
    --model tinyllama \
    --block-height 32 \
    --block-width 16 \
    --mantissa-bits 3 4 \
    --top-k 16 \
    --n-samples 128 \
    --seq-len 512
```

## Examples

### Example 1: Large Matrix Quantization

As mentioned in the problem statement, quantize a 2048×2048 matrix with 32×16 blocks:

```python
import torch
from quantize_model_script.block_quantization_2d import block_floating_point_quantize_2d

# Create a 2048×2048 weight matrix
weight = torch.randn(2048, 2048)

# Quantize with 32×16 blocks
# This creates 64×128 = 8192 blocks
quantized = block_floating_point_quantize_2d(
    weight,
    block_height=32,
    block_width=16,
    mantissa_bits=4
)

# Each block shares one scaling factor
# Total: 8192 scaling factors + quantized mantissas
```

### Example 2: Different Block Configurations

```python
# Square blocks
quantized = block_floating_point_quantize_2d(weight, 16, 16, mantissa_bits=4)

# Rectangular blocks
quantized = block_floating_point_quantize_2d(weight, 32, 16, mantissa_bits=4)

# Larger blocks (fewer scaling factors, more compression)
quantized = block_floating_point_quantize_2d(weight, 64, 64, mantissa_bits=4)
```

### Example 3: AWQ with Salient Weights

```python
import torch
from quantize_model_script.block_quantization_2d import awq_quantize_2d

weight = torch.randn(128, 256)
activations = torch.randn(8, 10, 256)

# Each 16×16 block (256 elements) preserves top 16 salient weights
quantized = awq_quantize_2d(
    weight, 
    activations,
    block_height=16,
    block_width=16,
    mantissa_bits=3,
    top_k=16
)

# Salient weights are identified based on activation importance
# They are preserved with full precision while others are quantized
```

## How It Works

### BFP 2D Quantization

1. **Divide into Blocks**: Split the weight matrix into 2D blocks of size `block_height × block_width`
2. **Calculate Scaling Factor**: For each block, find the maximum absolute value and compute the shared exponent
3. **Quantize Mantissas**: Normalize weights by the scaling factor and quantize to `mantissa_bits`
4. **Reconstruct**: Dequantize by multiplying mantissas with scaling factors
5. **Reshape**: Convert back to original matrix dimensions

### AWQ 2D Quantization

1. **Calculate Importance**: Compute per-feature importance from activations
2. **Weight Matrix**: Weight the weight matrix by activation importance
3. **Identify Salient Weights**: For each block, identify top-K weights with highest importance
4. **Quantize**: Apply BFP quantization to all weights
5. **Preserve Salient**: Replace quantized values with original values for salient weights
6. **Reconstruct**: Return the final quantized matrix

## Parameters

### BFP Parameters

- `block_height`: Height of each 2D block (e.g., 32)
- `block_width`: Width of each 2D block (e.g., 16)
- `mantissa_bits`: Number of bits for quantizing mantissas (2-6 typical)

### AWQ Parameters

- `block_height`, `block_width`, `mantissa_bits`: Same as BFP
- `top_k`: Number of salient weights to preserve per block (e.g., 16)

## Testing

Run the comprehensive test suite:

```bash
python quantize_model_script/test_block_quantization_2d.py
```

Tests include:
- Basic functionality
- Various block sizes
- Non-divisible dimensions
- AWQ with different top-K values
- Large matrices (2048×2048)
- Effect of mantissa bits

## Performance Considerations

### Block Size Selection

- **Smaller blocks** (e.g., 16×16):
  - More scaling factors to store
  - Better granularity, potentially lower quantization error
  - Higher memory overhead for scaling factors

- **Larger blocks** (e.g., 64×64):
  - Fewer scaling factors
  - More compression
  - Potentially higher quantization error within blocks

### Mantissa Bits

- **2-3 bits**: High compression, noticeable accuracy loss
- **4-5 bits**: Good balance between compression and accuracy
- **6+ bits**: Lower compression, better accuracy

### AWQ Top-K

- Higher `top_k` values preserve more weights → better accuracy but less compression
- Typical values: 8-32 per block
- Depends on block size: should be less than block size

## Comparison with 1D Quantization

| Aspect | 1D (Original) | 2D (New) |
|--------|---------------|----------|
| Structure | Flattens to 1D | Preserves 2D |
| Flexibility | Fixed block size | Independent H×W |
| Padding | 1D padding | 2D padding |
| Use Case | General | Better for matrices |

## Integration

The 2D quantization modules are designed to work alongside existing quantization tools:

```python
# Original 1D BFP
from quantize_model_script.block_quantization import block_floating_point_quantize

# New 2D BFP
from quantize_model_script.block_quantization_2d import block_floating_point_quantize_2d

# Both can be used in the same codebase
```

## References

- Block Floating Point (BFP): Quantization method that shares exponents within blocks
- Activation-Aware Weight Quantization (AWQ): Preserves salient weights based on activation importance
