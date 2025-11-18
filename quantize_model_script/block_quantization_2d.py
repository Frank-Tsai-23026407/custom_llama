import torch
import math

def block_floating_point_quantize_2d(weight: torch.Tensor, block_height: int, block_width: int, mantissa_bits: int) -> torch.Tensor:
    """
    Quantizes a 2D weight matrix using Block Floating Point (BFP) quantization with 2D blocks.
    
    BFP quantizes the elements in each 2D block by finding the maximum absolute value
    (which determines a shared exponent for that block) and quantizing the mantissas.
    
    Args:
        weight (torch.Tensor): The input weight matrix (2D tensor, e.g., from a linear layer).
        block_height (int): The height of each 2D block.
        block_width (int): The width of each 2D block.
        mantissa_bits (int): The number of bits used to quantize the mantissa.
    
    Returns:
        torch.Tensor: The reconstructed (de-quantized) weight matrix.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {weight.dim()}D")
    
    original_shape = weight.shape
    out_features, in_features = original_shape
    
    # 1. Padding to ensure the tensor is divisible by the block size
    pad_height = (block_height - (out_features % block_height)) % block_height
    pad_width = (block_width - (in_features % block_width)) % block_width
    
    if pad_height > 0 or pad_width > 0:
        weight = torch.nn.functional.pad(weight, (0, pad_width, 0, pad_height))
    
    padded_height, padded_width = weight.shape
    num_blocks_height = padded_height // block_height
    num_blocks_width = padded_width // block_width
    
    # 2. Reshape into blocks
    # Reshape to (num_blocks_height, block_height, num_blocks_width, block_width)
    blocks = weight.reshape(num_blocks_height, block_height, num_blocks_width, block_width)
    # Rearrange to (num_blocks_height, num_blocks_width, block_height, block_width)
    blocks = blocks.permute(0, 2, 1, 3)
    # Reshape to (total_blocks, block_height * block_width)
    total_blocks = num_blocks_height * num_blocks_width
    blocks = blocks.reshape(total_blocks, block_height * block_width)
    
    # 3. Determine the shared exponent (E) for each block
    # Find the maximum absolute value in each block
    max_abs_values = torch.amax(torch.abs(blocks), dim=1, keepdim=True)
    
    # Calculate the exponent E: E = ceil(log2(max_abs_value))
    epsilon = 1e-9
    E = torch.ceil(torch.log2(max_abs_values + epsilon))
    
    # 4. Calculate the shared scaling factor (S) for each block
    S = torch.pow(2.0, E)
    
    # 5. Calculate the floating-point mantissa (M)
    M = blocks / S
    
    # 6. Quantize the mantissa (M)
    # Determine the number of quantization levels (Q_max) for the mantissa
    Q_max = (1 << (mantissa_bits - 1)) - 1
    
    # Linearly map the floating-point mantissa M to integer levels
    M_quant = torch.round(M * Q_max)
    
    # Clamp the quantized mantissa to the available integer range
    M_quant = torch.clamp(M_quant, -Q_max, Q_max)
    
    # 7. De-quantize (reconstruct) the weight
    M_dequant = M_quant / Q_max
    
    # Reconstruct the blocks
    reconstructed_blocks = M_dequant * S
    
    # 8. Reshape back to 2D array
    # Reshape to (num_blocks_height, num_blocks_width, block_height, block_width)
    reconstructed_blocks = reconstructed_blocks.reshape(num_blocks_height, num_blocks_width, block_height, block_width)
    # Permute to (num_blocks_height, block_height, num_blocks_width, block_width)
    reconstructed_blocks = reconstructed_blocks.permute(0, 2, 1, 3)
    # Reshape to (padded_height, padded_width)
    reconstructed_weight = reconstructed_blocks.reshape(padded_height, padded_width)
    
    # 9. Remove padding
    if pad_height > 0 or pad_width > 0:
        reconstructed_weight = reconstructed_weight[:out_features, :in_features]
    
    return reconstructed_weight


def awq_quantize_2d(weight: torch.Tensor, activation: torch.Tensor, block_height: int, block_width: int, mantissa_bits: int, top_k: int = 16) -> torch.Tensor:
    """
    Performs activation-aware weight quantization with 2D blocks.
    
    For each 2D block, identifies the top_k most salient weights based on activation magnitudes
    and applies different quantization strategies.
    
    Args:
        weight (torch.Tensor): The weight tensor (2D).
        activation (torch.Tensor): The input activations (batch, seq_len, in_features).
        block_height (int): The height of each 2D block.
        block_width (int): The width of each 2D block.
        mantissa_bits (int): The number of mantissa bits for BFP quantization.
        top_k (int): The number of salient weights to preserve per block (default: 16).
    
    Returns:
        torch.Tensor: The quantized and de-quantized weight tensor.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {weight.dim()}D")
    
    epsilon = 1e-9
    
    # 1. Calculate per-input-feature importance from activations
    if activation.dim() == 3:
        # Reshape to (batch * seq_len, in_features)
        activation = activation.reshape(-1, activation.shape[-1])
    
    # Calculate the mean absolute value for each input feature dimension
    importance_scores = torch.abs(activation).mean(dim=0)  # Shape: (in_features,)
    
    # Ensure importance scores are on the same device as weights
    importance_scores = importance_scores.to(weight.device)
    
    # 2. Weight the weight matrix by importance
    # Weight matrix is (out_features, in_features)
    # Broadcast importance_scores across rows
    weighted_weight = weight * importance_scores.unsqueeze(0)
    
    # 3. Apply 2D block quantization
    original_shape = weight.shape
    out_features, in_features = original_shape
    
    # Padding
    pad_height = (block_height - (out_features % block_height)) % block_height
    pad_width = (block_width - (in_features % block_width)) % block_width
    
    if pad_height > 0 or pad_width > 0:
        weighted_weight = torch.nn.functional.pad(weighted_weight, (0, pad_width, 0, pad_height))
        weight_padded = torch.nn.functional.pad(weight, (0, pad_width, 0, pad_height))
    else:
        weight_padded = weight
    
    padded_height, padded_width = weighted_weight.shape
    num_blocks_height = padded_height // block_height
    num_blocks_width = padded_width // block_width
    
    # Reshape into blocks
    weighted_blocks = weighted_weight.reshape(num_blocks_height, block_height, num_blocks_width, block_width)
    weighted_blocks = weighted_blocks.permute(0, 2, 1, 3)
    total_blocks = num_blocks_height * num_blocks_width
    weighted_blocks = weighted_blocks.reshape(total_blocks, block_height * block_width)
    
    weight_blocks = weight_padded.reshape(num_blocks_height, block_height, num_blocks_width, block_width)
    weight_blocks = weight_blocks.permute(0, 2, 1, 3)
    weight_blocks = weight_blocks.reshape(total_blocks, block_height * block_width)
    
    # 4. For each block, identify top_k salient weights
    # Calculate salience: absolute value of weighted weights
    salience = torch.abs(weighted_blocks)  # Shape: (total_blocks, block_size)
    
    # Get top_k indices for each block
    block_size = block_height * block_width
    k = min(top_k, block_size)  # Ensure k doesn't exceed block size
    
    # Get top-k values and indices
    topk_salience, topk_indices = torch.topk(salience, k, dim=1)
    
    # 5. Apply standard BFP quantization to all weights
    # Determine the shared exponent (E) for each block
    max_abs_values = torch.amax(torch.abs(weight_blocks), dim=1, keepdim=True)
    E = torch.ceil(torch.log2(max_abs_values + epsilon))
    S = torch.pow(2.0, E)
    
    M = weight_blocks / S
    Q_max = (1 << (mantissa_bits - 1)) - 1
    M_quant = torch.round(M * Q_max)
    M_quant = torch.clamp(M_quant, -Q_max, Q_max)
    M_dequant = M_quant / Q_max
    reconstructed_blocks = M_dequant * S
    
    # 6. For salient weights, preserve with higher precision (skip quantization)
    # Create a mask for the top-k salient weights
    mask = torch.zeros_like(weight_blocks, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    
    # Replace quantized values with original values for salient weights
    reconstructed_blocks = torch.where(mask, weight_blocks, reconstructed_blocks)
    
    # 7. Reshape back to 2D array
    reconstructed_blocks = reconstructed_blocks.reshape(num_blocks_height, num_blocks_width, block_height, block_width)
    reconstructed_blocks = reconstructed_blocks.permute(0, 2, 1, 3)
    reconstructed_weight = reconstructed_blocks.reshape(padded_height, padded_width)
    
    # 8. Remove padding
    if pad_height > 0 or pad_width > 0:
        reconstructed_weight = reconstructed_weight[:out_features, :in_features]
    
    return reconstructed_weight


if __name__ == '__main__':
    print("=" * 70)
    print("2D Block Floating Point Quantization Example")
    print("=" * 70)
    
    # Example 1: BFP with 2D blocks
    print("\n1. Testing BFP with 2D blocks (32x16 blocks on 2048x2048 matrix)")
    print("-" * 70)
    
    # Create a 2048x2048 weight matrix
    weight_matrix = torch.randn(2048, 2048) * 0.1
    
    # Define 2D block size
    BLOCK_HEIGHT = 32
    BLOCK_WIDTH = 16
    MANTISSA_BITS = 4
    
    print(f"Original Weight Shape: {weight_matrix.shape}")
    print(f"Block Size: {BLOCK_HEIGHT}x{BLOCK_WIDTH}")
    print(f"Mantissa Bits: {MANTISSA_BITS}")
    print(f"Number of blocks: {(2048//BLOCK_HEIGHT)}x{(2048//BLOCK_WIDTH)} = {(2048//BLOCK_HEIGHT)*(2048//BLOCK_WIDTH)}")
    
    # Apply BFP quantization
    quantized_weight = block_floating_point_quantize_2d(
        weight_matrix, 
        block_height=BLOCK_HEIGHT,
        block_width=BLOCK_WIDTH,
        mantissa_bits=MANTISSA_BITS
    )
    
    # Calculate MSE
    mse = torch.mean((weight_matrix - quantized_weight) ** 2)
    print(f"\nMean Squared Error (MSE): {mse.item():.6e}")
    
    # Show statistics of one block
    block_i, block_j = 0, 0
    start_h = block_i * BLOCK_HEIGHT
    end_h = start_h + BLOCK_HEIGHT
    start_w = block_j * BLOCK_WIDTH
    end_w = start_w + BLOCK_WIDTH
    
    original_block = weight_matrix[start_h:end_h, start_w:end_w]
    quantized_block = quantized_weight[start_h:end_h, start_w:end_w]
    
    print(f"\nBlock [{block_i}, {block_j}] Analysis:")
    print(f"  Original block shape: {original_block.shape}")
    print(f"  Max absolute value: {torch.max(torch.abs(original_block)).item():.4f}")
    print(f"  Block MSE: {torch.mean((original_block - quantized_block) ** 2).item():.6e}")
    
    # Example 2: AWQ with 2D blocks
    print("\n" + "=" * 70)
    print("2. Testing AWQ with 2D blocks")
    print("=" * 70)
    
    # Create smaller matrices for demonstration
    small_weight = torch.randn(128, 256) * 0.1
    # Simulate activations (batch_size=8, seq_len=10, in_features=256)
    activations = torch.randn(8, 10, 256) * 0.5
    
    BLOCK_HEIGHT = 16
    BLOCK_WIDTH = 16
    MANTISSA_BITS = 3
    TOP_K = 16  # Number of salient weights per block
    
    print(f"\nWeight Shape: {small_weight.shape}")
    print(f"Activation Shape: {activations.shape}")
    print(f"Block Size: {BLOCK_HEIGHT}x{BLOCK_WIDTH}")
    print(f"Mantissa Bits: {MANTISSA_BITS}")
    print(f"Top-K Salient Weights per Block: {TOP_K}")
    
    # Apply AWQ quantization
    quantized_awq = awq_quantize_2d(
        small_weight,
        activations,
        block_height=BLOCK_HEIGHT,
        block_width=BLOCK_WIDTH,
        mantissa_bits=MANTISSA_BITS,
        top_k=TOP_K
    )
    
    # Calculate MSE
    mse_awq = torch.mean((small_weight - quantized_awq) ** 2)
    print(f"\nMean Squared Error (MSE) with AWQ: {mse_awq.item():.6e}")
    
    # Compare with BFP only (without AWQ)
    quantized_bfp = block_floating_point_quantize_2d(
        small_weight,
        block_height=BLOCK_HEIGHT,
        block_width=BLOCK_WIDTH,
        mantissa_bits=MANTISSA_BITS
    )
    mse_bfp = torch.mean((small_weight - quantized_bfp) ** 2)
    print(f"Mean Squared Error (MSE) with BFP only: {mse_bfp.item():.6e}")
    print(f"AWQ Improvement: {((mse_bfp - mse_awq) / mse_bfp * 100).item():.2f}%")
    
    print("\n" + "=" * 70)
    print("Testing complete!")
    print("=" * 70)
