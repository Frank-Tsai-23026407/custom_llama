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


def block_floating_point_quantize(weight: torch.Tensor, block_size: int, mantissa_bits: int) -> torch.Tensor:
    """Quantizes a weight tensor using the Block Floating-Point (BFP) format.

    BFP is a quantization scheme that groups tensor elements into blocks. Within each
    block, a single, shared exponent is used for all elements, while each element
    retains its own quantized mantissa. This approach provides a good balance between
    compression and numerical precision, especially for tensors with a wide dynamic range.

    The quantization process for each block is as follows:
    1.  Find the maximum absolute value in the block.
    2.  Calculate a shared exponent `E` based on this maximum value.
    3.  Compute a scaling factor `S = 2^E`.
    4.  Normalize the block elements by dividing by `S` to get floating-point mantissas.
    5.  Linearly quantize these mantissas to a fixed number of bits.
    6.  De-quantize the mantissas and multiply by `S` to reconstruct the block.

    Args:
        weight (torch.Tensor): The input weight tensor to be quantized.
        block_size (int): The number of elements to group into each quantization block.
        mantissa_bits (int): The number of bits to use for representing the mantissa.
            The sign bit is handled separately, so `mantissa_bits=4` would allow for
            `2^(4-1)` positive levels.

    Returns:
        torch.Tensor: The de-quantized weight tensor with the same shape as the input.
    """
    original_shape = weight.shape
    # Flatten the weight tensor for block processing
    flat_weight = weight.flatten()
    num_elements = flat_weight.numel()

    # 1. Padding to ensure the tensor is divisible by the block size
    padding_needed = (block_size - (num_elements % block_size)) % block_size
    if padding_needed > 0:
        flat_weight = torch.nn.functional.pad(flat_weight, (0, padding_needed))

    num_blocks = flat_weight.numel() // block_size
    
    # Reshape into blocks
    blocks = flat_weight.view(num_blocks, block_size)

    # 2. Determine the shared exponent (E) for each block
    
    # Find the maximum absolute value in each block (the characteristic of the block)
    max_abs_values = torch.amax(torch.abs(blocks), dim=1, keepdim=True)
    
    # Calculate the exponent E: E = ceil(log2(max_abs_value))
    # E will be the smallest integer such that 2^E >= max_abs_value
    # We add a small epsilon to handle log2(0) if the block is all zeros
    epsilon = 1e-9 
    
    # Use torch.ceil(torch.log2(x)) to find the exponent E
    # Note: torch.log2(0) is -inf, but since we used max_abs_values, only zero blocks are an issue, 
    # which the epsilon handles.
    E = torch.ceil(torch.log2(max_abs_values + epsilon))

    # 3. Calculate the shared scaling factor (S) for each block
    # S = 2^E. This is used to normalize the block elements into the mantissa range [-1, 1).
    S = torch.pow(2.0, E)

    # 4. Calculate the floating-point mantissa (M)
    M = blocks / S
    
    # 5. Quantize the mantissa (M)
    
    # Determine the number of quantization levels (Q_max) for the mantissa
    # Symmetric quantization: levels from -Q_max to Q_max
    Q_max = (1 << (mantissa_bits - 1)) - 1
    
    # Linearly map the floating-point mantissa M (range ~[-1, 1)) to integer levels
    # M_quant = round(M * Q_max)
    M_quant = torch.round(M * Q_max)
    
    # Clamp the quantized mantissa to the available integer range
    M_quant = torch.clamp(M_quant, -Q_max, Q_max)
    
    # 6. De-quantize (reconstruct) the weight
    
    # M_dequant = M_quant / Q_max
    M_dequant = M_quant / Q_max
    
    # Reconstruct the block: W_approx = M_dequant * S
    reconstructed_blocks = M_dequant * S
    
    # 7. Reshape and remove padding
    reconstructed_flat = reconstructed_blocks.flatten()
    
    # Remove the padding added earlier
    if padding_needed > 0:
        reconstructed_flat = reconstructed_flat[:-padding_needed]
        
    # Reshape back to the original matrix shape
    reconstructed_weight = reconstructed_flat.view(original_shape)

    return reconstructed_weight


def awq_mix_precision_quantize_2d(weight: torch.Tensor, activation: torch.Tensor, block_height: int, block_width: int, mantissa_bits: int, top_k: int = None) -> torch.Tensor:
    """
    Performs mix-precision activation-aware weight quantization with 2D blocks.
    
    Mix-precision: Identifies the top_k most salient weights based on activation magnitudes
    and preserves them in full floating-point (FP32) precision. The remaining weights are
    quantized using standard BFP.
    
    Top-K Selection:
        - Per 2D block: Select top-k individual ENTRIES (elements) with highest salience
        - NOT per-row or per-column, but within the entire flattened block
        - These k entries can be scattered across different rows/columns in the block
        - Default: top_k = block_width (one full input channel row per block)
        - Example: For a 16×16 block, default top_k=16 (one row = 6.25% of 256 elements)
    
    Args:
        weight (torch.Tensor): The weight tensor (2D).
        activation (torch.Tensor): The input activations (batch, seq_len, in_features).
        block_height (int): The height of each 2D block. (output channels)
        block_width (int): The width of each 2D block. (input channels)
        mantissa_bits (int): The number of mantissa bits for BFP quantization.
        top_k (int, optional): The number of salient weight ENTRIES to preserve in FP32 per block.
                               These entries can be in different rows/columns within the block.
                               If None (default), uses block_width (one full input channel row).
    
    Returns:
        torch.Tensor: The quantized weight tensor with mix-precision (FP32 + BFP).
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {weight.dim()}D")
    
    # Set default top_k to block_width (one input channel row per block)
    if top_k is None:
        top_k = block_width
    
    epsilon = 1e-9
    
    # 1. Calculate per-input-feature importance from activations
    if activation.dim() == 3:
        # Reshape to (batch * seq_len, in_features)
        activation = activation.reshape(-1, activation.shape[-1])
    
    # Calculate the mean absolute value for each input feature dimension
    importance_scores = torch.abs(activation).mean(dim=0)  # Shape: (in_features,)
    
    # Ensure importance scores are on the same device as weights
    importance_scores = importance_scores.to(weight.device)
    
    # 2. Apply 2D block quantization
    original_shape = weight.shape
    out_features, in_features = original_shape
    
    # Padding
    pad_height = (block_height - (out_features % block_height)) % block_height
    pad_width = (block_width - (in_features % block_width)) % block_width
    
    if pad_height > 0 or pad_width > 0:
        weight_padded = torch.nn.functional.pad(weight, (0, pad_width, 0, pad_height))
    else:
        weight_padded = weight
    
    padded_height, padded_width = weight_padded.shape
    num_blocks_height = padded_height // block_height
    num_blocks_width = padded_width // block_width
    
    # Reshape into blocks
    total_blocks = num_blocks_height * num_blocks_width
    
    # weighted_blocks = weighted_weight.reshape(num_blocks_height, block_height, num_blocks_width, block_width)
    # weighted_blocks = weighted_blocks.permute(0, 2, 1, 3)
    # weighted_blocks = weighted_blocks.reshape(total_blocks, block_height * block_width)
    
    weight_blocks = weight_padded.reshape(num_blocks_height, block_height, num_blocks_width, block_width)
    weight_blocks = weight_blocks.permute(0, 2, 1, 3)
    weight_blocks = weight_blocks.reshape(total_blocks, block_height * block_width)
    
    # 4. For each block, identify top_k salient weights
    # Salience should only consider the magnitude of activation (importance scores),
    # not the weighted product. We need to broadcast importance_scores to match block structure.
    
    # Create importance score blocks (same structure as weight blocks)
    # Pad importance_scores if needed, then broadcast to full height
    if pad_width > 0:
        importance_padded = torch.nn.functional.pad(importance_scores.unsqueeze(0), (0, pad_width))
    else:
        importance_padded = importance_scores.unsqueeze(0)
    
    # Expand to match padded height
    importance_padded = importance_padded.expand(padded_height, -1)
    
    importance_blocks = importance_padded.reshape(num_blocks_height, block_height, num_blocks_width, block_width)
    importance_blocks = importance_blocks.permute(0, 2, 1, 3)
    importance_blocks = importance_blocks.reshape(total_blocks, block_height * block_width)
    
    # Salience is based on activation magnitude only
    salience = torch.abs(importance_blocks)  # Shape: (total_blocks, block_size)
    
    # Get top_k indices for each block
    # IMPORTANT: top_k is applied per block, selecting k individual ENTRIES
    # within the flattened block (not per row or per column)
    block_size = block_height * block_width
    k = min(top_k, block_size)  # Ensure k doesn't exceed block size
    
    # Get top-k values and indices (dim=1 means per-block)
    # topk_indices shape: (total_blocks, k)
    # Each row contains k indices into the flattened block
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


def awq_fix_precision_quantize_2d(weight: torch.Tensor, activation: torch.Tensor, block_height: int, block_width: int, mantissa_bits: int) -> torch.Tensor:
    """
    Performs fix-precision activation-aware weight quantization with 2D blocks.
    
    Fix-precision: Uses scaling factors based on activation magnitudes to reduce quantization
    error for salient weights. All weights (including salient ones) are still represented in
    BFP format, but the scaling protects important weights during quantization.
    
    The process:
    1. Calculate per-input-feature importance from activations
    2. Scale weights UP by importance (W_scaled = W * s)
    3. Apply standard BFP quantization to scaled weights
    4. Scale quantized weights back DOWN (W_final = Q(W_scaled) / s)
    
    This effectively allocates more quantization precision to important weights.
    
    Args:
        weight (torch.Tensor): The weight tensor (2D).
        activation (torch.Tensor): The input activations (batch, seq_len, in_features).
        block_height (int): The height of each 2D block.
        block_width (int): The width of each 2D block.
        mantissa_bits (int): The number of mantissa bits for BFP quantization.
    
    Returns:
        torch.Tensor: The quantized and de-quantized weight tensor (all in BFP).
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {weight.dim()}D")
    
    epsilon = 1e-9
    
    # 1. Calculate per-input-feature importance from activations
    if activation.dim() == 3:
        activation = activation.reshape(-1, activation.shape[-1])
    
    # Calculate the mean absolute value for each input feature dimension
    importance_scores = torch.abs(activation).mean(dim=0)  # Shape: (in_features,)
    
    # Ensure importance scores are on the same device as weights
    importance_scores = importance_scores.to(weight.device)
    
    # 2. Scale weights UP by importance (W_scaled = W * s)
    # Weight matrix is (out_features, in_features)
    # Broadcast importance_scores across rows
    scaled_weight = weight * importance_scores.unsqueeze(0)
    
    # 3. Apply standard 2D BFP quantization to scaled weights
    quantized_scaled_weight = block_floating_point_quantize_2d(
        scaled_weight,
        block_height=block_height,
        block_width=block_width,
        mantissa_bits=mantissa_bits
    )
    
    # 4. Scale quantized weights back DOWN (W_final = Q(W_scaled) / s)
    reconstructed_weight = quantized_scaled_weight / (importance_scores.unsqueeze(0) + epsilon)
    
    return reconstructed_weight


# Backward compatibility aliases
awq_quantize_2d = awq_mix_precision_quantize_2d


if __name__ == '__main__':
    print("=" * 80)
    print("Block Floating Point Quantization - Unified Implementation")
    print("=" * 80)
    
    # --- Example 1: 1D BFP Quantization ---
    print("\n[1] 1D BFP Quantization Example")
    print("-" * 80)
    
    BLOCK_SIZE = 16
    MANTISSA_BITS = 5
    
    W_orig = torch.randn(4, 32) * 0.1
    W_orig[:, 20:] *= 10.0
    W_orig[0, :] = torch.arange(32) / 10.0
    
    print(f"Original Weight Tensor Shape: {W_orig.shape}")
    print(f"BFP Parameters: Block Size={BLOCK_SIZE}, Mantissa Bits={MANTISSA_BITS}")
    
    W_quantized = block_floating_point_quantize(W_orig, BLOCK_SIZE, MANTISSA_BITS)
    mse = torch.mean((W_orig - W_quantized) ** 2)
    print(f"Mean Squared Error (MSE) after BFP: {mse.item():.6e}")
    
    # --- Example 2: 2D BFP Quantization ---
    print("\n" + "=" * 80)
    print("[2] 2D BFP Quantization Example")
    print("-" * 80)
    
    weight_matrix = torch.randn(2048, 2048) * 0.1
    BLOCK_HEIGHT = 32
    BLOCK_WIDTH = 16
    MANTISSA_BITS = 4
    
    print(f"Original Weight Shape: {weight_matrix.shape}")
    print(f"Block Size: {BLOCK_HEIGHT}x{BLOCK_WIDTH}")
    print(f"Mantissa Bits: {MANTISSA_BITS}")
    print(f"Number of blocks: {(2048//BLOCK_HEIGHT)}x{(2048//BLOCK_WIDTH)} = {(2048//BLOCK_HEIGHT)*(2048//BLOCK_WIDTH)}")
    
    quantized_weight = block_floating_point_quantize_2d(
        weight_matrix, 
        block_height=BLOCK_HEIGHT,
        block_width=BLOCK_WIDTH,
        mantissa_bits=MANTISSA_BITS
    )
    
    mse = torch.mean((weight_matrix - quantized_weight) ** 2)
    print(f"Mean Squared Error (MSE): {mse.item():.6e}")
    
    # --- Example 3: AWQ with 2D blocks (Fix-Precision vs Mix-Precision) ---
    print("\n" + "=" * 80)
    print("[3] AWQ with 2D Blocks (Fix-Precision vs Mix-Precision)")
    print("-" * 80)
    
    small_weight = torch.randn(128, 256) * 0.1
    activations = torch.randn(8, 10, 256) * 0.5
    
    BLOCK_HEIGHT = 16
    BLOCK_WIDTH = 16
    MANTISSA_BITS = 3
    TOP_K = 16
    
    print(f"Weight Shape: {small_weight.shape}")
    print(f"Activation Shape: {activations.shape}")
    print(f"Block Size: {BLOCK_HEIGHT}x{BLOCK_WIDTH}")
    print(f"Mantissa Bits: {MANTISSA_BITS}")
    
    # Baseline: BFP only
    quantized_bfp = block_floating_point_quantize_2d(
        small_weight,
        block_height=BLOCK_HEIGHT,
        block_width=BLOCK_WIDTH,
        mantissa_bits=MANTISSA_BITS
    )
    mse_bfp = torch.mean((small_weight - quantized_bfp) ** 2)
    
    # AWQ Fix-Precision
    print(f"\n--- Fix-Precision AWQ (scaling-based, all BFP) ---")
    quantized_awq_fix = awq_fix_precision_quantize_2d(
        small_weight,
        activations,
        block_height=BLOCK_HEIGHT,
        block_width=BLOCK_WIDTH,
        mantissa_bits=MANTISSA_BITS
    )
    mse_awq_fix = torch.mean((small_weight - quantized_awq_fix) ** 2)
    
    # AWQ Mix-Precision
    print(f"\n--- Mix-Precision AWQ (top-{TOP_K} in FP32, rest BFP) ---")
    quantized_awq_mix = awq_mix_precision_quantize_2d(
        small_weight,
        activations,
        block_height=BLOCK_HEIGHT,
        block_width=BLOCK_WIDTH,
        mantissa_bits=MANTISSA_BITS,
        top_k=TOP_K
    )
    mse_awq_mix = torch.mean((small_weight - quantized_awq_mix) ** 2)
    
    # Print comparison
    print("\n" + "=" * 80)
    print("Quantization Method Comparison:")
    print("=" * 80)
    print(f"BFP only (baseline):           MSE = {mse_bfp.item():.6e}")
    print(f"AWQ Fix-Precision (scaling):   MSE = {mse_awq_fix.item():.6e}  ({((mse_bfp - mse_awq_fix) / mse_bfp * 100).item():+.2f}%)")
    print(f"AWQ Mix-Precision (FP32+BFP):  MSE = {mse_awq_mix.item():.6e}  ({((mse_bfp - mse_awq_mix) / mse_bfp * 100).item():+.2f}%)")
    print("=" * 80)
    print("Note:")
    print("  - Fix-Precision: All weights in BFP, uses scaling to protect salient weights")
    print("  - Mix-Precision: Top-K salient weights in FP32, rest in BFP")
    print("\n" + "=" * 80)
    print("All tests complete!")
    print("=" * 80)