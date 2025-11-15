import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantize_model_script.block_quantization import block_floating_point_quantize

def get_weight_scaling_factor(activation: torch.Tensor) -> torch.Tensor:
    """
    Calculates the scaling factor for weights based on activation magnitudes.

    Args:
        activation (torch.Tensor): The input activations.
        weight (torch.Tensor): The weight tensor.

    Returns:
        torch.Tensor: The scaling factor for the weights.
    """
    # The activation tensor has a shape of (batch, seq_len, in_features)
    # We need to calculate the average magnitude of each input feature.
    if activation.dim() == 3:
        # Reshape to (batch * seq_len, in_features)
        activation = activation.reshape(-1, activation.shape[-1])
    
    # Calculate the mean absolute value for each input feature dimension
    # This gives us a sense of the importance of each input channel.
    scales = torch.abs(activation).mean(dim=0)

    # Reshape scales for broadcasting with the weight matrix.
    # The weight matrix is (out_features, in_features). We want to scale each column (in_feature).
    # So, scales should be (1, in_features).
    scales = scales.view(1, -1)
    
    return scales

def awq_quantize(weight: torch.Tensor, activation: torch.Tensor, block_size: int, mantissa_bits: int) -> torch.Tensor:
    """
    Performs activation-aware weight quantization.

    Args:
        weight (torch.Tensor): The weight tensor.
        activation (torch.Tensor): The input activations.
        block_size (int): The block size for BFP quantization.
        mantissa_bits (int): The number of mantissa bits for BFP quantization.
        q_group_size (int): The quantization group size for AWQ.

    Returns:
        torch.Tensor: The quantized and de-quantized weight tensor.
    """
    # Add a small epsilon to prevent division by zero for stability
    epsilon = 1e-9

    # 1. Get scaling factors
    scales = get_weight_scaling_factor(activation)
    # NOTE: The line below was disabling AWQ by overriding scales with all ones
    # scales = torch.ones(scales.shape, device=weight.device)
    
    # Ensure scales are on the same device as weights
    scales = scales.to(weight.device)
    
    # 2. Scale down weights by dividing by activation-based scales
    # This reduces the magnitude of weights for important features, protecting them from quantization error.
    scaled_weight = weight * (scales + epsilon)
    
    # 3. Quantize the scaled weights
    quantized_weight = block_floating_point_quantize(scaled_weight, block_size, mantissa_bits)
    
    # 4. Scale the weights back up by multiplying with the scales
    de_scaled_weight = quantized_weight / (scales + epsilon)
    
    
    return de_scaled_weight