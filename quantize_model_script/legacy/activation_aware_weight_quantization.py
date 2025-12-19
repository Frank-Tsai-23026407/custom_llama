import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantize_model_script.block_quantization import block_floating_point_quantize

def get_weight_scaling_factor(activation: torch.Tensor) -> torch.Tensor:
    """Calculates weight scaling factors based on activation magnitudes.

    This function implements a core component of Activation-aware Weight Quantization (AWQ).
    It computes a per-channel scaling factor by taking the mean absolute value of the
    input activations. Channels with larger activations are considered more important,
    and their corresponding weights will be scaled down to protect them from quantization
    errors.

    Args:
        activation (torch.Tensor): The input activation tensor, expected to have a shape
            of (batch_size, sequence_length, in_features) or (batch_size, in_features).

    Returns:
        torch.Tensor: A tensor of scaling factors with shape (1, in_features), ready
            to be broadcasted with a weight matrix.
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
    """Performs activation-aware weight quantization on a weight tensor.

    This function applies the AWQ method by first scaling the weights based on the
    provided activations, then quantizing the scaled weights using the Block
    Floating-Point (BFP) method, and finally de-scaling them back. This process
    selectively reduces quantization error for weights that are multiplied by
    large activation values.

    The process is as follows:
    1.  Calculate scaling factors from the activations.
    2.  Scale down the weights by multiplying them with the scaling factors.
    3.  Apply BFP quantization to the scaled weights.
    4.  Scale the quantized weights back up by dividing by the scaling factors.

    Args:
        weight (torch.Tensor): The weight tensor to be quantized.
        activation (torch.Tensor): The activation tensor used to determine the scaling factors.
        block_size (int): The block size to be used for BFP quantization.
        mantissa_bits (int): The number of mantissa bits for BFP quantization.

    Returns:
        torch.Tensor: The de-quantized weight tensor after applying AWQ.
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