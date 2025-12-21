"""
Runtime Quantization Wrapper for TinyLlama

This module applies quantization on-the-fly during inference without saving quantized models.
This significantly reduces disk space usage.

Usage:
    from runtime_quantizer import apply_runtime_quantization
    
    model = AutoModelForCausalLM.from_pretrained("model_path")
    
    # Apply BFP quantization at runtime
    apply_runtime_quantization(
        model, 
        method="bfp",
        block_height=16, 
        block_width=16, 
        mantissa_bits=4
    )
    
    # Model is now quantized in-memory, ready for inference
    outputs = model.generate(...)
"""

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_field.block_quantization_2d import (
    block_floating_point_quantize_2d,
    awq_fix_precision_quantize_2d,
    awq_mix_precision_quantize_2d
)


class QuantizationConfig:
    """Configuration for runtime quantization."""
    
    def __init__(self, method="bfp", block_height=16, block_width=16, 
                 mantissa_bits=4, top_k=16, activations=None):
        """
        Args:
            method: Quantization method ("bfp", "awq-fix", "awq-mix")
            block_height: Height of 2D blocks
            block_width: Width of 2D blocks
            mantissa_bits: Number of mantissa bits for BFP
            top_k: Number of salient weights for mix-precision (only for awq-mix)
            activations: Pre-collected activations (only for AWQ methods)
        """
        self.method = method
        self.block_height = block_height
        self.block_width = block_width
        self.mantissa_bits = mantissa_bits
        self.top_k = top_k
        self.activations = activations
    
    def __repr__(self):
        return (f"QuantizationConfig(method={self.method}, "
                f"block={self.block_height}x{self.block_width}, "
                f"mantissa={self.mantissa_bits}, top_k={self.top_k})")


def apply_runtime_quantization(model, method="bfp", block_height=16, block_width=16,
                               mantissa_bits=4, top_k=16, activations=None,
                               skip_lm_head=True, verbose=True):
    """
    Apply quantization to model weights in-place at runtime.
    
    This function quantizes the model WITHOUT saving it, suitable for:
    - Inference/evaluation with quantized weights
    - Reducing memory footprint during runtime
    - Testing quantization configurations quickly
    
    Args:
        model: The model to quantize (will be modified in-place)
        method: Quantization method ("bfp", "awq-fix", "awq-mix")
        block_height: Height of 2D blocks
        block_width: Width of 2D blocks
        mantissa_bits: Number of mantissa bits
        top_k: Number of salient weights for mix-precision
        activations: Pre-collected activations dict (required for AWQ methods)
        skip_lm_head: Whether to skip quantizing lm_head layer
        verbose: Print quantization progress
    
    Returns:
        model: The same model object with quantized weights
    """
    if method in ["awq-fix", "awq-mix"] and activations is None:
        raise ValueError(f"Method '{method}' requires activations. "
                        "Please provide pre-collected activations.")
    
    quantized_count = 0
    total_mse = 0.0
    
    if verbose:
        print(f"\nApplying runtime quantization:")
        print(f"  Method: {method}")
        print(f"  Block size: {block_height}×{block_width}")
        print(f"  Mantissa bits: {mantissa_bits}")
        if method == "awq-mix":
            print(f"  Top-K: {top_k}")
        print("-" * 70)
    
    for name, module in model.named_modules():
        # Skip lm_head if requested
        if skip_lm_head and "lm_head" in name:
            continue
        
        if isinstance(module, torch.nn.Linear):
            original_weight = module.weight.data.clone()
            
            # Apply appropriate quantization method
            if method == "bfp":
                quantized_weight = block_floating_point_quantize_2d(
                    module.weight.data,
                    block_height=block_height,
                    block_width=block_width,
                    mantissa_bits=mantissa_bits
                )
            elif method == "awq-fix":
                if name not in activations:
                    if verbose:
                        print(f"  [SKIP] {name} (no activation)")
                    continue
                quantized_weight = awq_fix_precision_quantize_2d(
                    module.weight.data,
                    activations[name],
                    block_height=block_height,
                    block_width=block_width,
                    mantissa_bits=mantissa_bits
                )
            elif method == "awq-mix":
                if name not in activations:
                    if verbose:
                        print(f"  [SKIP] {name} (no activation)")
                    continue
                quantized_weight = awq_mix_precision_quantize_2d(
                    module.weight.data,
                    activations[name],
                    block_height=block_height,
                    block_width=block_width,
                    mantissa_bits=mantissa_bits,
                    top_k=top_k
                )
            else:
                raise ValueError(f"Unknown quantization method: {method}")
            
            # Update weight in-place
            module.weight.data = quantized_weight
            quantized_count += 1
            
            # Calculate MSE
            mse = torch.mean((original_weight - quantized_weight) ** 2).item()
            total_mse += mse
            
            if verbose:
                print(f"  [{quantized_count:3d}] {name:50s} MSE: {mse:.6e}")
    
    if verbose:
        avg_mse = total_mse / quantized_count if quantized_count > 0 else 0
        print("-" * 70)
        print(f"Quantized {quantized_count} layers")
        print(f"Average MSE: {avg_mse:.6e}")
    
    return model


def save_quantization_config(config, path):
    """Save quantization configuration to file."""
    import json
    config_dict = {
        'method': config.method,
        'block_height': config.block_height,
        'block_width': config.block_width,
        'mantissa_bits': config.mantissa_bits,
        'top_k': config.top_k
    }
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_quantization_config(path):
    """Load quantization configuration from file."""
    import json
    with open(path, 'r') as f:
        config_dict = json.load(f)
    return QuantizationConfig(**config_dict)


if __name__ == "__main__":
    # Example usage
    print("Runtime Quantization Wrapper")
    print("=" * 70)
    print("\nThis module provides on-the-fly quantization for inference.")
    print("\nExample usage:")
    print("""
    from transformers import AutoModelForCausalLM
    from runtime_quantizer import apply_runtime_quantization
    
    # Load original model (BF16 recommended)
    model = AutoModelForCausalLM.from_pretrained(
        "model_path",
        torch_dtype=torch.bfloat16
    )
    
    # Apply quantization at runtime (no disk save)
    apply_runtime_quantization(
        model,
        method="bfp",
        block_height=16,
        block_width=16,
        mantissa_bits=4
    )
    
    # Use model for inference
    outputs = model.generate(...)
    """)
    print("\nBenefits:")
    print("  ✓ No need to save quantized models (saves disk space)")
    print("  ✓ Fast quantization application (seconds)")
    print("  ✓ Easy to test different quantization configs")
    print("  ✓ Original model stays unmodified on disk")
