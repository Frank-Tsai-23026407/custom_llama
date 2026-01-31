"""
Playground Runtime Quantization Module

This module consolidates runtime quantization logic for the Playground.
It implements AWQ variants (Fix/Mix Precision) for runtime use and provides
a unified interface for applying quantization to models on-the-fly.
"""

import torch
import sys
import os

# Import BFP core logic
from custom_model_only_quantize.block_floating_point.block_quantization_2d import block_floating_point_quantize_2d

# Import AWQ logic from their respective modules
from custom_model_only_quantize.fix_precision_awq.awq_utils import awq_fix_precision_quantize_2d
from custom_model_only_quantize.mix_precision_awq.awq_utils import awq_mix_precision_quantize_2d


class QuantizationConfig:
    """Configuration for runtime quantization."""
    
    def __init__(self, method="bfp", block_height=16, block_width=16, 
                 mantissa_bits=4, top_k=16, activations=None, calibration_dataset=None):
        """
        Args:
            method: Quantization method ("bfp", "awq-fix", "awq-mix")
            block_height: Height of 2D blocks
            block_width: Width of 2D blocks
            mantissa_bits: Number of mantissa bits for BFP
            top_k: Number of salient weights for mix-precision (only for awq-mix)
            activations: Pre-collected activations (only for AWQ methods)
            calibration_dataset: Name of the dataset to use for on-the-fly calibration (e.g. "wikitext2")
        """
        self.method = method
        self.block_height = block_height
        self.block_width = block_width
        self.mantissa_bits = mantissa_bits
        self.top_k = top_k
        self.activations = activations # Dict of pre-computed activations
        self.calibration_dataset = calibration_dataset # String name
        self.calibration_data = None # List of input tensors (packed samples)
    
    def __repr__(self):
        return (f"QuantizationConfig(method={self.method}, "
                f"block={self.block_height}x{self.block_width}, "
                f"mantissa={self.mantissa_bits}, top_k={self.top_k})")


def apply_runtime_quantization(model, method="bfp", block_height=16, block_width=16,
                               mantissa_bits=4, top_k=16, activations=None, calibration_data=None,
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
        activations: Pre-collected activations dict (Legacy/Fast for small models)
        calibration_data: List of packed input tensors for on-the-fly layer-wise calibration (Memory Safe)
        skip_lm_head: Whether to skip quantizing lm_head layer
        verbose: Print quantization progress
    """
    
    # Import helper here to avoid circular dependencies if any, or just convenience
    # runtime_quantize is in custom_model_only_quantize, activation_utils is in playground.
    # We might need to adjust python path or assume it works.
    # Given the previous context `sys.path.insert(0, ...)` in utils.py, it should work if we import from activation_utils
    # BUT `playground` is not a package in some contexts.
    # Safeguard import
    get_layer_activations = None
    if calibration_data is not None:
        try:
            from activation_utils import get_layer_activations
        except ImportError:
            # Try relative if running from different root
            try:
                from playground.activation_utils import get_layer_activations
            except ImportError:
                print("Warning: Could not import get_layer_activations. Layer-wise calibration unavailable.")

    if method in ["awq-fix", "awq-mix"]:
        if activations is None and calibration_data is None:
            raise ValueError(f"Method '{method}' requires activations or calibration_data.")
    
    quantized_count = 0
    total_mse = 0.0
    
    if verbose:
        print(f"\nApplying runtime quantization:")
        print(f"  Method: {method}")
        print(f"  Block size: {block_height}×{block_width}")
        print(f"  Mantissa bits: {mantissa_bits}")
        if method == "awq-mix":
            print(f"  Top-K: {top_k}")
        if calibration_data is not None:
            print(f"  Mode: Layer-wise Calibration (Memory Safe)")
        elif activations is not None:
            print(f"  Mode: Pre-computed Activations")
        print("-" * 70)
    
    # Pre-calculate device/dtype for efficiency if needed
    
    # Using specific iteration to handle layer-wise logic
    for name, module in model.named_modules():
        # Skip lm_head if requested
        if skip_lm_head and "lm_head" in name:
            continue
        
        if isinstance(module, torch.nn.Linear):
            original_weight = module.weight.data.clone()
            
            # Prepare Activations for AWQ
            layer_activations = None
            if method in ["awq-fix", "awq-mix"]:
                # 1. Try pre-computed dict
                if activations is not None:
                    if name in activations:
                        layer_activations = activations[name]
                
                # 2. Try layer-wise calibration
                if layer_activations is None and calibration_data is not None and get_layer_activations is not None:
                    if verbose: print(f"  .. calibrating {name} ...", end="\r")
                    layer_activations = get_layer_activations(model, calibration_data, name, device=module.weight.device)
                
                # 3. If still None, skip
                if layer_activations is None:
                    if verbose:
                        print(f"  [SKIP] {name} (no activation)")
                    continue

            # Apply appropriate quantization method
            if method == "bfp":
                quantized_weight = block_floating_point_quantize_2d(
                    module.weight.data,
                    block_height=block_height,
                    block_width=block_width,
                    mantissa_bits=mantissa_bits
                )
            elif method == "awq-fix":
                quantized_weight = awq_fix_precision_quantize_2d(
                    module.weight.data,
                    layer_activations,
                    block_height=block_height,
                    block_width=block_width,
                    mantissa_bits=mantissa_bits
                )
            elif method == "awq-mix":
                quantized_weight = awq_mix_precision_quantize_2d(
                    module.weight.data,
                    layer_activations,
                    block_height=block_height,
                    block_width=block_width,
                    mantissa_bits=mantissa_bits,
                    top_k=top_k
                )
            else:
                raise ValueError(f"Unknown quantization method: {method}")
            
            # Clean up activation immediately to save memory
            del layer_activations
            
            # Update weight in-place
            module.weight.data = quantized_weight
            quantized_count += 1
            
            # Calculate MSE
            mse = torch.mean((original_weight - quantized_weight) ** 2).item()
            total_mse += mse
            
            if verbose:
                print(f"  [{quantized_count:3d}] {name:50s} MSE: {mse:.6e}")
                
            # GC occasionally?
            if quantized_count % 10 == 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
    
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
