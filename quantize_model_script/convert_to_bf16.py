#!/usr/bin/env python3
"""
Convert model to BF16 format to save disk space.

This script converts FP32 models to BF16, reducing file size by ~50%.
Use this as the base model for runtime quantization.

Usage:
    python convert_to_bf16.py --model tinyllama
    python convert_to_bf16.py --model /path/to/model --output /path/to/output
"""

import argparse
import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_model_path(model: str) -> str:
    """Resolve model preset to actual path."""
    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    presets = {
        "llama-3.2-1b": os.path.join(ws_root, "model/llama-3.2-1b/Llama-3.2-1B"),
        "tinyllama": os.path.join(ws_root, "model/tinyllama/TinyLlama_1.1v"),
    }
    if model in presets:
        return presets[model]
    return model


def get_model_size(model_path):
    """Get total size of model files in GB."""
    total_size = 0
    for root, dirs, files in os.walk(model_path):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024**3)  # Convert to GB


def main():
    parser = argparse.ArgumentParser(
        description="Convert model to BF16 format for space savings"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model preset or path to convert")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <model_path>_bf16)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"],
                        help="Target dtype (default: bfloat16)")
    
    args = parser.parse_args()
    
    # Resolve paths
    model_path = resolve_model_path(args.model)
    
    if args.output:
        output_path = args.output
    else:
        output_path = f"{model_path}_{args.dtype}"
    
    print("=" * 80)
    print(f"Converting Model to {args.dtype.upper()}")
    print("=" * 80)
    print(f"\nInput:  {model_path}")
    print(f"Output: {output_path}")
    
    # Check input size
    if os.path.exists(model_path):
        input_size = get_model_size(model_path)
        print(f"\nInput size: {input_size:.2f} GB")
    
    # Load model
    print(f"\nLoading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Convert dtype
    print(f"\nConverting to {args.dtype}...")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    model = model.to(dtype_map[args.dtype])
    print("Conversion complete!")
    
    # Save
    print(f"\nSaving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, torch_dtype=torch.bfloat16)
    tokenizer.save_pretrained(output_path)
    print("Saved successfully!")
    
    # Check output size
    if os.path.exists(output_path):
        output_size = get_model_size(output_path)
        print(f"\nOutput size: {output_size:.2f} GB")
        
        if os.path.exists(model_path):
            savings = ((input_size - output_size) / input_size) * 100
            print(f"Space saved: {input_size - output_size:.2f} GB ({savings:.1f}%)")
    
    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"\nYou can now use this model for runtime quantization:")
    print(f"\npython evaluate_with_runtime_quantization.py \\")
    print(f"    --model {output_path} \\")
    print(f"    --method bfp \\")
    print(f"    --block-height 16 \\")
    print(f"    --block-width 16 \\")
    print(f"    --mantissa-bits 4")


if __name__ == "__main__":
    main()
