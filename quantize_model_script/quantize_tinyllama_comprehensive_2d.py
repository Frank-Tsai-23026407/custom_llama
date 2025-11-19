"""
Comprehensive TinyLlama 2D Quantization Script

Supports multiple quantization methods:
1. BFP (Block Floating Point) - Standard 2D block quantization
2. AWQ (Activation-Aware Weight Quantization) - 2D with activation awareness
3. Mix-Precision AWQ - Preserves most important weights in FP32
4. Fix-Precision AWQ - Standard AWQ with fixed precision

All with 2D block configurations.
"""

import argparse
import os
import sys
import copy
import torch
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantize_model_script.block_quantization_2d import (
    block_floating_point_quantize_2d,
    awq_fix_precision_quantize_2d,
    awq_mix_precision_quantize_2d
)


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


def collect_activations(model, tokenizer, dataset_name="Salesforce/wikitext", 
                       dataset_config="wikitext-103-raw-v1", num_samples=128):
    """
    Collect activations from model using calibration dataset.
    
    Returns:
        dict: Mapping from layer name to activation tensor
    """
    print(f"\nCollecting activations from {num_samples} samples of {dataset_name}...")
    
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split="train").select(range(num_samples))
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids
    
    # Dictionary to store activations
    activations = {}
    
    # Hook function to capture activations
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = input[0].detach()
        return hook
    
    # Register hooks for all linear layers
    hooks = []
    for name, module in model.named_modules():
        if "lm_head" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(get_activation(name)))
    
    # Perform forward pass to get activations
    print("Running forward pass...")
    with torch.no_grad():
        model(tokens)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    print(f"Collected activations for {len(activations)} layers")
    return activations


def quantize_model_bfp_2d(model, block_height: int, block_width: int, mantissa_bits: int):
    """Standard BFP 2D quantization (no activation awareness)."""
    quantized_count = 0
    
    for name, module in model.named_modules():
        if "lm_head" in name:
            continue
            
        if isinstance(module, torch.nn.Linear):
            original_weight = module.weight.data.clone()
            
            q_weight = block_floating_point_quantize_2d(
                module.weight.data, 
                block_height=block_height, 
                block_width=block_width,
                mantissa_bits=mantissa_bits
            )
            
            module.weight.data = q_weight
            quantized_count += 1
            
            mse = torch.mean((original_weight - q_weight) ** 2).item()
            print(f"  [{quantized_count:3d}] {name:50s} MSE: {mse:.6e}")
    
    print(f"\nQuantized {quantized_count} layers")
    return model


def quantize_model_awq_2d(model, activations, block_height: int, block_width: int, 
                         mantissa_bits: int, top_k: int = 16, method: str = "fix"):
    """
    AWQ 2D quantization with activation awareness.
    
    Args:
        method: "fix" for fix-precision (scaling-based, all weights in BFP), 
                "mix" for mix-precision (top-k in FP32, rest in BFP)
    """
    quantized_count = 0
    
    for name, module in model.named_modules():
        if "lm_head" in name:
            continue
            
        if isinstance(module, torch.nn.Linear):
            if name not in activations:
                print(f"  Skipping {name} (no activation)")
                continue
            
            original_weight = module.weight.data.clone()
            activation = activations[name]
            
            if method == "fix":
                # Fix-precision: scaling-based, all weights in BFP
                q_weight = awq_fix_precision_quantize_2d(
                    module.weight.data,
                    activation,
                    block_height=block_height,
                    block_width=block_width,
                    mantissa_bits=mantissa_bits
                )
            else:  # "mix"
                # Mix-precision: top-k weights in FP32, rest in BFP
                q_weight = awq_mix_precision_quantize_2d(
                    module.weight.data,
                    activation,
                    block_height=block_height,
                    block_width=block_width,
                    mantissa_bits=mantissa_bits,
                    top_k=top_k
                )
            
            module.weight.data = q_weight
            quantized_count += 1
            
            mse = torch.mean((original_weight - q_weight) ** 2).item()
            print(f"  [{quantized_count:3d}] {name:50s} MSE: {mse:.6e}")
    
    print(f"\nQuantized {quantized_count} layers")
    return model


def main():
    # Quantization configurations
    BLOCK_SIZES = [
        (128, 1), (64, 2), (32, 4), (16, 8),
        (8, 16), (4, 32), (2, 64), (1, 128),
    ]
    MANTISSA_BITS = [5, 4]
    
    parser = argparse.ArgumentParser(
        description="Comprehensive TinyLlama 2D Quantization (BFP, AWQ, Mix/Fix-Precision)"
    )
    parser.add_argument("--model", type=str, default="tinyllama",
                        help="Model preset or path (default: tinyllama)")
    parser.add_argument("--method", type=str, 
                        choices=["bfp", "awq-fix", "awq-mix", "all"],
                        default="all",
                        help="Quantization method: bfp, awq-fix, awq-mix, or all (default: all)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory")
    parser.add_argument("--top-k", type=int, default=16,
                        help="Top-k salient weights to preserve for AWQ (default: 16)")
    parser.add_argument("--dataset", type=str, default="Salesforce/wikitext",
                        help="Calibration dataset for AWQ (default: Salesforce/wikitext)")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1",
                        help="Dataset configuration (default: wikitext-103-raw-v1)")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without saving models")
    parser.add_argument("--skip-generation-test", action="store_true",
                        help="Skip generation test after quantization")
    parser.add_argument("--save-dtype", type=str, 
                        choices=["float32", "float16", "bfloat16"],
                        default="bfloat16",
                        help="Data type for saving model (default: bfloat16). Note: This only changes storage format, not the quantization itself.")
    
    args = parser.parse_args()
    
    # Resolve model path
    model_path = resolve_model_path(args.model)
    print(f"Model path: {model_path}")
    
    # Set output directory
    if args.output_dir:
        output_base = args.output_dir
    else:
        output_base = f"{model_path}-2d-comprehensive"
    
    print(f"Output base directory: {output_base}")
    
    # Determine which methods to run
    methods = []
    if args.method == "all":
        methods = ["bfp", "awq-fix", "awq-mix"]
    else:
        methods = [args.method]
    
    total_configs = len(BLOCK_SIZES) * len(MANTISSA_BITS) * len(methods)
    print(f"Total configurations: {len(BLOCK_SIZES)} × {len(MANTISSA_BITS)} × {len(methods)} = {total_configs}")
    print("=" * 80)
    
    # Load base model and tokenizer
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\nLoading base model from: {model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Model loaded successfully!")
    except ImportError:
        print("ERROR: transformers library not found. Please install it:")
        print("  pip install transformers datasets")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
    
    # Collect activations if needed (for AWQ methods)
    activations = None
    if "awq-fix" in methods or "awq-mix" in methods:
        activations = collect_activations(
            base_model, tokenizer, 
            args.dataset, args.dataset_config, args.num_samples
        )
    
    # Process each configuration
    config_num = 0
    
    for method in methods:
        for block_height, block_width in BLOCK_SIZES:
            for mantissa_bits in MANTISSA_BITS:
                config_num += 1
                print("\n" + "=" * 80)
                print(f"Configuration {config_num}/{total_configs}")
                print(f"Method: {method.upper()}, Block: {block_height}×{block_width}, Mantissa: {mantissa_bits}")
                print("=" * 80)
                
                # Deep copy the model
                print("Creating model copy...")
                model_copy = copy.deepcopy(base_model)
                
                # Apply quantization based on method
                print(f"\nQuantizing with {method}:")
                if method == "bfp":
                    quantize_model_bfp_2d(model_copy, block_height, block_width, mantissa_bits)
                elif method == "awq-fix":
                    quantize_model_awq_2d(
                        model_copy, activations, block_height, block_width, 
                        mantissa_bits, top_k=args.top_k, method="fix"
                    )
                elif method == "awq-mix":
                    quantize_model_awq_2d(
                        model_copy, activations, block_height, block_width, 
                        mantissa_bits, top_k=args.top_k, method="mix"
                    )
                
                # Test generation (optional)
                if not args.skip_generation_test:
                    print("\nTesting generation...")
                    try:
                        inputs = tokenizer("Hello, how are you?", return_tensors="pt")
                        with torch.no_grad():
                            outputs = model_copy.generate(
                                **inputs,
                                max_new_tokens=30,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id
                            )
                        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        print(f"Generated: {generated[:100]}...")
                    except Exception as e:
                        print(f"Generation test failed: {e}")
                
                # Save model
                if not args.dry_run:
                    method_dir = method.replace("-", "_")
                    output_dir = os.path.join(
                        output_base,
                        method_dir,
                        f"block_{block_height}x{block_width}_mantissa_{mantissa_bits}"
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"\nSaving to: {output_dir}")
                    
                    # Convert dtype before saving if requested
                    if args.save_dtype == "float16":
                        print("Converting to float16 for storage...")
                        model_copy = model_copy.half()
                    elif args.save_dtype == "bfloat16":
                        print("Converting to bfloat16 for storage...")
                        model_copy = model_copy.to(torch.bfloat16)
                    
                    # Save with appropriate dtype parameter
                    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
                    model_copy.save_pretrained(output_dir, torch_dtype=dtype_map.get(args.save_dtype, torch.float32))
                    tokenizer.save_pretrained(output_dir)
                    print(f"Saved successfully (dtype: {args.save_dtype})!")
                else:
                    print("\n[DRY RUN] Skipping save")
                
                # Clean up
                del model_copy
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                print(f"\nConfiguration {config_num} completed")
    
    print("\n" + "=" * 80)
    print("All quantization configurations completed!")
    print("=" * 80)
    
    if not args.dry_run:
        print(f"\nQuantized models saved in: {output_base}")
        print("\nDirectory structure:")
        for method in methods:
            method_dir = method.replace("-", "_")
            print(f"\n{method_dir}/")
            for block_height, block_width in BLOCK_SIZES:
                for mantissa_bits in MANTISSA_BITS:
                    dir_name = f"  - block_{block_height}x{block_width}_mantissa_{mantissa_bits}"
                    print(dir_name)


if __name__ == "__main__":
    main()
