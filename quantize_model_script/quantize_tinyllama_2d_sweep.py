"""
Quantize TinyLlama model with 2D block-based BFP quantization.
Sweeps through multiple block size configurations and mantissa bit settings.
"""

import argparse
import os
import sys
import copy
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantize_model_script.block_quantization_2d import block_floating_point_quantize_2d


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


def quantize_model_bfp_2d(model, block_height: int, block_width: int, mantissa_bits: int):
    """
    Quantize model weights using 2D block floating point quantization.
    
    Args:
        model: The model to quantize (in-place)
        block_height: Height of each 2D block
        block_width: Width of each 2D block
        mantissa_bits: Number of mantissa bits for quantization
    
    Returns:
        Quantized model (same object, modified in-place)
    """
    quantized_count = 0
    
    for name, module in model.named_modules():
        # Skip lm_head to preserve output layer
        if "lm_head" in name:
            continue
            
        if isinstance(module, torch.nn.Linear):
            original_weight = module.weight.data.clone()
            
            # Apply 2D BFP quantization
            q_weight = block_floating_point_quantize_2d(
                module.weight.data, 
                block_height=block_height, 
                block_width=block_width,
                mantissa_bits=mantissa_bits
            )
            
            module.weight.data = q_weight
            quantized_count += 1
            
            # Calculate MSE for this layer
            mse = torch.mean((original_weight - q_weight) ** 2).item()
            print(f"  [{quantized_count:3d}] {name:50s} {tuple(original_weight.shape)} MSE: {mse:.6e}")
    
    print(f"\nQuantized {quantized_count} layers")
    return model


def main():
    # Quantization configurations
    BLOCK_SIZES = [
        (128, 1),
        (64, 2),
        (32, 4),
        (16, 8),
        (8, 16),
        (4, 32),
        (2, 64),
        (1, 128),
    ]
    MANTISSA_BITS = [5, 4]
    
    parser = argparse.ArgumentParser(description="TinyLlama 2D BFP Quantization Sweep")
    parser.add_argument("--model", type=str, default="tinyllama",
                        help="Model preset or path (default: tinyllama)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory (default: <model_path>-bfp2d-sweep)")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Run without saving models")
    parser.add_argument("--skip-generation-test", action="store_true",
                        help="Skip generation test after quantization")
    parser.add_argument("--save-dtype", type=str,
                        choices=["float32", "float16", "bfloat16"],
                        default="bfloat16",
                        help="Data type for saving (default: bfloat16)")
    
    args = parser.parse_args()
    
    # Resolve model path
    model_path = resolve_model_path(args.model)
    print(f"Model path: {model_path}")
    
    # Set output directory
    if args.output_dir:
        output_base = args.output_dir
    else:
        output_base = f"{model_path}-bfp2d-sweep"
    
    print(f"Output base directory: {output_base}")
    print(f"Total configurations: {len(BLOCK_SIZES)} × {len(MANTISSA_BITS)} = {len(BLOCK_SIZES) * len(MANTISSA_BITS)}")
    print("=" * 80)
    
    # Load base model and tokenizer once
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\nLoading base model from: {model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Model loaded successfully!")
    except ImportError:
        print("ERROR: transformers library not found. Please install it:")
        print("  pip install transformers")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)
    
    # Process each configuration
    config_num = 0
    for block_height, block_width in BLOCK_SIZES:
        for mantissa_bits in MANTISSA_BITS:
            config_num += 1
            print("\n" + "=" * 80)
            print(f"Configuration {config_num}/{len(BLOCK_SIZES) * len(MANTISSA_BITS)}")
            print(f"Block Size: {block_height}×{block_width}, Mantissa Bits: {mantissa_bits}")
            print("=" * 80)
            
            # Deep copy the model for this configuration
            print("Creating model copy...")
            model_copy = copy.deepcopy(base_model)
            
            # Quantize
            print(f"\nQuantizing with block={block_height}×{block_width}, mantissa={mantissa_bits}:")
            quantize_model_bfp_2d(model_copy, block_height, block_width, mantissa_bits)
            
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
                output_dir = os.path.join(
                    output_base,
                    f"block_{block_height}x{block_width}_mantissa_{mantissa_bits}"
                )
                os.makedirs(output_dir, exist_ok=True)
                print(f"\nSaving to: {output_dir}")
                
                # Convert dtype if requested
                if args.save_dtype == "float16":
                    print("Converting to float16 for storage...")
                    model_copy = model_copy.half()
                elif args.save_dtype == "bfloat16":
                    print("Converting to bfloat16 for storage...")
                    model_copy = model_copy.to(torch.bfloat16)
                
                model_copy.save_pretrained(output_dir)
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
        for block_height, block_width in BLOCK_SIZES:
            for mantissa_bits in MANTISSA_BITS:
                dir_name = f"block_{block_height}x{block_width}_mantissa_{mantissa_bits}"
                print(f"  - {dir_name}")

if __name__ == "__main__":
    main()
