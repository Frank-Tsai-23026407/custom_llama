"""
Example: Evaluate TinyLlama with Runtime Quantization

This script demonstrates how to:
1. Load original model (in BF16 to save memory)
2. Apply quantization at runtime (in-memory, no disk save)
3. Run evaluation/generation
4. Repeat with different quantization configs

Disk space savings: Only need to store 1 original model instead of 48 quantized variants!
"""

import argparse
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from runtime_quantizer import apply_runtime_quantization
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


def test_generation(model, tokenizer, prompt="Hello, how are you?", max_tokens=50):
    """Test model generation."""
    print(f"\nTesting generation with prompt: '{prompt}'")
    print("-" * 70)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated}")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TinyLlama with Runtime Quantization (No Disk Save)"
    )
    parser.add_argument("--model", type=str, default="tinyllama",
                        help="Model preset or path")
    parser.add_argument("--method", type=str, default="bfp",
                        choices=["bfp", "awq-fix", "awq-mix"],
                        help="Quantization method")
    parser.add_argument("--block-height", type=int, default=16,
                        help="Block height")
    parser.add_argument("--block-width", type=int, default=16,
                        help="Block width")
    parser.add_argument("--mantissa-bits", type=int, default=4,
                        help="Mantissa bits")
    parser.add_argument("--top-k", type=int, default=16,
                        help="Top-k for mix-precision")
    parser.add_argument("--load-dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data type to load model in (default: bfloat16)")
    
    args = parser.parse_args()
    
    model_path = resolve_model_path(args.model)
    
    print("=" * 80)
    print("TinyLlama Runtime Quantization Evaluation")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Load dtype: {args.load_dtype}")
    print(f"Quantization method: {args.method}")
    print(f"Block size: {args.block_height}×{args.block_width}")
    print(f"Mantissa bits: {args.mantissa_bits}")
    
    # Load model in BF16 (or FP16) to save memory
    print(f"\nLoading model in {args.load_dtype} format...")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[args.load_dtype],
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Model loaded!")
    
    # Test baseline (before quantization)
    print("\n" + "=" * 80)
    print("BASELINE (Before Quantization)")
    print("=" * 80)
    test_generation(model, tokenizer, max_tokens=30)
    
    # Collect activations if needed for AWQ
    activations = None
    if args.method in ["awq-fix", "awq-mix"]:
        print("\n" + "=" * 80)
        print("Collecting Activations for AWQ...")
        print("=" * 80)
        
        from datasets import load_dataset
        
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        dataset = dataset.select(range(128))
        text = "\n\n".join(dataset["text"])
        tokens = tokenizer(text, return_tensors="pt").input_ids
        
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = input[0].detach()
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if "lm_head" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        with torch.no_grad():
            model(tokens)
        
        for hook in hooks:
            hook.remove()
        
        print(f"Collected activations for {len(activations)} layers")
    
    # Apply quantization at runtime
    print("\n" + "=" * 80)
    print("APPLYING QUANTIZATION (Runtime)")
    print("=" * 80)
    
    apply_runtime_quantization(
        model,
        method=args.method,
        block_height=args.block_height,
        block_width=args.block_width,
        mantissa_bits=args.mantissa_bits,
        top_k=args.top_k,
        activations=activations,
        verbose=True
    )
    
    # Test after quantization
    print("\n" + "=" * 80)
    print("AFTER QUANTIZATION")
    print("=" * 80)
    test_generation(model, tokenizer, max_tokens=30)
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print("\nNote: Model was quantized in-memory only.")
    print("No quantized model was saved to disk (saves disk space).")
    print("\nTo use this model again:")
    print("  1. Load original model")
    print("  2. Apply runtime quantization")
    print("  3. Run inference")


if __name__ == "__main__":
    main()
