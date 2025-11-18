import argparse
import os
import sys
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantize_model_script.block_quantization_2d import awq_quantize_2d


def resolve_model_path(model: str) -> str:
    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    presets = {
        "llama-3.2-1b": os.path.join(ws_root, "model/llama-3.2-1b/Llama-3.2-1B"),
        "tinyllama": os.path.join(ws_root, "model/tinyllama/TinyLlama_1.1v"),
    }
    if model in presets:
        return presets[model]
    return model


def get_calibration_data(model, tokenizer, n_samples=128, seq_len=512):
    """
    Get calibration data for AWQ quantization.
    
    Args:
        model: The model to calibrate
        tokenizer: Tokenizer for the model
        n_samples: Number of calibration samples
        seq_len: Sequence length for calibration
        
    Returns:
        Dictionary mapping layer names to activation tensors
    """
    print(f"Loading calibration dataset (n_samples={n_samples}, seq_len={seq_len})...")
    
    # Load WikiText dataset for calibration
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Prepare calibration samples
    calibration_texts = []
    for i in range(min(n_samples, len(dataset))):
        text = dataset[i]["text"]
        if text.strip():  # Skip empty texts
            calibration_texts.append(text)
            if len(calibration_texts) >= n_samples:
                break
    
    # Tokenize
    print("Tokenizing calibration data...")
    encoded = tokenizer(
        calibration_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_len
    )
    
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # Collect activations
    print("Collecting activations...")
    activations = {}
    hooks = []
    
    def get_activation_hook(name):
        def hook(module, input, output):
            # Store the input activation for this layer
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input
            
            if name not in activations:
                activations[name] = []
            
            # Detach and move to CPU to save memory
            activations[name].append(inp.detach().cpu())
        return hook
    
    # Register hooks for all linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            hook = module.register_forward_hook(get_activation_hook(name))
            hooks.append(hook)
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate activations
    print("Processing activations...")
    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0)
    
    return activations


def quantize_model_awq_2d(model, activations, block_height: int, block_width: int, 
                          mantissa_bits: int, top_k: int = 16):
    """
    Quantize model weights using 2D AWQ quantization.
    
    Args:
        model: The model to quantize
        activations: Dictionary mapping layer names to activation tensors
        block_height: Height of each 2D block
        block_width: Width of each 2D block
        mantissa_bits: Number of mantissa bits for quantization
        top_k: Number of salient weights to preserve per block
    """
    for name, module in model.named_modules():
        if "lm_head" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            if name in activations:
                act = activations[name]
                # Ensure activation is on the same device as the weight
                act = act.to(module.weight.device)
                
                q_weight = awq_quantize_2d(
                    module.weight.data,
                    act,
                    block_height=block_height,
                    block_width=block_width,
                    mantissa_bits=mantissa_bits,
                    top_k=top_k
                )
                module.weight.data = q_weight
            else:
                print(f"Warning: No activation found for {name}, skipping AWQ")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="2D AWQ Weight Quantization CLI")
    parser.add_argument("--model", type=str, default="tinyllama",
                        help="Model preset or path. Presets: llama-3.2-1b, tinyllama")
    parser.add_argument("--block-height", type=int, default=32,
                        help="Block height (default: 32)")
    parser.add_argument("--block-width", type=int, default=16,
                        help="Block width (default: 16)")
    parser.add_argument("--mantissa-bits", type=int, nargs="+", default=[3, 4],
                        help="Mantissa bits to try, e.g., --mantissa-bits 3 4")
    parser.add_argument("--top-k", type=int, default=16,
                        help="Number of salient weights per block (default: 16)")
    parser.add_argument("--n-samples", type=int, default=128,
                        help="Number of calibration samples (default: 128)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for calibration (default: 512)")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving models")

    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    print(f"Resolved model path: {model_path}")
    print(f"Block size: {args.block_height}x{args.block_width}")
    print(f"Top-K salient weights per block: {args.top_k}")

    print("Loading model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get calibration data
    activations = get_calibration_data(base_model, tokenizer, 
                                       n_samples=args.n_samples, 
                                       seq_len=args.seq_len)

    for m in args.mantissa_bits:
        print("----------------------------------------------------------------")
        print(f"Running 2D AWQ with Mantissa Bits = {m}, Block Size = {args.block_height}x{args.block_width}, Top-K = {args.top_k}")
        print("----------------------------------------------------------------")
        model_copy = copy.deepcopy(base_model)
        quantize_model_awq_2d(
            model_copy, 
            activations,
            block_height=args.block_height,
            block_width=args.block_width,
            mantissa_bits=m,
            top_k=args.top_k
        )

        if not args.dry_run:
            output_dir = f"{model_path}-awq2d-quantized-b{args.block_height}x{args.block_width}-m{m}-k{args.top_k}"
            os.makedirs(output_dir, exist_ok=True)
            model_copy.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Quantized model saved to: {output_dir}")
        else:
            print("Dry run complete; not saving model.")

    print("\nAll 2D AWQ quantization tasks are complete.")


if __name__ == "__main__":
    main()
