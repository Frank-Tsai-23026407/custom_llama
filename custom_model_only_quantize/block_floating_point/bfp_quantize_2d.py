import argparse
import os
import sys
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from custom_model_only_quantize.block_floating_point.block_quantization_2d import block_floating_point_quantize_2d
from testbench import task_utils as TU


def resolve_model_path(model: str) -> str:
    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    presets = {
        "llama-3.2-1b": os.path.join(ws_root, "model/llama-3.2-1b/Llama-3.2-1B"),
        "tinyllama": os.path.join(ws_root, "model/tinyllama/TinyLlama_1.1v"),
    }
    if model in presets:
        print(f"Resolved model path on local: {presets[model]}")
        return presets[model]
    print(f"Load model from remote: {model}")
    return model


def quantize_model_bfp_2d(model, block_height: int, block_width: int, mantissa_bits: int):
    """
    Quantize model weights using 2D block floating point quantization.
    
    Args:
        model: The model to quantize
        block_height: Height of each 2D block
        block_width: Width of each 2D block
        mantissa_bits: Number of mantissa bits for quantization
    """
    for name, module in model.named_modules():
        if "lm_head" in name or "embed_tokens" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            # module.weight.data has shape (out_features, in_features)
            q_weight = block_floating_point_quantize_2d(
                module.weight.data, 
                block_height=block_height, 
                block_width=block_width,
                mantissa_bits=mantissa_bits
            )
            module.weight.data = q_weight
    return model


def main():
    parser = argparse.ArgumentParser(description="2D Block BFP Weight Quantization CLI")
    parser.add_argument("--model", type=str, default="tinyllama",
                        help="Model preset or path. Presets: llama-3.2-1b, tinyllama")
    parser.add_argument("--block-height", type=int, default=32, 
                        help="BFP block height (default: 32)")
    parser.add_argument("--block-width", type=int, default=16, 
                        help="BFP block width (default: 16)")
    parser.add_argument("--mantissa-bits", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="Mantissa bits to try, e.g., --mantissa-bits 2 3 4 5")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving models")
    parser.add_argument("--eval-ppl", action="store_true", help="Evaluate WikiText2 PPL")
    parser.add_argument("--eval-hellaswag", action="store_true", help="Evaluate HellaSwag Accuracy")
    parser.add_argument("--max-samples-hs", type=int, default=None, help="Max samples for HellaSwag evaluation")
    parser.add_argument("--limit-tokens-ppl", type=int, default=None, help="Limit tokens for PPL evaluation for speed")

    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    print("Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Block size: {args.block_height}x{args.block_width}")

    for m in args.mantissa_bits:
        print("----------------------------------------------------------------")
        print(f"Running 2D BFP with Mantissa Bits = {m}, Block Size = {args.block_height}x{args.block_width}")
        print("----------------------------------------------------------------")
        model_copy = copy.deepcopy(base_model)
        quantize_model_bfp_2d(model_copy, block_height=args.block_height, 
                              block_width=args.block_width, mantissa_bits=m)

        if args.eval_ppl:
            TU.evaluate_ppl(model_copy, tokenizer, device, limit_tokens=args.limit_tokens_ppl)
        
        if args.eval_hellaswag:
            TU.evaluate_hellaswag(model_copy, tokenizer, device, max_samples=args.max_samples_hs)

        if not args.dry_run:
            output_dir = f"{model_path}-bfp2d-quantized-b{args.block_height}x{args.block_width}-m{m}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to bf16 to save disk space
            model_copy = model_copy.to(torch.bfloat16)
            model_copy.save_pretrained(output_dir, torch_dtype=torch.bfloat16)
            tokenizer.save_pretrained(output_dir)
            print(f"Quantized model saved to: {output_dir} (bf16 format)")
        else:
            print("Dry run complete; not saving model.")

    print("\nAll 2D BFP quantization tasks are complete.")


if __name__ == "__main__":
    main()
