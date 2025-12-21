import argparse
import os
import sys
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from custom_model_only_quantize.block_floating_point.block_quantization import block_floating_point_quantize
from testbench import task_utils as TU


def resolve_model_path(model: str) -> str:
    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    presets = {
        "llama-3.2-1b": os.path.join(ws_root, "model/llama-3.2-1b/Llama-3.2-1B"),
        "tinyllama": os.path.join(ws_root, "model/tinyllama/TinyLlama_1.1v"),
    }
    if model in presets:
        return presets[model]
    return model


def quantize_model_bfp(model, block_size: int, mantissa_bits: int):
    for name, module in model.named_modules():
        if "lm_head" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            q_weight = block_floating_point_quantize(module.weight.data, block_size=block_size, mantissa_bits=mantissa_bits)
            module.weight.data = q_weight
    return model




def main():
    parser = argparse.ArgumentParser(description="Pure BFP Weight Quantization CLI")
    parser.add_argument("--model", type=str, default="llama-3.2-1b",
                        help="Model preset or path. Presets: llama-3.2-1b, tinyllama")
    parser.add_argument("--block-size", type=int, default=64, help="BFP block size")
    parser.add_argument("--mantissa-bits", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="Mantissa bits to try, e.g., --mantissa-bits 2 3 4 5")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving models")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for quantized models")
    parser.add_argument("--eval-ppl", action="store_true", help="Evaluate WikiText2 PPL")
    parser.add_argument("--eval-hellaswag", action="store_true", help="Evaluate HellaSwag Accuracy")
    parser.add_argument("--max-samples-hs", type=int, default=None, help="Max samples for HellaSwag evaluation")
    parser.add_argument("--limit-tokens-ppl", type=int, default=None, help="Limit tokens for PPL evaluation for speed")

    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    print(f"Resolved model path: {model_path}")

    print("Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for m in args.mantissa_bits:
        print("----------------------------------------------------------------")
        print(f"Running BFP with Mantissa Bits = {m}, Block Size = {args.block_size}")
        print("----------------------------------------------------------------")
        model_copy = copy.deepcopy(base_model)
        quantize_model_bfp(model_copy, block_size=args.block_size, mantissa_bits=m)

        if args.eval_ppl:
            TU.evaluate_ppl(model_copy, tokenizer, device, limit_tokens=args.limit_tokens_ppl)
        
        if args.eval_hellaswag:
            TU.evaluate_hellaswag(model_copy, tokenizer, device, max_samples=args.max_samples_hs)

        if not args.dry_run:
            output_dir = args.output_dir if args.output_dir else f"{model_path}-bfp-quantized-b{args.block_size}-m{m}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to bf16 to save disk space
            model_to_save = model_copy.half() # Use half for saving
            model_to_save.save_pretrained(output_dir, torch_dtype=torch.float16)
            tokenizer.save_pretrained(output_dir)
            print(f"Quantized model saved to: {output_dir}")
        else:
            print("Dry run complete; not saving model.")

    print("\nAll BFP quantization tasks are complete.")


if __name__ == "__main__":
    main()
