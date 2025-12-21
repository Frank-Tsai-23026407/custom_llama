import argparse
import copy
import os
import sys
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import re
from awq_utils import pseudo_quantize, search_awq_scale, apply_awq_scale

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from testbench import task_utils as TU
from custom_model_only_quantize.block_floating_point.block_quantization_2d import block_floating_point_quantize_2d




def resolve_model_path(model: str) -> str:
    """Resolves a model alias to a full path or returns a given path.

    This function provides a convenient way to refer to commonly used models with
    short aliases (e.g., "tinyllama") instead of typing the full path. If the
    provided `model` string does not match a preset, it is assumed to be a valid
    path and is returned as is.

    Args:
        model (str): The model alias or a direct path to the model directory.

    Returns:
        str: The resolved, absolute path to the model directory.
    """
    # Workspace-relative presets
    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    presets = {
        "llama-3.2-1b": os.path.join(ws_root, "model/llama-3.2-1b/Llama-3.2-1B"),
        "tinyllama": os.path.join(ws_root, "model/tinyllama/TinyLlama_1.1v"),
    }
    if model in presets:
        return presets[model]
    # If user passed a path, return as-is
    return model

def quantize_model(model_path, dataset_name, dataset_config, num_samples, block_height, block_width, mantissa_bits, device="auto", dry_run=False):
    """Applies fixed-precision AWQ to a language model.

    This function orchestrates the end-to-end quantization process. It performs the
    following steps:
    1.  Loads the pretrained model and tokenizer from the specified path.
    2.  Loads and preprocesses a calibration dataset.
    3.  Registers forward hooks on all linear layers to capture input activations.
    4.  Runs a forward pass with the calibration data to collect these activations.
    5.  Applies the AWQ algorithm to each linear layer using the captured activations.
    6.  Saves the newly quantized model and its tokenizer to a new directory.

    Args:
        model_path (str): The path to the pretrained model to be quantized.
        dataset_name (str): The name of the Hugging Face dataset for calibration (e.g., "wikitext").
        dataset_config (str): The specific configuration of the dataset to use.
        num_samples (int): The number of samples to use from the calibration dataset.
        block_height (int): The block height for Block Floating-Point (BFP) quantization.
        block_width (int): The block width for Block Floating-Point (BFP) quantization.
        mantissa_bits (int): The number of mantissa bits for BFP quantization.
        device (str, optional): The device to perform quantization on ('auto', 'cpu', 'cuda').
            Defaults to "auto".
        dry_run (bool, optional): Whether to skip saving the quantized model. Defaults to False.
    """
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading and preparing dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, dataset_config, split="train").select(range(num_samples))
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)

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

    print("Performing forward pass to get activations...")
    model.to(device)
    with torch.no_grad():
        model(tokens)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Identify target layers
    target_layers = []
    for name, module in model.named_modules():
        if "lm_head" in name or "embed_tokens" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            if name in activations:
                target_layers.append(name)
    
    print(f"Target layers for quantization ({len(target_layers)}):")
    for name in target_layers:
        print(f"  - {name}")

    print("Applying AWQ quantization...")
    awq_scales = {}
    for name, module in model.named_modules():
        if name in target_layers:
            print(f"Quantizing layer: {name}")
            X = activations[name]
            # 將活化值展平為 [N, in_features]以符合矩陣乘法
            X = X.view(-1, X.shape[-1])
            W = module.weight.data
            
            # 搜尋最優縮放向量 s
            s = search_awq_scale(W, X, block_height=block_height, block_width=block_width)
            awq_scales[name] = s.clone().cpu()
            
            # 套用縮放並量化 (bits 使用傳入的 mantissa_bits)
            W_scaled = apply_awq_scale(W, s)
            q_W_scaled = pseudo_quantize(W_scaled, n_bits=mantissa_bits, block_height=block_height, block_width=block_width)
            
            # 還原縮放以維持 output 維度正確 (Pseudo-quantization 模式)
            # W_final = Q(W*s) / s
            W_final = q_W_scaled / s.view(1, -1)
            module.weight.data = W_final
        elif isinstance(module, torch.nn.Linear) and ("lm_head" not in name and "embed_tokens" not in name):
            print(f"Skipping layer {name} as no activation was captured.")


    if not dry_run:
        print("Saving quantized model...")
        output_dir = f"{model_path}-awq-quantized-fix-precision-bh{block_height}-bw{block_width}-m{mantissa_bits}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata and scales
        metadata = {
            "quantized_layers": target_layers,
            "config": {
                "block_height": block_height,
                "block_width": block_width,
                "mantissa_bits": mantissa_bits,
            }
        }
        with open(os.path.join(output_dir, "awq_config.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        torch.save(awq_scales, os.path.join(output_dir, "awq_scales.pt"))
        print(f"Saved AWQ metadata and scales to: {output_dir}")

        # Convert to bf16 to save disk space
        model = model.to(torch.bfloat16)
        model.save_pretrained(output_dir, torch_dtype=torch.bfloat16)
        tokenizer.save_pretrained(output_dir)
        print(f"Quantized model saved to: {output_dir} (bf16 format)")
    else:
        print("Dry run complete; not saving model.")

def main():
    """Main entry point for the AWQ fixed-precision quantization script.

    This function serves as the command-line interface for the quantization
    workflow. It parses user arguments for model selection, dataset configuration,
    and quantization parameters.

    The script performs a single forward pass to collect activations and then
    iterates through a user-specified list of mantissa bit settings. For each
    setting, it creates a fresh copy of the model, applies AWQ, and saves the
    result, allowing for efficient sweeps over different precision levels.
    """
    parser = argparse.ArgumentParser(description="AWQ Fix-Precision Quantization CLI")
    parser.add_argument("--model", type=str, default="llama-3.2-1b",
                        help="Model preset or path. Presets: llama-3.2-1b, tinyllama")
    parser.add_argument("--dataset", type=str, default="Salesforce/wikitext",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1",
                        help="Dataset config name if applicable")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples for calibration")
    parser.add_argument("--block-height", type=int, default=1,
                        help="BFP block height")
    parser.add_argument("--block-width", type=int, default=64,
                        help="BFP block width")
    parser.add_argument("--mantissa-bits", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="Mantissa bits to try, e.g., --mantissa-bits 2 3 4 5")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Computation device")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving models")
    parser.add_argument("--eval-ppl", action="store_true", help="Evaluate WikiText2 PPL")
    parser.add_argument("--eval-hellaswag", action="store_true", help="Evaluate HellaSwag Accuracy")
    parser.add_argument("--max-samples-hs", type=int, default=None, help="Max samples for HellaSwag evaluation")
    parser.add_argument("--limit-tokens-ppl", type=int, default=None, help="Limit tokens for PPL evaluation for speed")

    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    print(f"Resolved model path: {model_path}")

    print("Loading model and tokenizer for activation capture...")
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    # Prepare dataset and tokens once
    dataset = load_dataset(args.dataset, args.dataset_config, split="train").select(range(args.num_samples))
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)

    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = input[0].detach()
        return hook

    hooks = []
    for name, module in base_model.named_modules():
        if "lm_head" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(get_activation(name)))

    with torch.no_grad():
        base_model(tokens)

    for h in hooks:
        h.remove()

    for m in args.mantissa_bits:
        print("----------------------------------------------------------------")
        print(f"Running AWQ fix-precision with Mantissa Bits = {m}, Block Height = {args.block_height}, Block Width = {args.block_width}")
        print("----------------------------------------------------------------")

        # Work on a copy to avoid accumulating quantization across runs
        model_copy = copy.deepcopy(base_model)
        # Identify target layers
        target_layers = []
        for name, module in model_copy.named_modules():
            if "lm_head" in name or "embed_tokens" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                if name in activations:
                    target_layers.append(name)
        
        print(f"Target layers for quantization ({len(target_layers)}):")
        # for name in target_layers:
        #     print(f"  - {name}")

        awq_scales = {}
        for name, module in model_copy.named_modules():
            if name in target_layers:
                print(f"Quantizing layer: {name}")
                X = activations[name]
                # 將活化值展平為 [N, in_features] 以符合權重矩陣乘法
                X = X.view(-1, X.shape[-1])
                W = module.weight.data
                
                # 搜尋最優縮放向量 s
                s = search_awq_scale(W, X, block_height=args.block_height, block_width=args.block_width)
                awq_scales[name] = s.clone().cpu()
                
                # 套用縮放並量化
                W_scaled = apply_awq_scale(W, s)
                q_W_scaled = pseudo_quantize(W_scaled, n_bits=m, block_height=args.block_height, block_width=args.block_width)
                
                # 還原縮放以維持推論輸出不變 (Pseudo-quantization 模式)
                # W_final = Q(W*s) / s
                W_final = q_W_scaled / s.view(1, -1)
                module.weight.data = W_final
            elif isinstance(module, torch.nn.Linear) and ("lm_head" not in name and "embed_tokens" not in name):
                print(f"Skipping layer {name} as no activation was captured.")

        if args.eval_ppl:
            TU.evaluate_ppl(model_copy, tokenizer, device, limit_tokens=args.limit_tokens_ppl)
        
        if args.eval_hellaswag:
            TU.evaluate_hellaswag(model_copy, tokenizer, device, max_samples=args.max_samples_hs)

        if not args.dry_run:
            output_dir = f"{model_path}-awq-quantized-fix-precision-bh{args.block_height}-bw{args.block_width}-m{m}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save metadata and scales
            metadata = {
                "quantized_layers": target_layers,
                "config": {
                    "block_height": args.block_height,
                    "block_width": args.block_width,
                    "mantissa_bits": m,
                }
            }
            with open(os.path.join(output_dir, "awq_config.json"), "w") as f:
                json.dump(metadata, f, indent=4)
            torch.save(awq_scales, os.path.join(output_dir, "awq_scales.pt"))
            print(f"Saved AWQ metadata and scales to: {output_dir}")

            # Convert to bf16 to save disk space
            model_copy = model_copy.to(torch.bfloat16)
            model_copy.save_pretrained(output_dir, torch_dtype=torch.bfloat16)
            tokenizer.save_pretrained(output_dir)
            print(f"Quantized model saved to: {output_dir} (bf16 format)")
        else:
            print("Dry run complete; not saving model.")

    print("\nAll quantization tasks are complete.")


if __name__ == '__main__':
    main()
