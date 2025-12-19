import argparse
import copy
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantize_model_script.activation_aware_weight_quantization import awq_quantize


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
    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    presets = {
        "llama-3.2-1b": os.path.join(ws_root, "model/llama-3.2-1b/Llama-3.2-1B"),
        "tinyllama": os.path.join(ws_root, "model/tinyllama/TinyLlama_1.1v"),
    }
    if model in presets:
        return presets[model]
    # If user passed a path, return as-is
    return model

def quantize_model(model_path, dataset_name, dataset_config, num_samples, block_size, mantissa_bits, device="auto"):
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
        block_size (int): The block size for Block Floating-Point (BFP) quantization.
        mantissa_bits (int): The number of mantissa bits for BFP quantization.
        device (str, optional): The device to perform quantization on ('auto', 'cpu', 'cuda').
            Defaults to "auto".
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

    print("Applying AWQ quantization...")
    for name, module in model.named_modules():
        if "lm_head" in name:
          continue
        if isinstance(module, torch.nn.Linear):
            if name in activations:
                print(f"Quantizing layer: {name}")
                q_weight = awq_quantize(module.weight.data, activations[name], block_size, mantissa_bits)
                module.weight.data = q_weight
            else:
                print(f"Skipping layer {name} as no activation was captured.")


    print("Saving quantized model...")
    output_dir = f"{model_path}-awq-quantized-fix-precision-b{block_size}-m{mantissa_bits}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to bf16 to save disk space
    model = model.to(torch.bfloat16)
    model.save_pretrained(output_dir, torch_dtype=torch.bfloat16)
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized model saved to: {output_dir} (bf16 format)")

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
    parser.add_argument("--block-size", type=int, default=64,
                        help="BFP block size")
    parser.add_argument("--mantissa-bits", type=int, nargs="+", default=[2, 3, 4, 5],
                        help="Mantissa bits to try, e.g., --mantissa-bits 2 3 4 5")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Computation device")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving models")

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
        print(f"Running AWQ fix-precision with Mantissa Bits = {m}, Block Size = {args.block_size}")
        print("----------------------------------------------------------------")

        # Work on a copy to avoid accumulating quantization across runs
        model_copy = copy.deepcopy(base_model)
        for name, module in model_copy.named_modules():
            if "lm_head" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                if name in activations:
                    print(f"Quantizing layer: {name}")
                    q_weight = awq_quantize(module.weight.data, activations[name], args.block_size, m)
                    module.weight.data = q_weight
                else:
                    print(f"Skipping layer {name} as no activation was captured.")

        if not args.dry_run:
            output_dir = f"{model_path}-awq-quantized-fix-precision-b{args.block_size}-m{m}"
            os.makedirs(output_dir, exist_ok=True)
            
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
