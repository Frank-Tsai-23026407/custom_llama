import argparse
import copy
import os
import sys
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from custom_model_only_quantize.utils.activation_utils import prepare_calibration_data, get_layer_activations, get_multiple_layers_activations
from awq_utils import awq_fix_precision_quantize_2d

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

    print("Loading model and tokenizer for quantization...")
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    # 1. Prepare packed calibration data (CPU tensors)
    # Using default seq_len=512 from default argument if needed, or hardcode based on safe practice
    print("Preparing packed calibration data...")
    packed_samples = prepare_calibration_data(tokenizer, args.dataset, num_samples=args.num_samples, seq_len=512)

    for m in args.mantissa_bits:
        print("----------------------------------------------------------------")
        print(f"Running AWQ fix-precision with Mantissa Bits = {m}, Block Height = {args.block_height}, Block Width = {args.block_width}")
        print("----------------------------------------------------------------")

        # Work on a copy to avoid accumulating quantization across runs
        model_copy = copy.deepcopy(base_model)
        model_copy.eval()
        
        awq_scales = {}
        target_layers = [] # For metadata

        # Helper to group layers by block (defined once per quantization run or could be global)
        def get_layer_groups(model):
            groups = {}
            others = []
            import re
            pattern = re.compile(r'\.(layers|h|blocks)\.(\d+)\.')
            
            for name, module in model.named_modules():
                if "lm_head" in name or "embed_tokens" in name:
                    continue
                if isinstance(module, torch.nn.Linear):
                    match = pattern.search(name)
                    if match:
                        # group_key is e.g. "model.layers.0"
                        group_key = name[:match.end()-1]
                        if group_key not in groups:
                            groups[group_key] = []
                        groups[group_key].append(name)
                    else:
                        others.append(name)
            return groups, others

        layer_groups, other_layers = get_layer_groups(model_copy)
        
        # Sort groups by index
        sorted_group_keys = sorted(layer_groups.keys(), key=lambda k: int(k.split('.')[-1]))
        
        # Combined list of tasks
        all_tasks = [layer_groups[k] for k in sorted_group_keys] + [[name] for name in other_layers]
        
        print(f"grouped layers into {len(all_tasks)} tasks (blocks + others) to optimize forward passes.")

        awq_scales = {}
        target_layers = []

        # Iterate through groups (Blocks)
        for group in tqdm(all_tasks, desc="Quantizing Blocks"):
            # 1. Get activations for ALL layers in this group at once
            # using base_model for clean activations
            group_activations = get_multiple_layers_activations(base_model, packed_samples, group, device=device)
            
            for name in group:
                if name not in group_activations or group_activations[name] is None:
                    print(f"Warning: No activations for {name}")
                    continue
                    
                target_layers.append(name)
                
                X = group_activations[name]
                X = X.view(-1, X.shape[-1])
                X = X.to(device)
                
                module = model_copy.get_submodule(name)
                W = module.weight.data
                
                # 3. Quantize using the utility function (DRY)
                W_final = awq_fix_precision_quantize_2d(
                    W, X, 
                    block_height=args.block_height, 
                    block_width=args.block_width, 
                    mantissa_bits=m
                )
                
                module.weight.data = W_final.to(W.dtype)
                
                # Free individual activation and intermediate tensors
                del X, W_final
            
            # Free dict
            del group_activations
            torch.cuda.empty_cache()


        if args.eval_ppl:
            TU.evaluate_ppl(model_copy, tokenizer, device, limit_tokens=args.limit_tokens_ppl)
        
        if args.eval_hellaswag:
            TU.evaluate_hellaswag(model_copy, tokenizer, device, max_samples=args.max_samples_hs)

        if not args.dry_run:
            output_dir = f"{model_path}-awq-quantized-fix-precision-bh{args.block_height}-bw{args.block_width}-m{m}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save metadata
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
            # awq_scales is not returned by the utility function, so we skip saving it.
            print(f"Saved AWQ metadata to: {output_dir}")

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
