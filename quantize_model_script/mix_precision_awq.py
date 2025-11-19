import argparse
import copy
import os
import sys
import torch
# Add the parent directory to the Python path to find the 'block_quantization' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantize_model_script.block_quantization import block_floating_point_quantize
from quantize_model_script.activation_aware_weight_quantization import get_weight_scaling_factor

def mix_precision_mixed_precision_bfp(
    weight: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 128,
    mantissa_bits: int = 4
) -> torch.Tensor:
    """Applies a mixed-precision BFP quantization strategy to a weight tensor.

    This function implements a form of mixed-precision quantization where, within each
    block of weights, one "salient" weight is kept in its original full precision,
    while the remaining weights in the block are quantized using BFP.

    The saliency of a weight is determined by the magnitude of its corresponding input
    activation, which is provided by the `scales` tensor. The weight connected to the
    input feature with the highest average activation is preserved. This technique aims
    to protect the most influential weights from quantization error.

    Args:
        weight (torch.Tensor): The 2D weight tensor of shape (out_features, in_features).
        scales (torch.Tensor): A 2D tensor of shape (1, in_features) containing the
            average activation magnitudes for each input feature.
        block_size (int, optional): The size of the quantization blocks. Defaults to 128.
        mantissa_bits (int, optional): The number of mantissa bits for BFP. Defaults to 4.

    Returns:
        torch.Tensor: The reconstructed weight tensor with mixed precision.
    """
    assert weight.dim() == 2, "Weight tensor must be 2D"
    assert scales.dim() == 2 and scales.shape[0] == 1, "Scales tensor must be 2D with shape (1, in_features)"
    assert weight.shape[1] == scales.shape[1], "in_features of weight and scales must match"

    original_shape = weight.shape
    out_features, in_features = original_shape

    # Flatten the weight tensor for block processing
    flat_weight = weight.flatten()
    num_elements = flat_weight.numel()

    # 1. Create a corresponding flattened tensor of activation scales
    flat_scales = scales.repeat(out_features, 1).flatten()

    # 2. Pad both weights and scales to be divisible by block_size
    padding_needed = (block_size - (num_elements % block_size)) % block_size
    if padding_needed > 0:
        flat_weight = torch.nn.functional.pad(flat_weight, (0, padding_needed))
        flat_scales = torch.nn.functional.pad(flat_scales, (0, padding_needed), value=0)

    num_blocks = flat_weight.numel() // block_size

    # Reshape into blocks for finding important indices
    scale_blocks = flat_scales.view(num_blocks, block_size)

    # 3. Find the index of the most important weight in each block
    important_indices_in_block = torch.argmax(scale_blocks, dim=1)
    # Convert block indices to indices in the flattened tensor
    important_indices_flat = important_indices_in_block + torch.arange(num_blocks, device=weight.device) * block_size

    # 4. Create a temporary tensor for quantization, zeroing out important weights
    temp_flat_weight = flat_weight.clone()
    # Store the original important weights before zeroing them out
    original_important_weights = temp_flat_weight[important_indices_flat].clone()
    # Zero out the important weights so they don't affect the shared exponent
    temp_flat_weight[important_indices_flat] = 0.0

    # 5. Quantize the temporary tensor
    # The shared exponents are now calculated correctly without the influence of the FP32 weights.
    quantized_temp_flat_weight = block_floating_point_quantize(
        temp_flat_weight,
        block_size=block_size,
        mantissa_bits=mantissa_bits
    )

    # 6. Restore the original, full-precision important weights in the quantized tensor
    quantized_temp_flat_weight[important_indices_flat] = original_important_weights

    reconstructed_flat = quantized_temp_flat_weight

    # 7. Reshape and remove padding
    reconstructed_weight_padded = reconstructed_flat.view(num_blocks, block_size)
    flat_weight_padded = flat_weight.view(num_blocks, block_size)
    mse = torch.mean((flat_weight_padded[:, :-padding_needed] - reconstructed_weight_padded[:, :-padding_needed]) ** 2) if padding_needed > 0 else torch.mean((flat_weight - reconstructed_flat) ** 2)
    print(f"    - Mix-Precision Mixed-Precision MSE (excluding padding): {mse.item():.8f}")

    if padding_needed > 0:
        reconstructed_flat = reconstructed_flat[:-padding_needed]

    reconstructed_weight = reconstructed_flat.view(original_shape)

    return reconstructed_weight

############################################################################################
# for tiny llama model weight loading
############################################################################################
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import copy
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quantize_model_script.activation_aware_weight_quantization import get_weight_scaling_factor


def resolve_model_path(model: str) -> str:
    """
    Resolves a preset model keyword to its full path or returns the path if it exists.

    This function supports preset keywords for commonly used models, making it easier
    to specify model paths. If the provided string is not a preset, it is assumed to
    be a direct path to the model.

    Args:
        model (str): The model keyword or path.

    Returns:
        str: The resolved full path to the model.
    """
    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    presets = {
        "llama-3.2-1b": os.path.join(ws_root, "model/llama-3.2-1b/Llama-3.2-1B"),
        "tinyllama": os.path.join(ws_root, "model/tinyllama/TinyLlama_1.1v"),
    }
    if model in presets:
        return presets[model]
    return model

def main():
    """
    The main entry point for the AWQ mix-precision quantization script.

    This function parses command-line arguments, loads the model and dataset, and then
    iterates through a list of mantissa bit settings, applying mix-precision AWQ
    quantization for each setting and saving the resulting models.
    """
    parser = argparse.ArgumentParser(description="AWQ Mix-Precision Quantization CLI")
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
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model.to(device)

    print(f"Loading and preparing dataset: {args.dataset}...")
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
        print(f"Running AWQ mix-precision with Mantissa Bits = {m}, Block Size = {args.block_size}")
        print("----------------------------------------------------------------")

        temp_model = copy.deepcopy(base_model)
        for name, module in temp_model.named_modules():
            if "lm_head" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                if name in activations:
                    print(f"Quantizing layer: {name}")
                    scales = get_weight_scaling_factor(activations[name])
                    q_weight = mix_precision_mixed_precision_bfp(
                        module.weight.data,
                        scales,
                        block_size=args.block_size,
                        mantissa_bits=m,
                    )
                    module.weight.data = q_weight
                else:
                    print(f"Skipping layer {name} as no activation was captured.")

        if not args.dry_run:
            print("Saving quantized model...")
            output_dir = f"{model_path}-awq-quantized-mix-precision-b{args.block_size}-m{m}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to bf16 to save disk space
            temp_model = temp_model.to(torch.bfloat16)
            temp_model.save_pretrained(output_dir, torch_dtype=torch.bfloat16)
            tokenizer.save_pretrained(output_dir)
            print(f"Quantized model saved to: {output_dir} (bf16 format)")
        else:
            print("Dry run complete; not saving model.")

    print("\nAll quantization tasks are complete.")


if __name__ == "__main__":
    main()
