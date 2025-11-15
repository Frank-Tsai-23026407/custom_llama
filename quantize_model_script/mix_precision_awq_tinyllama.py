import torch
import os
import sys
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
    """
    Quantizes a weight matrix using a mix-precision mixed-precision BFP approach.

    For each block of weights, it identifies the weight corresponding to the
    input feature with the highest activation magnitude (importance) and keeps
    it in full floating-point precision. The rest of the weights in the block
    are quantized using standard BFP.

    Args:
        weight (torch.Tensor): The input weight matrix of shape (out_features, in_features).
        scales (torch.Tensor): A tensor of shape (1, in_features) representing the
                               average magnitude (importance) of each input activation feature.
        block_size (int): The number of elements in each quantization block.
        mantissa_bits (int): The number of mantissa bits for BFP quantization.

    Returns:
        torch.Tensor: The reconstructed weight matrix with mixed precision.
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

if __name__ == "__main__":
    model_path = "/home/frank23026407/TinyLlama/model/tinyllama/TinyLlmam_1.1v"
    dataset_name = "Salesforce/wikitext"
    dataset_config = "wikitext-103-raw-v1"
    num_samples = 128
    block_size = 64
    
    # load quantization and model
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load dataset
    print(f"Loading and preparing dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, dataset_config, split="train").select(range(num_samples))
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids
    print(f"Training text: {text[:30]}")
    
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
    with torch.no_grad():
        model(tokens)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()   
        
    for mantissa_bits in range(2, 6):
        print("----------------------------------------------------------------")
        print(f"Running AWQ quantization with Mantissa Bits = {mantissa_bits}, Block Size = {block_size}")
        print("----------------------------------------------------------------")
        # performing mix-precision quantization
        temp_model = copy.deepcopy(model)
        for name, module in temp_model.named_modules():
            if "lm_head" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                if name in activations:
                    print(f"Quantizing layer: {name}")
                    # Calculate scales from activations before quantizing
                    scales = get_weight_scaling_factor(activations[name])
                    q_weight = mix_precision_mixed_precision_bfp(module.weight.data, scales, block_size=block_size, mantissa_bits=mantissa_bits)
                    module.weight.data = q_weight
                else:
                    print(f"Skipping layer {name} as no activation was captured.")
                    
        print("Saving quantized model...")
        output_dir = f"{model_path}-awq-quantized-mix-precision-b{block_size}-m{mantissa_bits}"
        os.makedirs(output_dir, exist_ok=True)
        temp_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Quantized model saved to: {output_dir}")
    
    print("\nAll quantization tasks are complete.")
