
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantize_model_script.activation_aware_weight_quantization import awq_quantize

def quantize_model(model_path, dataset_name, dataset_config, num_samples, block_size, mantissa_bits):
    """
    Quantizes a model using Activation-aware Weight Quantization (AWQ).

    This function loads a pre-trained model, collects activations using a calibration dataset,
    and then applies AWQ to quantize the model's linear layers. The quantized model is
    saved to a new directory.

    Args:
        model_path (str): The path to the pre-trained model to be quantized.
        dataset_name (str): The name of the dataset to use for calibration (e.g., "wikitext").
        dataset_config (str): The specific configuration of the dataset to use.
        num_samples (int): The number of samples from the dataset to use for calibration.
        block_size (int): The block size to be used for block floating-point (BFP) quantization.
        mantissa_bits (int): The number of mantissa bits for BFP quantization.
    """
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading and preparing dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, dataset_config, split="train").select(range(num_samples))
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids

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

if __name__ == '__main__':
    """
    The main entry point for the TinyLlama fix-precision AWQ quantization script.

    This script loads the TinyLlama model, collects activations using a calibration
    dataset, and then applies fix-precision AWQ quantization for a range of mantissa
    bit settings. The resulting quantized models are saved to separate directories.
    """
    MODEL_PATH = "/home/frank23026407/TinyLlama/model/tinyllama/TinyLlmam_1.1v"
    DATASET_NAME = "Salesforce/wikitext"
    DATASET_CONFIG = "wikitext-103-raw-v1"
    NUM_SAMPLES = 128
    BLOCK_SIZE = 64  # Fixed block size

    # Loop through mantissa bits from 2 to 5
    for mantissa_bits in range(2, 6):
        print("----------------------------------------------------------------")
        print(f"Running AWQ quantization with Mantissa Bits = {mantissa_bits}, Block Size = {BLOCK_SIZE}")
        print("----------------------------------------------------------------")
        quantize_model(MODEL_PATH, DATASET_NAME, DATASET_CONFIG, NUM_SAMPLES, BLOCK_SIZE, mantissa_bits)

    print("\nAll quantization tasks are complete.")
