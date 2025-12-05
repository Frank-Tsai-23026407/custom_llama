import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import argparse
from typing import Optional
import sys

# Ensure the current directory is in sys.path so that pickle can find 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Copying _dequantize_weight from src/utils.py to make this script standalone-ish
# or we could import it. Let's import to avoid duplication if possible, 
# but the user asked for "from scratch", so explicit implementation is better for demonstration.

def dequantize_weight(codes, codebooks, scales=None):
    """
    Decode float weights from quantization codes.
    """
    num_out_groups, num_in_groups, num_codebooks = codes.shape[-3:]
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    out_features = num_out_groups * out_group_size
    in_features = num_in_groups * in_group_size
    
    codebook_offsets = torch.arange(
        0, num_codebooks * codebook_size, codebook_size, device=codes.device
    )
    
    # codes: [..., num_out_groups, num_in_groups, num_codebooks]
    # codebooks: [num_codebooks, codebook_size, out_group_size, in_group_size]
    
    # Flatten codes to look up in flattened codebooks
    # codes.flatten(0, -2) -> [..., num_codebooks]
    # + offsets -> indices into flattened codebooks
    
    reconstructed_weight_flat = F.embedding_bag(
        codes.flatten(0, -2) + codebook_offsets, 
        codebooks.flatten(0, 1).flatten(-2, -1), 
        mode="sum"
    ) 
    
    reconstructed_weight_groupwise = reconstructed_weight_flat.view(
        list(codes.shape[:-3]) + [num_out_groups, num_in_groups, out_group_size, in_group_size]
    )
    
    if scales is not None:
        reconstructed_weight_groupwise = reconstructed_weight_groupwise.mul(scales)
        
    return reconstructed_weight_groupwise.swapaxes(-3, -2).reshape(list(codes.shape[:-3]) + [out_features, in_features])

def load_and_dequantize_model(model_path, quantized_path, device="cuda"):
    print(f"Loading config from {model_path}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    print("Initializing empty model...")
    # Initialize model with meta device to save memory, then move to device
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    model = model.to_empty(device=device)
    
    # Load non-quantized weights (embeddings, norms, head)
    print("Loading non-quantized weights...")
    not_quantized_weights = torch.load(os.path.join(quantized_path, "not_quantized_weights.pt"), map_location=device)
    model.load_state_dict(not_quantized_weights, strict=False)
    
    # Iterate over layers and load quantized weights
    num_layers = config.num_hidden_layers
    
    for i in range(num_layers):
        layer_path = os.path.join(quantized_path, f"{i}.pth")
        if not os.path.exists(layer_path):
            print(f"Warning: Layer {i} not found at {layer_path}")
            continue
            
        print(f"Loading and dequantizing layer {i}...")
        quantized_layer_state = torch.load(layer_path, map_location=device)
        
        # The saved object is the whole layer (TransformerBlock) with QuantizedLinear modules
        # We need to extract the quantized weights and replace the weights in our model
        
        # We can traverse the saved layer to find QuantizedWeight instances
        # But since we initialized a fresh model, we need to map the saved structure to the model structure.
        # The saved layer IS a torch.nn.Module (the layer itself), so we can iterate its named modules.
        
        current_layer = model.model.layers[i]
        
        # Helper to find QuantizedLinear/Weight in the loaded layer
        for name, module in quantized_layer_state.named_modules():
            if name == "": continue
            
            # Check if it's a quantized linear layer (has quantized_weight)
            if hasattr(module, "quantized_weight") and isinstance(module.quantized_weight, torch.nn.Module):
                # This is a QuantizedLinear. The actual weights are in module.quantized_weight
                qw = module.quantized_weight
                
                # Extract components
                codes = qw.get_codes()
                codebooks = qw.get_codebooks()
                scales = qw.get_scales()
                
                # Dequantize
                print(f"  Dequantizing {name}...")
                weight = dequantize_weight(codes, codebooks, scales)
                
                # Place into current_layer
                # name is relative to the layer, e.g. "self_attn.q_proj"
                
                # Navigate to the submodule in current_layer
                submodule = current_layer
                parts = name.split('.')
                for part in parts:
                    submodule = getattr(submodule, part)
                
                # Assign weight
                # Note: Transpose might be needed depending on how Linear stores weights vs how dequantize returns
                # Linear weights are [out, in]. dequantize returns [out, in]. So it should match.
                # However, if the saved module was a custom QuantizedLinear, we need to be careful.
                # The AQLM code replaces Linear with QuantizedLinear.
                
                submodule.weight.data = weight.to(dtype=submodule.weight.dtype)
                
                # Bias
                if module.bias is not None:
                    submodule.bias.data = module.bias.data.to(dtype=submodule.bias.dtype)
                    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--quantized_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello, my name is")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_and_dequantize_model(args.model_path, args.quantized_path, args.device)
    
    model.eval()
    
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    
    print(f"Generating for prompt: '{args.prompt}'")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
        
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
