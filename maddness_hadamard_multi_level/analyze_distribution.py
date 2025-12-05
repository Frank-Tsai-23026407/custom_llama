import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import scipy.stats
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from maddness_hadamard_multi_level.utils import plot_3d_activations, get_hadamard_matrix, apply_block_hadamard
from maddness_hadamard_multi_level.maddness import MaddnessQuantizerMultiLevel

# Suppress specific warnings
warnings.filterwarnings("ignore", message="var(): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message="Tight layout not applied")


def collect_activations(model, tokenizer, data, target_layers, num_samples=None, seq_len=512):
    """
    Collects activations from specified layers for a given dataset or single text.
    
    Args:
        model: The model to run.
        tokenizer: The tokenizer.
        data: A list of strings (dataset) or a single string.
        target_layers: List of layer names to capture activations from.
        num_samples: Max number of samples to process (if data is a list).
        seq_len: Max sequence length.
        
    Returns:
        activations: Dict {layer_name: [tensor_sample_1, tensor_sample_2, ...]}
        collected_input_ids: List of input_id tensors corresponding to processed samples.
    """
    if isinstance(data, str):
        data = [data]
        
    print(f"Collecting activations for {len(target_layers)} layers...")
    activations = {name: [] for name in target_layers}
    
    def hook_fn(name):
        def hook(module, input, output):
            # input[0] is [batch, seq, dim]
            if name in activations:
                activations[name].append(input[0].detach().cpu())
        return hook
    
    handles = []
    found_layers = []
    for name, module in model.named_modules():
        if name in target_layers:
            handles.append(module.register_forward_hook(hook_fn(name)))
            found_layers.append(name)
            
    if not handles:
        print(f"Could not find any of {target_layers}")
        return {}, []
        
    # Check if we missed any layers
    missed = set(target_layers) - set(found_layers)
    if missed:
        print(f"Warning: Could not find layers: {missed}")

    model.eval()
    collected_input_ids = []
    count = 0
    
    with torch.no_grad():
        for text in data:
            if num_samples is not None and count >= num_samples:
                break
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            if inputs.input_ids.shape[1] == 0: continue
            
            collected_input_ids.append(inputs.input_ids.cpu())
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)
            count += 1
            
    for h in handles: h.remove()
    
    return activations, collected_input_ids

def main():
    model_path = "model/tinyllama/TinyLlama_1.1v"
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    
    # Specific sample analysis for 3D plot
    text = "Summer is warm . Winter is cold ."
    print(f"Analyzing sample: '{text}'")
    
    # Pick layers likely to contain outliers
    target_layers = [
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.1.self_attn.k_proj",
        "model.layers.1.self_attn.v_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.self_attn.o_proj",
        "model.layers.1.mlp.down_proj",
        "model.layers.1.mlp.up_proj",
        "model.layers.1.mlp.gate_proj",
        "model.layers.2.self_attn.k_proj",
        "model.layers.2.self_attn.v_proj",
        "model.layers.2.self_attn.q_proj",
        "model.layers.2.self_attn.o_proj",
        "model.layers.2.mlp.down_proj",
        "model.layers.2.mlp.up_proj",
        "model.layers.2.mlp.gate_proj",
        "model.layers.10.self_attn.k_proj",
        "model.layers.10.self_attn.v_proj",
        "model.layers.10.self_attn.q_proj",
        "model.layers.10.self_attn.o_proj",
        "model.layers.10.mlp.down_proj",
        "model.layers.10.mlp.up_proj",
        "model.layers.10.mlp.gate_proj",
        "model.layers.20.self_attn.k_proj",
        "model.layers.20.self_attn.v_proj",
        "model.layers.20.self_attn.q_proj",
        "model.layers.20.self_attn.o_proj",
        "model.layers.20.mlp.down_proj",
        "model.layers.20.mlp.up_proj",
        "model.layers.20.mlp.gate_proj",
        "model.layers.21.self_attn.k_proj",
        "model.layers.21.self_attn.v_proj",
        "model.layers.21.self_attn.q_proj",
        "model.layers.21.self_attn.o_proj",
        "model.layers.21.mlp.down_proj",
        "model.layers.21.mlp.up_proj",
        "model.layers.21.mlp.gate_proj",
    ]
    print(f"Targeting layers: {target_layers}")

    # 1. Collect Bulk Training Data (for building codebooks)
    # We collect activations from a subset of the Wikitext dataset.
    # These activations will be used to train the MaddnessQuantizer (build trees and centroids).
    # This ensures the quantization is based on the general data distribution, not just the single sample we plot.
    print("\nLoading dataset for training quantizers...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    dataset = [x["text"] for x in dataset if len(x["text"]) > 20]
    
    print("Collecting training activations...")
    train_activations_dict, _ = collect_activations(model, tokenizer, dataset, target_layers, num_samples=50) # 50 samples for training
    
    # 2. Collect Specific Sample (for plotting)
    # We collect activations for a specific sentence to visualize how the quantization affects a single input.
    print(f"\nCollecting sample activations for: '{text}'")
    sample_activations_dict, input_ids_list = collect_activations(model, tokenizer, text, target_layers)
    tokens = [tokenizer.decode(t) for t in input_ids_list[0][0]]
    
    tree_depths = [4]
    plot_directory = "maddness_hadamard_multi_level/plots"
    os.makedirs(plot_directory, exist_ok=True)

    
    # 3. Iterate through the model layers
    for layer_name in target_layers:
        if layer_name not in train_activations_dict or not train_activations_dict[layer_name]:
            continue
            
        print(f"Processing {layer_name}...")
        
        # Prepare Training Data (Concatenate all collected samples)
        X_train = torch.cat(train_activations_dict[layer_name], dim=1).squeeze(0).view(-1, train_activations_dict[layer_name][0].shape[-1]).float()
        
        # Prepare Sample Data (Single sentence)
        X_sample = sample_activations_dict[layer_name][0].squeeze(0).float()
        
        # Plot Original Vector (Only once, at depth 4, as it doesn't change with depth)
        plot_3d_activations(X_sample, tokens, f"1. Original\n{layer_name}", f"{plot_directory}/{layer_name}_1_original.png")

        # --- A. Original Maddness Quantization ---
        # Train quantizer on raw activations
        hidden_dim = X_train.shape[1]
        subspace_dim = 64
        
        for depth in tree_depths:
            print(f"\n=== Analyzing Tree Depth {depth} ===")    
        
            quantizer_orig = MaddnessQuantizerMultiLevel(subspace_dim=subspace_dim, num_levels=2, tree_depth=depth, hidden_dim=hidden_dim)
            quantizer_orig.fit(X_train)
        
            # Quantize Sample using the trained Original Quantizer
            X_sample_hat = quantizer_orig.transform(X_sample, tokens=tokens, plot_prefix=f"{plot_directory}/{layer_name}_depth{depth}_4_maddness_val_combined")
        
            # --- B. Hadamard + Maddness Quantization ---
            dim = X_train.shape[1]
            BLOCK_SIZE = 64
            if dim % BLOCK_SIZE == 0:
                # 1. Transform Training Data using Block Hadamard
                H_block = get_hadamard_matrix(BLOCK_SIZE, device=X_train.device)
                X_train_had = apply_block_hadamard(X_train, H_block)
                
                # 2. Train Quantizer on Transformed Data
                # The quantizer now learns to quantize the "mixed" features
                quantizer_had = MaddnessQuantizerMultiLevel(subspace_dim=subspace_dim, num_levels=2, tree_depth=depth, hidden_dim=dim)
                quantizer_had.fit(X_train_had)
                
                # 3. Transform Sample Data
                X_sample_had = apply_block_hadamard(X_sample, H_block)
                
                # 4. Quantize Transformed Sample
                X_sample_had_hat = quantizer_had.transform(X_sample_had, tokens=tokens, plot_prefix=f"{plot_directory}/{layer_name}_depth{depth}_4_had_maddness_val_combined")
                
            else:
                if depth == tree_depths[0]: print(f"Skipping Hadamard for {layer_name} (dim {dim})")
        
        # Close all figures to prevent "More than 20 figures have been opened" warning
        plt.close('all')

    print("Plots saved to maddness_hadamard/plots/")

    # 2. Bulk Analysis
    print("\nLoading dataset for bulk analysis...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    dataset = [x["text"] for x in dataset if len(x["text"]) > 20]
    
    # Identify ALL linear layers for bulk analysis
    all_linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "model.layers" in name:
            all_linear_layers.append(name)
            
    print(f"Analyzing all {len(all_linear_layers)} linear layers...")
    raw_activations, _ = collect_activations(model, tokenizer, dataset, all_linear_layers, num_samples=20)
    
    # Concatenate for stats
    activations = {}
    for name, acts in raw_activations.items():
        if acts:
            activations[name] = torch.cat(acts, dim=1).squeeze(0).view(-1, acts[0].shape[-1])
    
    results_file = open("maddness_hadamard/analysis_results.txt", "w")
    
    for name, X in activations.items():
        print(f"Analyzing {name}...")
        X = X.float()
        dim = X.shape[1]
        
        # Original stats
        kurtosis_orig = scipy.stats.kurtosis(X.flatten().numpy())
        max_val_orig = X.abs().max().item()
        min_val_orig = X.min().item()
        print(f"  Original: Kurtosis={kurtosis_orig:.2f}, Min={min_val_orig:.2f}, Max={max_val_orig:.2f}")
        results_file.write(f"{name} Original: Kurtosis={kurtosis_orig:.2f}, Min={min_val_orig:.2f}, Max={max_val_orig:.2f}\n")
        
        # Block Hadamard Transform
        BLOCK_SIZE = 512
        if dim % BLOCK_SIZE == 0:
            H_block = get_hadamard_matrix(BLOCK_SIZE, device=X.device)
            X_had = apply_block_hadamard(X, H_block)
            
            kurtosis_had = scipy.stats.kurtosis(X_had.flatten().numpy())
            max_val_had = X_had.abs().max().item()
            min_val_had = X_had.min().item()
            print(f"  Hadamard (Block {BLOCK_SIZE}): Kurtosis={kurtosis_had:.2f}, Min={min_val_had:.2f}, Max={max_val_had:.2f}")
            results_file.write(f"{name} Hadamard: Kurtosis={kurtosis_had:.2f}, Min={min_val_had:.2f}, Max={max_val_had:.2f}\n")
        else:
            print(f"  Skipping Hadamard for {name} (dim {dim} not divisible by {BLOCK_SIZE})")

    # Analyze Weights
    print("\nAnalyzing Weights...")
    for name, module in model.named_modules():
        if name in activations: # Only analyze weights for layers we analyzed activations for
            print(f"Analyzing Weight {name}...")
            W = module.weight.data.float() # [out, in]
            
            # Original
            kurtosis_orig = scipy.stats.kurtosis(W.flatten().cpu().numpy())
            max_val_orig = W.abs().max().item()
            min_val_orig = W.min().item()
            print(f"  Original: Kurtosis={kurtosis_orig:.2f}, Min={min_val_orig:.2f}, Max={max_val_orig:.2f}")
            results_file.write(f"{name} Weight Original: Kurtosis={kurtosis_orig:.2f}, Min={min_val_orig:.2f}, Max={max_val_orig:.2f}\n")
            
            # Block Hadamard
            in_dim = W.shape[1]
            out_dim = W.shape[0]
            BLOCK_SIZE = 512
            
            if in_dim % BLOCK_SIZE == 0:
                # 1. Apply H_in to Input Dimension
                # W' = W @ H_in.T
                num_blocks_in = in_dim // BLOCK_SIZE
                W_reshaped = W.view(out_dim, num_blocks_in, BLOCK_SIZE)
                H_block = get_hadamard_matrix(BLOCK_SIZE, device=W.device)
                
                W_had = W_reshaped @ H_block.t()
                W_had = W_had.view(out_dim, in_dim)
                
                # 2. Apply H_out to Output Dimension (if applicable)
                # W'' = H_out @ W'
                if out_dim % BLOCK_SIZE == 0:
                    H_block_out = get_hadamard_matrix(BLOCK_SIZE, device=W.device)
                    # We need to apply H_block_out to the first dimension of W_had
                    # W_had is [out, in]
                    # Reshape to [num_blocks_out, BLOCK_SIZE, in]
                    num_blocks_out = out_dim // BLOCK_SIZE
                    W_had_reshaped = W_had.view(num_blocks_out, BLOCK_SIZE, in_dim)
                    
                    # Apply H_block_out [B, B] to the block dimension [B, in]
                    # Result should be [num_blocks_out, B, in]
                    # W_final[b, :, :] = H_block_out @ W_had_reshaped[b, :, :]
                    W_final = torch.matmul(H_block_out, W_had_reshaped)
                    W_final = W_final.view(out_dim, in_dim)
                else:
                    W_final = W_had
                    print(f"  Skipping H_out for Weight {name} (out_dim {out_dim} not divisible by {BLOCK_SIZE})")
                
                kurtosis_had = scipy.stats.kurtosis(W_final.flatten().cpu().numpy())
                max_val_had = W_final.abs().max().item()
                min_val_had = W_final.min().item()
                print(f"  Hadamard (Block {BLOCK_SIZE}): Kurtosis={kurtosis_had:.2f}, Min={min_val_had:.2f}, Max={max_val_had:.2f}")
                results_file.write(f"{name} Weight Hadamard: Kurtosis={kurtosis_had:.2f}, Min={min_val_had:.2f}, Max={max_val_had:.2f}\n")
            else:
                print(f"  Skipping Hadamard for Weight {name} (in_dim {in_dim} not divisible by {BLOCK_SIZE})")

    results_file.close()
    print("Analysis complete. Results saved to analysis_results.txt")

if __name__ == "__main__":
    main()
