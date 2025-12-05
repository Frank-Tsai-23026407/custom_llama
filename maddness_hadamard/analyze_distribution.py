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

# Suppress specific warnings
warnings.filterwarnings("ignore", message="var(): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message="Tight layout not applied")

def get_hadamard_matrix(n, device='cpu'):
    """
    Generates a normalized Hadamard matrix of size n x n using Sylvester's construction.
    
    Args:
        n (int): Size of the matrix. Must be a power of 2.
        device (str): Device to create the tensor on.
        
    Returns:
        torch.Tensor: Normalized Hadamard matrix (orthogonal).
    """
    # Assumes n is power of 2
    if n == 1:
        return torch.ones(1, 1, device=device)
    
    h_n_2 = get_hadamard_matrix(n // 2, device=device)
    top = torch.cat([h_n_2, h_n_2], dim=1)
    bottom = torch.cat([h_n_2, -h_n_2], dim=1)
    h_n = torch.cat([top, bottom], dim=0)
    return h_n / torch.sqrt(torch.tensor(2, dtype=torch.float32, device=device)) # Normalized

def plot_3d_activations(activations, tokens, title, save_path, z_lim=None):
    """
    Plots activations in a 3D visualization where:
    - X-axis: Tokens (Sequence dimension)
    - Y-axis: Channels (Feature dimension)
    - Z-axis: Activation Value
    
    Args:
        activations (torch.Tensor): Tensor of shape (seq_len, hidden_dim).
        tokens (list): List of token strings corresponding to the sequence.
        title (str): Title of the plot.
        save_path (str): File path to save the plot image.
        z_lim (tuple, optional): Fixed range for Z-axis (min, max).
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    seq_len, hidden_dim = activations.shape
    
    # Calculate RMS
    rms = torch.sqrt(torch.mean(activations.float() ** 2)).item()
    title = f"{title}\nRMS: {rms:.4f}"
    
    x_data = np.arange(seq_len)
    y_data = np.arange(hidden_dim)
    
    for i in range(seq_len):
        # x is constant = i
        # y is 0 to hidden_dim
        # z is activation values
        
        xs = np.full(hidden_dim, i)
        ys = np.arange(hidden_dim)
        zs = activations[i].cpu().numpy()
        
        ax.plot(xs, ys, zs, color='b', alpha=0.6)

    ax.set_xlabel('Tokens')
    ax.set_ylabel('Channels')
    ax.set_zlabel('Activation')
    ax.set_title(title)
    
    if z_lim:
        ax.set_zlim(z_lim)
    
    # Set x-ticks to tokens
    ax.set_xticks(np.arange(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

 

class MaddnessQuantizer:
    """
    A simple implementation of the Maddness quantization method for analysis purposes.
    
    This class simulates the training and inference of Maddness quantization:
    1.  Splits the input dimension into `num_subspaces`.
    2.  For each subspace, builds a decision tree to cluster data into `2^tree_depth` buckets.
    3.  Computes a prototype (centroid) for each bucket.
    4.  Quantizes data by mapping each vector to its corresponding prototype.
    """
    def __init__(self, num_subspaces=4, tree_depth=4):
        """
        Args:
            num_subspaces (int): Number of subspaces to split the input vector into.
            tree_depth (int): Depth of the decision tree for each subspace. 
                              Number of prototypes per subspace = 2^tree_depth.
        """
        self.num_subspaces = num_subspaces
        self.tree_depth = tree_depth
        self.split_dims = []      # Stores split dimension indices for each subspace [num_subspaces, num_nodes]
        self.split_threshs = []   # Stores split thresholds for each subspace [num_subspaces, num_nodes]
        self.prototypes = []      # Stores centroids for each leaf [num_subspaces, num_leaves, subspace_dim]
        
    def fit(self, X):
        """
        Trains the quantizer on the provided dataset X.
        
        Args:
            X (torch.Tensor): Training data of shape [N, D].
        """
        N, D = X.shape
        subspace_dim = D // self.num_subspaces
        
        self.split_dims = []
        self.split_threshs = []
        self.prototypes = []
        
        for s in range(self.num_subspaces):
            start_dim = s * subspace_dim
            end_dim = (s + 1) * subspace_dim
            X_sub = X[:, start_dim:end_dim]
            
            # 1. Build Tree & Store Splits
            # This recursively finds the best dimension and threshold to split the data
            indices, thresholds, buckets = self._build_tree(X_sub)
            self.split_dims.append(indices)
            self.split_threshs.append(thresholds)
            
            # 2. Compute Prototypes (Centroids)
            # Calculate the mean of all samples falling into each leaf node
            num_leaves = 2**self.tree_depth
            protos = torch.zeros(num_leaves, subspace_dim, device=X.device)
            
            for leaf_idx, sample_idxs in buckets.items():
                if len(sample_idxs) > 0:
                    protos[leaf_idx] = torch.mean(X_sub[sample_idxs], dim=0)
            
            self.prototypes.append(protos)
            
    def transform(self, X):
        """
        Quantizes and reconstructs the input X using the trained codebooks.
        
        Args:
            X (torch.Tensor): Input data of shape [N, D].
            
        Returns:
            X_hat (torch.Tensor): Reconstructed (quantized) data of shape [N, D].
        """
        N, D = X.shape
        subspace_dim = D // self.num_subspaces
        X_hat = torch.zeros_like(X)
        
        for s in range(self.num_subspaces):
            start_dim = s * subspace_dim
            end_dim = (s + 1) * subspace_dim
            X_sub = X[:, start_dim:end_dim]
            
            # Traverse Tree to find leaf index for each sample
            curr_nodes = torch.zeros(N, dtype=torch.long, device=X.device)
            for d in range(self.tree_depth):
                # Retrieve split parameters for the current node of each sample
                s_indices = self.split_dims[s][curr_nodes]
                s_thresholds = self.split_threshs[s][curr_nodes]
                
                # Compare feature value with threshold
                vals = torch.gather(X_sub, 1, s_indices.unsqueeze(1)).squeeze(1)
                decision = (vals >= s_thresholds).long()
                
                # Move to left (2*i+1) or right (2*i+2) child
                curr_nodes = 2 * curr_nodes + 1 + decision
                
            # Convert tree node index to leaf index (0 to 2^depth - 1)
            leaf_indices = curr_nodes - (2**self.tree_depth - 1)
            
            # Look up prototypes
            X_hat[:, start_dim:end_dim] = self.prototypes[s][leaf_indices]
            
        return X_hat

    def _build_tree(self, X):
        """
        Helper function to build the decision tree for a single subspace.
        Uses a greedy approach to minimize variance at each split.
        """
        n_split_nodes = 2 ** self.tree_depth - 1
        split_indices = torch.zeros(n_split_nodes, dtype=torch.long, device=X.device)
        split_thresholds = torch.zeros(n_split_nodes, device=X.device)
        buckets = {0: torch.arange(X.shape[0], device=X.device)}
        
        final_buckets = {} # Maps leaf_idx -> sample_idxs
        
        for node_idx in range(n_split_nodes):
            if node_idx not in buckets or len(buckets[node_idx]) == 0:
                continue
            
            sample_idxs = buckets[node_idx]
            X_node = X[sample_idxs]
            
            # Find best split dimension (max variance) and threshold (mean)
            variances = torch.var(X_node, dim=0)
            best_dim = torch.argmax(variances)
            best_thresh = torch.mean(X_node[:, best_dim])
            
            split_indices[node_idx] = best_dim
            split_thresholds[node_idx] = best_thresh
            
            # Split data
            vals = X_node[:, best_dim]
            left_mask = vals < best_thresh
            right_mask = ~left_mask
            
            left_child = 2 * node_idx + 1
            right_child = 2 * node_idx + 2
            
            # Assign to children
            if left_child < n_split_nodes:
                buckets[left_child] = sample_idxs[left_mask]
                buckets[right_child] = sample_idxs[right_mask]
            else:
                # Children are leaves
                leaf_left = left_child - n_split_nodes
                leaf_right = right_child - n_split_nodes
                final_buckets[leaf_left] = sample_idxs[left_mask]
                final_buckets[leaf_right] = sample_idxs[right_mask]
                
        return split_indices, split_thresholds, final_buckets


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

def apply_block_hadamard(X, block_size=512):
    """
    Applies Hadamard transform in blocks of size `block_size`.
    X: [..., dim]
    """
    dim = X.shape[-1]
    assert dim % block_size == 0, f"Dimension {dim} must be divisible by block size {block_size}"
    
    # Generate H for the block
    H_block = get_hadamard_matrix(block_size, device=X.device) # [B, B]
    
    # Reshape X to [..., num_blocks, block_size]
    original_shape = X.shape
    num_blocks = dim // block_size
    X_reshaped = X.view(*original_shape[:-1], num_blocks, block_size)
    
    # Apply H to each block: X' = X @ H.T
    # Here X is [..., num_blocks, block_size]
    # We want to multiply the last dim by H.T
    
    X_transformed = X_reshaped @ H_block.t()
    
    # Flatten back
    return X_transformed.view(*original_shape)

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
        "model.layers.10.mlp.down_proj",
        "model.layers.20.mlp.down_proj",
        "model.layers.20.self_attn.o_proj"
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
    
    tree_depths = [4, 5, 6, 7]
    os.makedirs("maddness_hadamard/plots", exist_ok=True)

    # 3. Iterate through different tree depths to analyze quantization quality vs depth
    for depth in tree_depths:
        print(f"\n=== Analyzing Tree Depth {depth} ===")
        
        for layer_name in target_layers:
            if layer_name not in train_activations_dict or not train_activations_dict[layer_name]:
                continue
                
            print(f"Processing {layer_name}...")
            
            # Prepare Training Data (Concatenate all collected samples)
            X_train = torch.cat(train_activations_dict[layer_name], dim=1).squeeze(0).view(-1, train_activations_dict[layer_name][0].shape[-1]).float()
            
            # Prepare Sample Data (Single sentence)
            X_sample = sample_activations_dict[layer_name][0].squeeze(0).float()
            
            # --- A. Original Maddness Quantization ---
            # Train quantizer on raw activations
            quantizer_orig = MaddnessQuantizer(num_subspaces=4, tree_depth=depth)
            quantizer_orig.fit(X_train)
            
            # Plot Original Vector (Only once, at depth 4, as it doesn't change with depth)
            if depth == 4:
                plot_3d_activations(X_sample, tokens, f"1. Original\n{layer_name}", f"maddness_hadamard/plots/{layer_name}_1_original.png")

            # Quantize Sample using the trained Original Quantizer
            X_sample_hat = quantizer_orig.transform(X_sample)
            plot_3d_activations(X_sample_hat, tokens, f"4. Only Maddness (Value) D{depth}\n{layer_name}", f"maddness_hadamard/plots/{layer_name}_depth{depth}_4_maddness_val.png")
            
            # Plot Error (Original - Quantized)
            X_sample_err = X_sample - X_sample_hat
            plot_3d_activations(X_sample_err, tokens, f"4. Only Maddness (Error) D{depth}\n{layer_name}", f"maddness_hadamard/plots/{layer_name}_depth{depth}_4_maddness_err.png", z_lim=(-0.5, 0.5))

            # --- B. Hadamard + Maddness Quantization ---
            dim = X_train.shape[1]
            BLOCK_SIZE = 512
            if dim % BLOCK_SIZE == 0:
                # 1. Transform Training Data using Block Hadamard
                X_train_had = apply_block_hadamard(X_train, block_size=BLOCK_SIZE)
                
                # 2. Train Quantizer on Transformed Data
                # The quantizer now learns to quantize the "mixed" features
                quantizer_had = MaddnessQuantizer(num_subspaces=4, tree_depth=depth)
                quantizer_had.fit(X_train_had)
                
                # 3. Transform Sample Data
                X_sample_had = apply_block_hadamard(X_sample, block_size=BLOCK_SIZE)
                
                if depth == 4:
                    plot_3d_activations(X_sample_had, tokens, f"2. Hadamard (Block {BLOCK_SIZE})\n{layer_name}", f"maddness_hadamard/plots/{layer_name}_2_hadamard.png")
                
                # 4. Quantize Transformed Sample
                X_sample_had_hat = quantizer_had.transform(X_sample_had)
                plot_3d_activations(X_sample_had_hat, tokens, f"3. Hadamard -> Maddness (Value) D{depth}\n{layer_name}", f"maddness_hadamard/plots/{layer_name}_depth{depth}_3_had_maddness_val.png")
                
                # Plot Error in Transformed Space
                X_sample_had_err = X_sample_had - X_sample_had_hat
                plot_3d_activations(X_sample_had_err, tokens, f"3. Hadamard -> Maddness (Error) D{depth}\n{layer_name}", f"maddness_hadamard/plots/{layer_name}_depth{depth}_3_had_maddness_err.png", z_lim=(-0.5, 0.5))
            else:
                if depth == 4: print(f"Skipping Hadamard for {layer_name} (dim {dim})")

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
            X_had = apply_block_hadamard(X, block_size=BLOCK_SIZE)
            
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
            BLOCK_SIZE = 512
            if in_dim % BLOCK_SIZE == 0:
                # W' = W @ H.T (if we transform input X -> X @ H.T)
                # Wait, Y = W X.T
                # If X' = X H.T, then X = X' H
                # Y = W (X' H).T = W H.T X'.T
                # So W' = W H.T
                # But here we use block diagonal H.
                # W is [out, in]. We need to apply H to the 'in' dimension.
                # W_reshaped: [out, num_blocks, block_size]
                
                num_blocks = in_dim // BLOCK_SIZE
                W_reshaped = W.view(W.shape[0], num_blocks, BLOCK_SIZE)
                H_block = get_hadamard_matrix(BLOCK_SIZE, device=W.device)
                
                # W' = W_reshaped @ H.T
                W_had = W_reshaped @ H_block.t()
                W_had = W_had.view(W.shape[0], in_dim)
                
                kurtosis_had = scipy.stats.kurtosis(W_had.flatten().cpu().numpy())
                max_val_had = W_had.abs().max().item()
                min_val_had = W_had.min().item()
                print(f"  Hadamard (Block {BLOCK_SIZE}): Kurtosis={kurtosis_had:.2f}, Min={min_val_had:.2f}, Max={max_val_had:.2f}")
                results_file.write(f"{name} Weight Hadamard: Kurtosis={kurtosis_had:.2f}, Min={min_val_had:.2f}, Max={max_val_had:.2f}\n")
            else:
                 print(f"  Skipping Hadamard for Weight {name} (in_dim {in_dim} not divisible by {BLOCK_SIZE})")

    results_file.close()
    print("Analysis complete. Results saved to analysis_results.txt")

if __name__ == "__main__":
    main()
