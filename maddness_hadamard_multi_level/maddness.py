import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from maddness_hadamard_multi_level.utils import plot_3d_activations
import matplotlib.pyplot as plt

def pseudo_codebook_quantize_func(x):
    return x
    

class MaddnessQuantizerMultiLevel:
    """
    Multi-level Maddness Quantizer.
    
    This class implements a residual quantization scheme using Maddness.
    Level 1: Quantize X -> X_hat1. Residual R1 = X - X_hat1.
    Level 2: Quantize R1 -> X_hat2. Residual R2 = R1 - X_hat2.
    ...
    Reconstruction: X_hat = X_hat1 + X_hat2 + ...
    """
    def __init__(self, subspace_dim=512, num_levels=2, tree_depth=4, hidden_dim=2048, codebook_quantize_func=None):
        """
        Args:
            subspace_dim (int): Dimension of each subspace.
            num_levels (int): Number of residual quantization levels.
            tree_depth (int): Depth of the decision tree for each subspace.
            hidden_dim (int): The dimension of the input activation.
        """
        self.subspace_dim = subspace_dim
        self.num_levels = num_levels
        self.tree_depth = tree_depth
        self.hidden_dim = hidden_dim
        self.num_subspaces = hidden_dim // subspace_dim
        self.codebook_quantize_func = codebook_quantize_func
        assert self.hidden_dim % self.num_subspaces == 0
        
        # Storage for each level's parameters
        # Each element is a list of parameters for the subspaces in that level
        self.levels_split_dims = []      # List[List[Tensor]] # (level, subspace)
        self.levels_split_threshs = []   # List[List[Tensor]] # (level, subspace)
        self.levels_prototypes = []      # List[List[Tensor]] # (level, subspace)
        
    def fit(self, X, regulization_lambda=1):
        """
        Trains the multi-level quantizer on the provided dataset X.
        
        Args:
            X (torch.Tensor): Training data of shape [N, D].
        """
        N, D = X.shape
        assert D == self.hidden_dim
        
        current_residual = X.clone()
        
        self.levels_split_dims = []
        self.levels_split_threshs = []
        self.levels_prototypes = []
        
        for l in range(self.num_levels):
            # print(f"Training Level {l+1}/{self.num_levels}...")
            level_dims = []
            level_threshs = []
            level_protos = []
            
            # We need to reconstruct the approximation for this level to update residual
            level_approximation = torch.zeros_like(current_residual)
            
            for s in range(self.num_subspaces):
                start_dim = s * self.subspace_dim
                end_dim = (s + 1) * self.subspace_dim
                X_sub = current_residual[:, start_dim:end_dim]
                
                # 1. Build Tree
                indices, thresholds, buckets = self._build_tree(X_sub)
                level_dims.append(indices)
                level_threshs.append(thresholds)
                
                # 2. Compute Prototypes
                num_leaves = 2**self.tree_depth
                
                # Optimize Prototypes using Ridge Regression
                N = X_sub.shape[0]
                G = torch.zeros(N, num_leaves, device=X.device)
                bucket_assignments = self._get_leaf_indices(X_sub, indices, thresholds)
                G.scatter_(1, bucket_assignments.unsqueeze(1), 1.0)
                
                I = torch.eye(num_leaves, device=X.device)
                GtG = G.t() @ G
                GtX = G.t() @ X_sub
                protos = torch.linalg.solve(GtG + regulization_lambda * I, GtX)

                # 3. Quantize Prototypes
                if self.codebook_quantize_func is not None:
                    protos = self.codebook_quantize_func(protos)

                level_protos.append(protos)
                
                # 4. Reconstruct for this subspace to update residual
                # Get assignments
                leaf_indices = self._get_leaf_indices(X_sub, indices, thresholds)
                level_approximation[:, start_dim:end_dim] = protos[leaf_indices]
                
            self.levels_split_dims.append(level_dims)
            self.levels_split_threshs.append(level_threshs)
            self.levels_prototypes.append(level_protos)
            
            # Update residual
            current_residual = current_residual - level_approximation
            
            # mse = torch.mean(current_residual ** 2).item()
            # print(f"  Level {l+1} Residual MSE: {mse:.6f}")

    def transform(self, X, tokens=None, plot_original=False, plot_quantized=False, plot_residual=False, plot_prefix=""):
        """
        Quantizes and reconstructs the input X using the trained codebooks.
        
        Args:
            X (torch.Tensor): Input data of shape [N, D].
            
        Returns:
            X_hat (torch.Tensor): Reconstructed (quantized) data of shape [N, D].
        """
        N, D = X.shape
        assert D == self.hidden_dim
        
        total_reconstruction = torch.zeros_like(X)
        residual = X.clone()
        residuals_history = []
        
        for l in range(self.num_levels):
            level_reconstruction = torch.zeros_like(X)
            
            for s in range(self.num_subspaces):
                start_dim = s * self.subspace_dim
                end_dim = (s + 1) * self.subspace_dim
                X_sub = residual[:, start_dim:end_dim]
                
                indices = self.levels_split_dims[l][s]
                thresholds = self.levels_split_threshs[l][s]
                protos = self.levels_prototypes[l][s]
                
                leaf_indices = self._get_leaf_indices(X_sub, indices, thresholds)
                level_reconstruction[:, start_dim:end_dim] = protos[leaf_indices]
            
            total_reconstruction += level_reconstruction
            residual = residual - level_reconstruction
            
            if plot_prefix:
                residuals_history.append(residual.clone())
            
        if plot_prefix:
            self._plot(X, total_reconstruction, residuals_history, tokens, plot_prefix)
            
        return total_reconstruction

    def _plot(self, original, quantized, residuals, tokens, plot_prefix):
        num_plots = 2 + len(residuals)
        # Increase height to 8 to prevent title clipping
        fig = plt.figure(figsize=(6 * num_plots, 8))
        
        # 1. Plot Original
        ax = fig.add_subplot(1, num_plots, 1, projection='3d')
        self._plot_on_ax(ax, original, tokens, "Original")
        
        # 2. Plot Quantized
        ax = fig.add_subplot(1, num_plots, 2, projection='3d')
        self._plot_on_ax(ax, quantized, tokens, "Quantized")
        
        # 3. Plot Residuals
        for i, res in enumerate(residuals):
            ax = fig.add_subplot(1, num_plots, 3 + i, projection='3d')
            self._plot_on_ax(ax, res, tokens, f"Residual L{i+1}", z_lim=(-0.5, 0.5))
            
        plt.tight_layout()
        # Adjust top margin to ensure titles are not clipped
        plt.subplots_adjust(top=0.85)
        plt.savefig(f"{plot_prefix}.png")
        plt.close()

    def _plot_on_ax(self, ax, activations, tokens, title, z_lim=None):
        seq_len, hidden_dim = activations.shape
        rms = torch.sqrt(torch.mean(activations.float() ** 2)).item()
        title = f"{title}\nRMS: {rms:.4f}"
        
        for i in range(seq_len):
            xs = np.full(hidden_dim, i)
            ys = np.arange(hidden_dim)
            zs = activations[i].cpu().numpy()
            ax.plot(xs, ys, zs, color='b', alpha=0.6)

        ax.set_xlabel('Tokens', labelpad=10)
        ax.set_ylabel('Channels', labelpad=10)
        ax.set_zlabel('Activation', labelpad=10)
        ax.set_title(title, pad=20)
        
        if z_lim:
            ax.set_zlim(z_lim)
        
        if tokens:
            # Ensure we only show ticks for available tokens
            tick_indices = np.arange(seq_len)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([tokens[i] for i in tick_indices], rotation=45, ha='right', fontsize=8)
            
        # Adjust viewing angle for better visibility
        ax.view_init(elev=30, azim=-60)

    def _build_tree(self, X):
        """
        Helper function to build the decision tree for a single subspace.
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
            
            # Find best split dimension and threshold
            best_dim, best_thresh = self._find_best_split(X_node)
            
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

    def _find_best_split(self, X, top_n=4):
        """
        Finds the best split dimension and threshold by minimizing SSE.
        """
        variances = torch.var(X, dim=0)
        top_n_dims = torch.topk(variances, min(top_n, X.shape[1])).indices
        best_dim, best_thresh, best_sse = None, None, float('inf')
        
        for dim in top_n_dims:
            vals = X[:, dim]
            # Use quantiles for candidate thresholds
            quantiles = torch.quantile(vals, torch.linspace(0.1, 0.9, 9, device=X.device))
            thresholds = torch.unique(quantiles)
            
            for thresh in thresholds:
                left_mask = vals < thresh
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0: continue
                
                vals_left = vals[left_mask]
                vals_right = vals[right_mask]
                sse = torch.sum((vals_left - torch.mean(vals_left))**2) + torch.sum((vals_right - torch.mean(vals_right))**2)
                
                if sse < best_sse:
                    best_sse = sse
                    best_dim = dim
                    best_thresh = thresh
                    
        if best_dim is None:
            best_dim = top_n_dims[0]
            best_thresh = torch.mean(X[:, best_dim])
            
        return best_dim, best_thresh

    def _get_leaf_indices(self, X, indices, thresholds):
        """
        Traverses the tree to find leaf indices for input X.
        """
        curr_nodes = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        n_split_nodes = 2 ** self.tree_depth - 1
        
        for d in range(self.tree_depth):
            s_indices = indices[curr_nodes]
            s_thresholds = thresholds[curr_nodes]
            
            vals = torch.gather(X, 1, s_indices.unsqueeze(1)).squeeze(1)
            decision = (vals >= s_thresholds).long()
            
            curr_nodes = 2 * curr_nodes + 1 + decision
            
        return curr_nodes - n_split_nodes
