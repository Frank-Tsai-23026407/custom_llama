import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from multiplying_matrix_without_multiplying.maddness import MaddnessLayer, MaddnessTrainer

class MaddnessOutlierLayer(MaddnessLayer):
    def __init__(self, original_layer, num_subspaces=4, tree_depth=4, outlier_k=32):
        super().__init__(original_layer, num_subspaces, tree_depth)
        self.outlier_k = outlier_k
        # Store original weights for exact computation of outliers
        # We need W in [out, in] format usually, but for column gathering it's easier if we have it accessible.
        # original_layer.weight is [out_features, in_features]
        self.register_buffer('original_weight_t', original_layer.weight.detach().t().clone()) # [in, out]
        
    def forward(self, x):
        # x shape: [batch, seq_len, in_features] or [batch, in_features]
        original_shape = x.shape
        x_flat = x.view(-1, self.in_features)
        batch_size = x_flat.shape[0]
        
        # 1. Identify Outliers
        # Find top k magnitude indices per sample
        # x_abs = x_flat.abs()
        # topk_values, topk_indices = torch.topk(x_abs, self.outlier_k, dim=1)
        
        # We need the original values, not absolute
        # But topk is based on magnitude
        # So we gather from x_flat using topk_indices
        
        # Optimization: If k is 0, skip
        if self.outlier_k > 0:
            topk_abs, topk_indices = torch.topk(x_flat.abs(), self.outlier_k, dim=1)
            
            # Create a mask for outliers to zero them out in the base input
            # scatter_ requires index to have same dims
            mask = torch.zeros_like(x_flat, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)
            
            # Zero out outliers for MADDNESS part
            x_maddness = torch.where(mask, torch.tensor(0.0, device=x.device, dtype=x.dtype), x_flat)
            
            # Get outlier values
            # We can gather them
            outlier_vals = torch.gather(x_flat, 1, topk_indices) # [B, k]
            
        else:
            x_maddness = x_flat
            
        # 2. MADDNESS Approximation
        # Call super().forward but we need to handle the reshaping carefully
        # super().forward expects [..., in]
        # We pass flattened [B, in]
        output_maddness = super().forward(x_maddness) # [B, out]
        
        # 3. Exact Outlier Computation
        if self.outlier_k > 0:
            output_exact = torch.zeros_like(output_maddness)
            
            # We need to compute: sum_{k} (val_k * W_{row_k})
            # W is [out, in], W.t is [in, out]
            # We have indices [B, k]
            
            # Iterate over k to save memory/complexity
            for k in range(self.outlier_k):
                idx = topk_indices[:, k] # [B]
                val = outlier_vals[:, k] # [B]
                
                # Gather weights: W.t()[idx] -> [B, out]
                w_rows = self.original_weight_t[idx] 
                
                output_exact += val.unsqueeze(1) * w_rows
                
            output = output_maddness + output_exact
        else:
            output = output_maddness
            
        return output.view(*original_shape[:-1], self.out_features)

class MaddnessOutlierTrainer(MaddnessTrainer):
    def __init__(self, in_features, num_subspaces=4, tree_depth=4, outlier_percent=0.05):
        super().__init__(in_features, num_subspaces, tree_depth)
        self.outlier_percent = outlier_percent
        
    def train(self, X_calib, W):
        # Pre-process X_calib to remove outliers
        # "remove the 5% outlier on each channel while learning"
        
        print(f"Filtering top {self.outlier_percent*100}% outliers for training...")
        
        # Calculate thresholds per dimension
        # X_calib: [N, in]
        # We want 95th percentile of absolute value
        
        # For speed, if N is large, we can sample
        if X_calib.shape[0] > 10000:
            X_sample = X_calib[:10000]
        else:
            X_sample = X_calib
            
        thresholds = torch.quantile(X_sample.abs(), 1.0 - self.outlier_percent, dim=0) # [in]
        
        # Clip X_calib to these thresholds
        # This effectively "removes" the outlier influence by capping them
        # Alternatively, we could set them to 0 or mean, but clipping preserves the sign and "max inlier" value
        # which is good for the tree to know the boundary.
        
        # Expand thresholds to [1, in]
        thresholds = thresholds.unsqueeze(0)
        
        # Clip
        X_clipped = torch.clamp(X_calib, min=-thresholds, max=thresholds)
        
        # Now train using the clipped data
        # The trees and prototypes will be learned on the "inlier" distribution
        split_indices, split_thresholds, lookup_table, prototypes, mse = super().train(X_clipped, W)
        
        # Visualization (Only for the first call/layer to avoid spam)
        if not hasattr(self, 'has_visualized'):
            self.has_visualized = True
            self.visualize_quantization(X_calib, X_clipped, split_indices, split_thresholds, prototypes)
            
        return split_indices, split_thresholds, lookup_table, mse

    def visualize_quantization(self, X_orig, X_clipped, split_indices, split_thresholds, prototypes):
        import matplotlib.pyplot as plt
        import os
        
        # Reconstruct X_hat from X_clipped (since prototypes are learned on X_clipped)
        # We use the learned trees and prototypes
        
        # Take first 100 samples
        N_vis = min(100, X_orig.shape[0])
        X_vis = X_clipped[:N_vis]
        X_orig_vis = X_orig[:N_vis]
        
        X_hat = torch.zeros_like(X_vis)
        
        for s in range(self.num_subspaces):
            start_dim = s * self.subspace_dim
            end_dim = (s + 1) * self.subspace_dim
            X_sub = X_vis[:, start_dim:end_dim]
            
            # Tree traversal
            curr_nodes = torch.zeros(N_vis, dtype=torch.long, device=X_vis.device)
            for d in range(self.tree_depth):
                s_indices = split_indices[s][curr_nodes]
                s_thresholds = split_thresholds[s][curr_nodes]
                
                vals = torch.gather(X_sub, 1, s_indices.unsqueeze(1)).squeeze(1)
                decision = (vals >= s_thresholds).long()
                curr_nodes = 2 * curr_nodes + 1 + decision
                
            leaf_indices = curr_nodes - (2**self.tree_depth - 1)
            
            # Get prototypes
            # prototypes: [num_subspaces, num_prototypes, sub_dim]
            P = prototypes[s] # [num_prototypes, sub_dim]
            X_sub_hat = P[leaf_indices] # [N_vis, sub_dim]
            
            X_hat[:, start_dim:end_dim] = X_sub_hat
            
        # Plotting
        # We can plot a few channels of the first sample
        # Or plot a scatter of Original vs Quantized values
        
        X_orig_np = X_orig_vis.detach().cpu().numpy().flatten()
        X_clipped_np = X_vis.detach().cpu().numpy().flatten()
        X_hat_np = X_hat.detach().cpu().numpy().flatten()
        
        plt.figure(figsize=(15, 5))
        
        # 1. Histogram of values
        plt.subplot(1, 3, 1)
        plt.hist(X_orig_np, bins=50, alpha=0.5, label='Original', log=True)
        plt.hist(X_hat_np, bins=50, alpha=0.5, label='Quantized', log=True)
        plt.legend()
        plt.title('Value Distribution (Log Scale)')
        
        # 2. Scatter Plot (Original vs Quantized)
        plt.subplot(1, 3, 2)
        # Downsample for scatter if too many points
        mask = np.random.choice(len(X_orig_np), size=min(2000, len(X_orig_np)), replace=False)
        plt.scatter(X_orig_np[mask], X_hat_np[mask], alpha=0.3, s=1)
        plt.plot([X_orig_np.min(), X_orig_np.max()], [X_orig_np.min(), X_orig_np.max()], 'r--', alpha=0.5)
        plt.xlabel('Original Value')
        plt.ylabel('Quantized Value')
        plt.title('Original vs Quantized')
        
        # 3. First Sample Vector
        plt.subplot(1, 3, 3)
        # Plot first 100 dims of first sample
        dims = min(100, X_orig.shape[1])
        plt.plot(X_orig_np[:dims], label='Original', alpha=0.7)
        plt.plot(X_hat_np[:dims], label='Quantized', alpha=0.7)
        plt.legend()
        plt.title(f'First Sample (First {dims} dims)')
        
        os.makedirs('maddness_remove_outlier/plots', exist_ok=True)
        plt.savefig('maddness_remove_outlier/plots/quantization_vis.png')
        plt.close()
        print("Saved visualization to maddness_remove_outlier/plots/quantization_vis.png")
