import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class MaddnessLayer(nn.Module):
    def __init__(self, original_layer, num_subspaces=4, tree_depth=4):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.num_subspaces = num_subspaces
        self.tree_depth = tree_depth
        self.n_split_nodes = 2 ** tree_depth - 1
        self.num_prototypes = self.n_split_nodes + 1
        
        # Dimensions per subspace
        self.subspace_dim = self.in_features // self.num_subspaces
        
        # Hash function parameters (learned)
        # We need a tree for each subspace. 
        # Depth 4 tree -> 15 split nodes.
        # Store split indices and thresholds.
        self.tree_depth = tree_depth
        self.n_split_nodes = 2 ** tree_depth - 1
        self.register_buffer('split_indices', torch.zeros(self.num_subspaces, self.n_split_nodes, dtype=torch.long))
        self.register_buffer('split_thresholds', torch.zeros(self.num_subspaces, self.n_split_nodes))
        
        # Lookup table T = P @ W (learned)
        # Shape: [num_subspaces, num_prototypes, out_features]
        self.register_buffer('lookup_table', torch.zeros(self.num_subspaces, self.num_prototypes, self.out_features))
        
        # Bias (optional)
        if original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias.clone())
        else:
            self.register_buffer('bias', None)

        
    def forward(self, x):
        # x shape: [batch, seq_len, in_features] or [batch, in_features]
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        batch_size = x.shape[0]
        
        output = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        
        # Iterate over subspaces
        for s in range(self.num_subspaces):
            # Extract subspace input
            start_dim = s * self.subspace_dim
            end_dim = (s + 1) * self.subspace_dim
            x_sub = x[:, start_dim:end_dim]
            
            # Hashing (Tree traversal)
            # This is a slow python implementation of the SIMD tree
            indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            
            # Level 1 (Root, node 0)
            # Indices in buffer are 0-indexed.
            # Tree nodes: 0 -> (1, 2) -> (3, 4, 5, 6) -> ...
            
            # To vectorize, we can't easily do the conditional branching per sample in Python efficiently without masks.
            # But since depth is small (4), we can just iterate levels.
            
            curr_nodes = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            
            for d in range(self.tree_depth): # Depth 4
                # Get split index and threshold for current nodes
                # We need to gather params because each sample might be at a different node
                # split_indices[s] has shape [15]
                
                # Gather split indices for the current nodes of all samples
                # curr_nodes ranges from 0 to 14 (internal nodes)
                # But wait, at depth d, the node indices are specific.
                # Actually, let's track the 'node index' in the array representation.
                # Root is 0. Children of i are 2*i+1 and 2*i+2.
                
                s_indices = self.split_indices[s][curr_nodes] # [batch]
                s_thresholds = self.split_thresholds[s][curr_nodes] # [batch]
                
                # Get values from x_sub
                # x_sub: [batch, subspace_dim]
                # s_indices: [batch] (values 0 to subspace_dim-1)
                
                # We need to gather x_sub values. 
                # torch.gather requires index to have same dims as input
                vals = torch.gather(x_sub, 1, s_indices.unsqueeze(1)).squeeze(1)
                
                # Compare
                decision = (vals >= s_thresholds).long()
                
                # Update node index
                # If decision is 0 (left), go to 2*i + 1
                # If decision is 1 (right), go to 2*i + 2
                # Note: The paper says "i <- 2i - 1 + b" but that assumes 1-based indexing?
                # Let's stick to 0-based: left=2i+1, right=2i+2.
                # Actually, the leaf indices are what we want.
                # At depth 4, we have traversed 4 splits.
                # Let's just accumulate the bits.
                
                # Alternative: Just track the tree index.
                curr_nodes = 2 * curr_nodes + 1 + decision
                
            # Now curr_nodes are in range [15, 30] (leaves)
            # Map to 0-15
            leaf_indices = curr_nodes - self.n_split_nodes
            
            # Lookup
            # lookup_table[s]: [16, out_features]
            # leaf_indices: [batch]
            
            val = self.lookup_table[s][leaf_indices] # [batch, out_features]
            
            output += val
            
        if self.bias is not None:
            output += self.bias
            
        return output.view(*original_shape[:-1], self.out_features)

class MaddnessTrainer:
    def __init__(self, in_features, num_subspaces=4, tree_depth=4):
        self.tree_depth = tree_depth
        self.n_split_nodes = 2 ** tree_depth - 1
        self.in_features = in_features
        self.num_subspaces = num_subspaces
        self.num_prototypes = self.n_split_nodes + 1
        self.subspace_dim = in_features // num_subspaces
        
    def train(self, X_calib, W):
        # X_calib: [N, in_features]
        # W: [out_features, in_features] (Standard PyTorch Linear weight shape is out, in)
        # But usually Y = X @ W.T. 
        # Let's assume W is passed as [in_features, out_features] for easier math, or transpose it.
        
        if W.shape[0] == self.in_features:
            W_t = W # [in, out]
        else:
            W_t = W.t() # [in, out]
             
        split_indices_all = []
        split_thresholds_all = []
        prototypes_all = []
        
        print("Training MADDNESS...")
        for s in range(self.num_subspaces):
            print(f"  Subspace {s+1}/{self.num_subspaces}")
            start_dim = s * self.subspace_dim
            end_dim = (s + 1) * self.subspace_dim
            
            X_sub = X_calib[:, start_dim:end_dim] # [N, sub_dim]
            
            # 1. Learn Hash Tree
            indices, thresholds, buckets = self._build_tree(X_sub)
            split_indices_all.append(indices)
            split_thresholds_all.append(thresholds)
            
            # 2. Optimize Prototypes
            # Construct G: [N, 16] (one-hot)
            N = X_sub.shape[0]
            G = torch.zeros(N, self.num_prototypes, device=X_calib.device)
            
            # Assign samples to buckets
            bucket_assignments = self._get_bucket_assignments(X_sub, indices, thresholds)
            G.scatter_(1, bucket_assignments.unsqueeze(1), 1.0)
            
            # Ridge Regression: P = (G.T @ G + lambda * I)^-1 @ G.T @ X_sub
            # We want to reconstruct X_sub with P.
            # P shape: [16, sub_dim]
            
            lamb = 1e-4
            I = torch.eye(self.num_prototypes, device=X_calib.device)
            
            # (G.T @ G) is [16, 16]
            GtG = G.t() @ G
            GtX = G.t() @ X_sub
            
            P = torch.linalg.solve(GtG + lamb * I, GtX)
            prototypes_all.append(P)
            
        # Stack results
        split_indices = torch.stack(split_indices_all) # [num_subspaces, 15]
        split_thresholds = torch.stack(split_thresholds_all) # [num_subspaces, 15]
        prototypes = torch.stack(prototypes_all) # [num_subspaces, 16, sub_dim]
        
        # Compute Lookup Table T = P @ W_sub
        # W_t: [in, out] -> split into subspaces
        lookup_table = []
        for s in range(self.num_subspaces):
            start_dim = s * self.subspace_dim
            end_dim = (s + 1) * self.subspace_dim
            W_sub = W_t[start_dim:end_dim, :] # [sub_dim, out]
            
            P = prototypes[s] # [16, sub_dim]
            T = P @ W_sub # [16, out]
            lookup_table.append(T)
            
        lookup_table = torch.stack(lookup_table)
        
        # 5. Compute reconstruction error
        # Approximation: sum of lookups
        # We need to perform the lookup on X using the trained tree
        
        # This logic is similar to forward pass but we can do it subspace by subspace
        # to save memory or just reuse the structure.
        
        # Let's do it efficiently here.
        
        # X is (N, in_features)
        # W is (in_features, out_features)
        # Y_true = X @ W
        
        Y_hat = torch.zeros(X_calib.shape[0], W.shape[1], device=X_calib.device, dtype=W.dtype)
        
        for s in range(self.num_subspaces):
            start_dim = s * self.subspace_dim
            end_dim = (s + 1) * self.subspace_dim
            X_sub = X_calib[:, start_dim:end_dim]
            
            # Find leaf indices for this subspace
            curr_nodes = torch.zeros(X_calib.shape[0], dtype=torch.long, device=X_calib.device)
            for d in range(self.tree_depth):
                s_indices = split_indices[s][curr_nodes]
                s_thresholds = split_thresholds[s][curr_nodes]
                
                vals = torch.gather(X_sub, 1, s_indices.unsqueeze(1)).squeeze(1)
                decision = (vals >= s_thresholds).long()
                curr_nodes = 2 * curr_nodes + 1 + decision
                
            leaf_indices = curr_nodes - self.n_split_nodes
            
            # Add prototype values
            Y_hat += lookup_table[s][leaf_indices]
            
        Y_true = X_calib @ W
        mse = torch.mean((Y_true - Y_hat) ** 2).item()
        
        return split_indices, split_thresholds, lookup_table, prototypes, mse

    def _build_tree(self, X):
        # Greedy tree construction
        # Return split_indices (15), split_thresholds (15)
        
        num_nodes = self.n_split_nodes
        split_indices = torch.zeros(num_nodes, dtype=torch.long, device=X.device)
        split_thresholds = torch.zeros(num_nodes, device=X.device)
        
        # Queue for BFS or just iterative level-by-level
        # We need to track which samples belong to which node
        # Initial: all samples in root (node 0)
        
        buckets = {0: torch.arange(X.shape[0], device=X.device)} # node_idx -> sample_indices
        
        # Levels: 0 (root), 1 (2 nodes), 2 (4 nodes), 3 (8 nodes) -> 15 nodes total
        # Node indices:
        # L0: 0
        # L1: 1, 2
        # L2: 3, 4, 5, 6
        # L3: 7..14
        
        for node_idx in range(self.n_split_nodes):
            if node_idx not in buckets or len(buckets[node_idx]) == 0:
                # Empty bucket, just pick dummy split
                split_indices[node_idx] = 0
                split_thresholds[node_idx] = 0
                continue
                
            sample_idxs = buckets[node_idx]
            X_node = X[sample_idxs]
            
            # Find best split
            best_dim, best_thresh = self._find_best_split(X_node)
            
            split_indices[node_idx] = best_dim
            split_thresholds[node_idx] = best_thresh
            
            # Split samples
            vals = X_node[:, best_dim]
            left_mask = vals < best_thresh
            right_mask = ~left_mask
            
            left_idxs = sample_idxs[left_mask]
            right_idxs = sample_idxs[right_mask]
            
            # Children indices
            left_child = 2 * node_idx + 1
            right_child = 2 * node_idx + 2
            
            if left_child < self.n_split_nodes: # Only add if children are internal nodes or we need to track for leaves?
                # We need to track for next level iteration
                buckets[left_child] = left_idxs
                buckets[right_child] = right_idxs
                
        return split_indices, split_thresholds, buckets

    def _find_best_split(self, X, top_n = 4):
        # X: [N, dim]
        # Heuristic: Just check a few quantiles of each dimension?
        # Or check mean/median?
        # The paper says: "Evaluate candidate feature indices... find optimal split thresholds... minimize SSE"
        
        # 1. find top n variance dimensions
        variances = torch.var(X, dim=0)
        top_n_dims = torch.topk(variances, min(top_n, X.shape[1])).indices
        
        # 2. find best threshold for each dimension
        best_dim = None
        best_thresh = None
        best_sse = float('inf')
        
        for dim in top_n_dims:
            vals = X[:, dim]
            
            # Optimization: Instead of checking all unique values, check quantiles
            # Checking all unique values is O(N^2) effectively if we loop.
            # We can use a vectorized approach or just check percentiles.
            # For speed, let's check 10 quantiles.
            
            quantiles = torch.quantile(vals, torch.linspace(0.1, 0.9, 9, device=X.device))
            thresholds = torch.unique(quantiles)
            
            # Vectorized SSE calculation for all thresholds
            # This is still a bit tricky to fully vectorize without expanding memory too much.
            # Let's loop over these few thresholds.
            
            for thresh in thresholds:
                left_mask = vals < thresh
                right_mask = ~left_mask
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                # SSE = var * N (roughly, actually sum((x-mean)^2))
                # We can compute this efficiently.
                
                vals_left = vals[left_mask]
                vals_right = vals[right_mask]
                
                left_sse = torch.sum((vals_left - torch.mean(vals_left))**2)
                right_sse = torch.sum((vals_right - torch.mean(vals_right))**2)
                
                sse = left_sse + right_sse
                
                if sse < best_sse:
                    best_sse = sse
                    best_dim = dim
                    best_thresh = thresh
                    
        if best_dim is None:
            # Fallback if no split found (e.g. all values same)
            best_dim = top_n_dims[0]
            best_thresh = torch.mean(X[:, best_dim])
        
        return best_dim, best_thresh

    def _get_bucket_assignments(self, X, indices, thresholds):
        # Run the tree
        curr_nodes = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        
        for d in range(self.tree_depth):
            s_indices = indices[curr_nodes]
            s_thresholds = thresholds[curr_nodes]
            
            vals = torch.gather(X, 1, s_indices.unsqueeze(1)).squeeze(1)
            decision = (vals >= s_thresholds).long()
            
            curr_nodes = 2 * curr_nodes + 1 + decision
            
        return curr_nodes - self.n_split_nodes
