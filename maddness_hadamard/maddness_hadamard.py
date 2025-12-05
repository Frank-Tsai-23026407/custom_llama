import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def _get_hadamard_unnormalized(n, device='cpu'):
    if n == 1:
        return torch.ones(1, 1, device=device)
    h_n_2 = _get_hadamard_unnormalized(n // 2, device=device)
    top = torch.cat([h_n_2, h_n_2], dim=1)
    bottom = torch.cat([h_n_2, -h_n_2], dim=1)
    return torch.cat([top, bottom], dim=0)

def get_hadamard_matrix(n, device='cpu'):
    H = _get_hadamard_unnormalized(n, device)
    return H / torch.sqrt(torch.tensor(n, dtype=torch.float32, device=device))

def apply_block_hadamard(x, H_block):
    """
    Applies block diagonal Hadamard transform.
    x: [..., dim]
    H_block: [B, B]
    """
    B = H_block.shape[0]
    dim = x.shape[-1]
    if dim % B != 0:
        return x # Should not happen if checked in init
    
    original_shape = x.shape
    num_blocks = dim // B
    x = x.view(*original_shape[:-1], num_blocks, B)
    x = x @ H_block.t()
    return x.view(*original_shape)

class MaddnessHadamardLayer(nn.Module):
    def __init__(self, original_layer, subspace_dim=512, tree_depth=4, block_size=512):
        super().__init__()
        self.block_size = block_size
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.subspace_dim = subspace_dim
        self.tree_depth = tree_depth
        self.n_split_nodes = 2 ** tree_depth - 1
        self.num_prototypes = self.n_split_nodes + 1
        self.num_subspaces = self.in_features // self.subspace_dim
        
        # Hadamard Matrices (Block Diagonal)
        # We store the small block matrix as buffer
        if self.in_features % self.block_size == 0:
            self.register_buffer('H_in', get_hadamard_matrix(self.block_size))
        else:
            self.register_buffer('H_in', None)
            
        if self.out_features % self.block_size == 0:
            self.register_buffer('H_out', get_hadamard_matrix(self.block_size))
        else:
            self.register_buffer('H_out', None)

        # Maddness Parameters
        self.register_buffer('split_indices', torch.zeros(self.num_subspaces, self.n_split_nodes, dtype=torch.long))
        self.register_buffer('split_thresholds', torch.zeros(self.num_subspaces, self.n_split_nodes))
        self.register_buffer('lookup_table', torch.zeros(self.num_subspaces, self.num_prototypes, self.out_features))
        
        if original_layer.bias is not None:
            self.register_buffer('bias', original_layer.bias.clone())
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        
        # Apply Hadamard to Input
        if self.H_in is not None:
            # x is [N, in]. H_in is [B, B].
            x = apply_block_hadamard(x, self.H_in)
            
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        
        for s in range(self.num_subspaces):
            start_dim = s * self.subspace_dim
            end_dim = (s + 1) * self.subspace_dim
            x_sub = x[:, start_dim:end_dim]
            
            curr_nodes = torch.zeros(batch_size, dtype=torch.long, device=x.device)
            
            for d in range(self.tree_depth):
                s_indices = self.split_indices[s][curr_nodes]
                s_thresholds = self.split_thresholds[s][curr_nodes]
                
                vals = torch.gather(x_sub, 1, s_indices.unsqueeze(1)).squeeze(1)
                decision = (vals >= s_thresholds).long()
                curr_nodes = 2 * curr_nodes + 1 + decision
                
            leaf_indices = curr_nodes - self.n_split_nodes
            val = self.lookup_table[s][leaf_indices]
            output += val
            
        # Apply Hadamard to Output (Inverse)
        # Y = Y' @ H_out.T
        if self.H_out is not None:
            output = apply_block_hadamard(output, self.H_out)
            
        if self.bias is not None:
            output += self.bias
            
        return output.view(*original_shape[:-1], self.out_features)

class MaddnessHadamardTrainer:
    def __init__(self, in_features, subspace_dim=512, tree_depth=4, block_size=512):
        self.block_size = block_size
        self.tree_depth = tree_depth
        self.n_split_nodes = 2 ** tree_depth - 1
        self.in_features = in_features
        self.subspace_dim = subspace_dim
        self.num_prototypes = self.n_split_nodes + 1
        self.num_subspaces = self.in_features // self.subspace_dim
        
    def train(self, X_calib, W):
        # X_calib: [N, in]
        # W: [out, in] (standard linear weight)
        
        # 1. Transform Inputs and Weights
        if self.in_features % self.block_size == 0:
            H_in = get_hadamard_matrix(self.block_size, device=X_calib.device)
            X_calib = apply_block_hadamard(X_calib, H_in)
            # W is [out, in]. 
            # We want to transform the input dimension of W.
            # W_new = W @ block_diag(H.T)
            # This is equivalent to applying block hadamard to the last dim of W
            W = apply_block_hadamard(W, H_in)
            
        out_features = W.shape[0]
        if out_features % self.block_size == 0:
            H_out = get_hadamard_matrix(self.block_size, device=W.device)
            # W' = H_out @ W
            # Here H_out is [B, B]. W is [out, in].
            # We want to apply H to the first dimension of W.
            # W' = block_diag(H) @ W
            # W.T is [in, out]. apply_block_hadamard(W.T, H) -> W.T @ H.T
            # (W.T @ H.T).T = H @ W. Correct.
            W = apply_block_hadamard(W.t(), H_out).t()
            
        # Now proceed with standard Maddness training on transformed data
        W_t = W.t() # [in, out]
        
        split_indices_all = []
        split_thresholds_all = []
        prototypes_all = []
        
        print("Training MaddnessHadamard...")
        for s in range(self.num_subspaces):
            start_dim = s * self.subspace_dim
            end_dim = (s + 1) * self.subspace_dim
            X_sub = X_calib[:, start_dim:end_dim]
            
            indices, thresholds, buckets = self._build_tree(X_sub)
            split_indices_all.append(indices)
            split_thresholds_all.append(thresholds)
            
            # Optimize Prototypes
            N = X_sub.shape[0]
            G = torch.zeros(N, self.num_prototypes, device=X_calib.device)
            bucket_assignments = self._get_bucket_assignments(X_sub, indices, thresholds)
            G.scatter_(1, bucket_assignments.unsqueeze(1), 1.0)
            
            lamb = 1e-4
            I = torch.eye(self.num_prototypes, device=X_calib.device)
            GtG = G.t() @ G
            GtX = G.t() @ X_sub
            P = torch.linalg.solve(GtG + lamb * I, GtX)
            prototypes_all.append(P)
            
        split_indices = torch.stack(split_indices_all)
        split_thresholds = torch.stack(split_thresholds_all)
        prototypes = torch.stack(prototypes_all)
        
        lookup_table = []
        for s in range(self.num_subspaces):
            start_dim = s * self.subspace_dim
            end_dim = (s + 1) * self.subspace_dim
            W_sub = W_t[start_dim:end_dim, :]
            P = prototypes[s]
            T = P @ W_sub
            lookup_table.append(T)
            
        lookup_table = torch.stack(lookup_table)
        
        # MSE Calculation (Approximation)
        # We need to reconstruct Y_hat and compare with Y_true (transformed)
        # Actually, we should compare in original domain?
        # MSE is preserved under orthogonal transform (Parseval's theorem).
        # So calculating MSE in transformed domain is fine.
        
        Y_hat = torch.zeros(X_calib.shape[0], W.shape[0], device=X_calib.device, dtype=W.dtype)
        for s in range(self.num_subspaces):
            start_dim = s * self.subspace_dim
            end_dim = (s + 1) * self.subspace_dim
            X_sub = X_calib[:, start_dim:end_dim]
            
            curr_nodes = torch.zeros(X_calib.shape[0], dtype=torch.long, device=X_calib.device)
            for d in range(self.tree_depth):
                s_indices = split_indices[s][curr_nodes]
                s_thresholds = split_thresholds[s][curr_nodes]
                vals = torch.gather(X_sub, 1, s_indices.unsqueeze(1)).squeeze(1)
                decision = (vals >= s_thresholds).long()
                curr_nodes = 2 * curr_nodes + 1 + decision
            leaf_indices = curr_nodes - self.n_split_nodes
            Y_hat += lookup_table[s][leaf_indices]
            
        Y_true = X_calib @ W.t()
        diff = (Y_true - Y_hat).abs()
        mse = torch.mean(diff ** 2).item()
        max_diff = diff.max().item()
        
        return split_indices, split_thresholds, lookup_table, prototypes, mse, max_diff

    def _build_tree(self, X):
        # Same as original Maddness
        num_nodes = self.n_split_nodes
        split_indices = torch.zeros(num_nodes, dtype=torch.long, device=X.device)
        split_thresholds = torch.zeros(num_nodes, device=X.device)
        buckets = {0: torch.arange(X.shape[0], device=X.device)}
        
        for node_idx in range(self.n_split_nodes):
            if node_idx not in buckets or len(buckets[node_idx]) == 0:
                continue
            sample_idxs = buckets[node_idx]
            X_node = X[sample_idxs]
            best_dim, best_thresh = self._find_best_split(X_node)
            split_indices[node_idx] = best_dim
            split_thresholds[node_idx] = best_thresh
            
            vals = X_node[:, best_dim]
            left_mask = vals < best_thresh
            right_mask = ~left_mask
            
            left_child = 2 * node_idx + 1
            right_child = 2 * node_idx + 2
            
            if left_child < self.n_split_nodes:
                buckets[left_child] = sample_idxs[left_mask]
                buckets[right_child] = sample_idxs[right_mask]
        return split_indices, split_thresholds, buckets

    def _find_best_split(self, X, top_n=4):
        variances = torch.var(X, dim=0)
        top_n_dims = torch.topk(variances, min(top_n, X.shape[1])).indices
        best_dim, best_thresh, best_sse = None, None, float('inf')
        
        for dim in top_n_dims:
            vals = X[:, dim]
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

    def _get_bucket_assignments(self, X, indices, thresholds):
        curr_nodes = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        for d in range(self.tree_depth):
            s_indices = indices[curr_nodes]
            s_thresholds = thresholds[curr_nodes]
            vals = torch.gather(X, 1, s_indices.unsqueeze(1)).squeeze(1)
            decision = (vals >= s_thresholds).long()
            curr_nodes = 2 * curr_nodes + 1 + decision
        return curr_nodes - self.n_split_nodes
