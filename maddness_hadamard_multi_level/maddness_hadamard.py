import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from maddness_hadamard_multi_level.utils import get_hadamard_matrix, apply_block_hadamard
from maddness_hadamard_multi_level.maddness import MaddnessQuantizerMultiLevel
import logging

class MaddnessHadamardLayer(nn.Module):
    def __init__(self, original_layer, subspace_dim=512, tree_depth=4, block_size=512, num_levels=1):
        super().__init__()
        self.block_size = block_size
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.subspace_dim = subspace_dim
        self.tree_depth = tree_depth
        self.num_levels = num_levels
        self.n_split_nodes = 2 ** tree_depth - 1
        self.num_prototypes = self.n_split_nodes + 1
        self.num_subspaces = self.in_features // self.subspace_dim
        
        # Hadamard Matrices (Block Diagonal)
        # We store the small block matrix as buffer
        if self.in_features % self.block_size == 0:
            self.register_buffer('H_in', get_hadamard_matrix(self.block_size))
        else:
            self.register_buffer('H_in', None)
            logging.warning(f"Input features {self.in_features} not divisible by block size {self.block_size}")
            
        if self.out_features % self.block_size == 0:
            self.register_buffer('H_out', get_hadamard_matrix(self.block_size))
        else:
            self.register_buffer('H_out', None)
            logging.warning(f"Output features {self.out_features} not divisible by block size {self.block_size}")

        # Maddness Parameters
        # Added num_levels dimension
        self.register_buffer('split_indices', torch.zeros(self.num_levels, self.num_subspaces, self.n_split_nodes, dtype=torch.long))
        self.register_buffer('split_thresholds', torch.zeros(self.num_levels, self.num_subspaces, self.n_split_nodes))
        self.register_buffer('lookup_table', torch.zeros(self.num_levels, self.num_subspaces, self.num_prototypes, self.out_features))
        
        # Input Prototypes for Residual Calculation (only needed if num_levels > 1)
        if self.num_levels > 1:
             self.register_buffer('input_prototypes', torch.zeros(self.num_levels, self.num_subspaces, self.num_prototypes, self.subspace_dim))
        
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
        
        current_residual = x.clone()
        
        for l in range(self.num_levels):
            level_approximation = torch.zeros_like(x)
            
            for s in range(self.num_subspaces):
                start_dim = s * self.subspace_dim
                end_dim = (s + 1) * self.subspace_dim
                x_sub = current_residual[:, start_dim:end_dim]
                
                curr_nodes = torch.zeros(batch_size, dtype=torch.long, device=x.device)
                
                for d in range(self.tree_depth):
                    s_indices = self.split_indices[l][s][curr_nodes]
                    s_thresholds = self.split_thresholds[l][s][curr_nodes]
                    
                    vals = torch.gather(x_sub, 1, s_indices.unsqueeze(1)).squeeze(1)
                    decision = (vals >= s_thresholds).long()
                    curr_nodes = 2 * curr_nodes + 1 + decision
                    
                leaf_indices = curr_nodes - self.n_split_nodes
                
                # 1. Add to output
                val = self.lookup_table[l][s][leaf_indices]
                output += val
                
                # 2. Reconstruct input for residual calculation (needed for next level)
                if l < self.num_levels - 1:
                    protos = self.input_prototypes[l][s]
                    level_approximation[:, start_dim:end_dim] = protos[leaf_indices]
            
            if l < self.num_levels - 1:
                current_residual = current_residual - level_approximation
            
        # Apply Hadamard to Output (Inverse)
        # Y = Y' @ H_out.T
        if self.H_out is not None:
            output = apply_block_hadamard(output, self.H_out)
            
        if self.bias is not None:
            output += self.bias
            
        return output.view(*original_shape[:-1], self.out_features)

    def train_and_configure(self, X_calib, W):
        """
        Trains the Maddness quantizer using calibration data and configures the layer's parameters.
        
        Args:
            X_calib (torch.Tensor): Calibration data [N, in_features].
            W (torch.Tensor): Original layer weights [out_features, in_features].
            
        Returns:
            mse (float): Mean Squared Error of the approximation on calibration data.
            max_diff (float): Maximum absolute difference.
        """
        # X_calib: [N, in]
        # W: [out, in] (standard linear weight)
        
        # 1. Transform Inputs and Weights
        if self.H_in is not None:
            X_calib = apply_block_hadamard(X_calib, self.H_in.to(dtype=X_calib.dtype))
            W = apply_block_hadamard(W, self.H_in.to(dtype=W.dtype))
            
        if self.H_out is not None:
            # W' = H_out @ W
            W = apply_block_hadamard(W.t(), self.H_out.to(dtype=W.dtype)).t()
            
        # Now proceed with standard Maddness training on transformed data
        W_t = W.t() # [in, out]
        
        print("Training MaddnessHadamard...")
        
        # Use MaddnessQuantizerMultiLevel for training
        quantizer = MaddnessQuantizerMultiLevel(
            subspace_dim=self.subspace_dim,
            num_levels=self.num_levels,
            tree_depth=self.tree_depth,
            hidden_dim=self.in_features
        )
        quantizer.fit(X_calib)
        
        # Extract parameters from the trained quantizer for ALL levels
        for l in range(self.num_levels):
            split_indices = torch.stack(quantizer.levels_split_dims[l])
            split_thresholds = torch.stack(quantizer.levels_split_threshs[l])
            prototypes = torch.stack(quantizer.levels_prototypes[l])
            
            # Store input prototypes if needed
            if self.num_levels > 1:
                self.input_prototypes[l].copy_(prototypes)
            
            lookup_table = []
            for s in range(self.num_subspaces):
                start_dim = s * self.subspace_dim
                end_dim = (s + 1) * self.subspace_dim
                W_sub = W_t[start_dim:end_dim, :]
                P = prototypes[s]
                T = P @ W_sub
                lookup_table.append(T)
                
            lookup_table = torch.stack(lookup_table)
            
            # Update buffers
            self.split_indices[l].copy_(split_indices)
            self.split_thresholds[l].copy_(split_thresholds)
            self.lookup_table[l].copy_(lookup_table)
        
        # MSE Calculation (Approximation)
        Y_hat = torch.zeros(X_calib.shape[0], W.shape[0], device=X_calib.device, dtype=W.dtype)
        current_residual = X_calib.clone()
        
        for l in range(self.num_levels):
            level_approx_input = torch.zeros_like(X_calib)
            for s in range(self.num_subspaces):
                start_dim = s * self.subspace_dim
                end_dim = (s + 1) * self.subspace_dim
                X_sub = current_residual[:, start_dim:end_dim]
                
                curr_nodes = torch.zeros(X_calib.shape[0], dtype=torch.long, device=X_calib.device)
                for d in range(self.tree_depth):
                    s_indices = self.split_indices[l][s][curr_nodes]
                    s_thresholds = self.split_thresholds[l][s][curr_nodes]
                    vals = torch.gather(X_sub, 1, s_indices.unsqueeze(1)).squeeze(1)
                    decision = (vals >= s_thresholds).long()
                    curr_nodes = 2 * curr_nodes + 1 + decision
                leaf_indices = curr_nodes - self.n_split_nodes
                
                Y_hat += self.lookup_table[l][s][leaf_indices]
                
                if self.num_levels > 1:
                     level_approx_input[:, start_dim:end_dim] = self.input_prototypes[l][s][leaf_indices]
            
            if self.num_levels > 1:
                current_residual = current_residual - level_approx_input
            
        Y_true = X_calib @ W.t()
        diff = (Y_true - Y_hat).abs()
        mse = torch.mean(diff ** 2).item()
        max_diff = diff.max().item()
        
        return mse, max_diff
