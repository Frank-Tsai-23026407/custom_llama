import argparse
import copy
import os
import sys
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from custom_model_only_quantize.block_floating_point.block_quantization_2d import block_floating_point_quantize_2d


def pseudo_quantize(W, n_bits=4, block_height=1, block_width=64):
    """
    實作一個 pseudo_quantize 函式，支援 block-wise quantization。
    使用的是 block_quantization_2d.py 中的 2D BFP 實作。
    
    Args:
        W (torch.Tensor): 權重矩陣，形狀為 [out_features, in_features]
        n_bits (int): 量化位元數 (Mantissa bits)
        block_height (int): 分塊高度
        block_width (int): 分塊寬度
        
    Returns:
        torch.Tensor: 量化後並還原 (dequantize) 的權重矩陣
    """
    # 調用 2D BFP 量化
    return block_floating_point_quantize_2d(W, block_height=block_height, block_width=block_width, mantissa_bits=n_bits)

def search_awq_scale(W, X, block_height=1, block_width=64, n_grid=20):
    """
    核心的 search_awq_scale 函式：遍歷不同的 alpha 值尋找最小化 MSE 的縮放向量 s。
    
    Args:
        W (torch.Tensor): 權重矩陣 [out_features, in_features]
        X (torch.Tensor): 輸入活化值 [seq_len, in_features]
        n_grid (int): 搜尋 alpha 的格點數 (例如 np.linspace(0, 1, 20))
        
    Returns:
        torch.Tensor: 最優縮放向量 s [in_features]
    """
    # 計算活化值在每個輸入通道 (input channel) 的平均量級
    # s_X 的維度為 [in_features]
    s_X = X.abs().mean(dim=0)
    
    best_error = float('inf')
    best_s = torch.ones_like(s_X)
    
    # 預先計算原始輸出 WX 以便後續比較誤差
    # X: [L, Ci], W: [Co, Ci], org_out: [L, Co]
    org_out = torch.matmul(X, W.t())
    
    # 搜尋 alpha 參數 (s = s_X ^ alpha)
    for alpha in np.linspace(0, 1, n_grid):
        # 根據公式 s = s_X^alpha 限制搜尋空間
        s = s_X.pow(alpha).clamp(min=1e-5)
        
        # 1. 準備縮放權重：W_scaled = W * diag(s)
        # 註解：W 維度是 [out_features, in_features]，s 維度是 [in_features]。
        # 我們將 s 擴展為 [1, in_features]，利用 PyTorch 的 Broadcasting (廣播) 功能，
        # 使 W 的每一橫列 (row) 都與 s 做逐元素相乘。這等同於對 W 的每一行 (column) 進行縮放。
        w_scaled = W * s.view(1, -1)
        
        # 2. 對縮放後的權重進行假量化 (通常搜尋時使用 4-bit 作為參考)
        q_w_scaled = pseudo_quantize(w_scaled, n_bits=4, block_height=block_height, block_width=block_width)
        
        # 3. 計算縮放後的輸出 Q(W') * X' = Q(W*s) * (X/s)
        # 註解：X 維度是 [seq_len, in_features]，s 維度是 [in_features]。
        # 同樣利用 Broadcasting 將 X 的每一橫列都除以 s。
        x_scaled = X / s.view(1, -1)
        q_out = torch.matmul(x_scaled, q_w_scaled.t())
        
        # 計算輸出之間的均方誤差 (MSE)
        error = (q_out - org_out).pow(2).mean()
        
        if error < best_error:
            best_error = error
            best_s = s
            
    return best_s

def apply_awq_scale(W, s):
    """
    實作套用 scale 的功能：將 W 乘以 s。
    註解：s 為 [in_features]，W 為 [out_features, in_features]。
    使用 s.view(1, -1) 將其擴展至與 W 的最後一個維度對齊以進行廣播乘法。
    """
    return W * s.view(1, -1)

def apply_awq_inverse_scale(X, s):
    """
    實作套用 scale 的功能：將 X 除以 s。
    註解：X 為 [..., in_features]，s 為 [in_features]。
    直接使用 X / s 即可觸發廣播機制，對最後一維的所有特徵值進行除法。
    """
    return X / s.view(1, -1)

def awq_fix_precision_quantize_2d(weight: torch.Tensor, activation: torch.Tensor, block_height: int, block_width: int, mantissa_bits: int) -> torch.Tensor:
    """
    Performs fix-precision activation-aware weight quantization with 2D blocks.
    
    Fix-precision: Uses scaling factors based on activation magnitudes to reduce quantization
    error for salient weights. All weights (including salient ones) are still represented in
    BFP format, but the scaling protects important weights during quantization.
    
    The process:
    1. Calculate per-input-feature importance from activations
    2. Scale weights UP by importance (W_scaled = W * s)
    3. Apply standard BFP quantization to scaled weights
    4. Scale quantized weights back DOWN (W_final = Q(W_scaled) / s)
    
    This effectively allocates more quantization precision to important weights.
    
    Args:
        weight (torch.Tensor): The weight tensor (2D).
        activation (torch.Tensor): The input activations (batch, seq_len, in_features).
        block_height (int): The height of each 2D block.
        block_width (int): The width of each 2D block.
        mantissa_bits (int): The number of mantissa bits for BFP quantization.
    
    Returns:
        torch.Tensor: The quantized and de-quantized weight tensor (all in BFP).
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {weight.dim()}D")
    
    epsilon = 1e-9
    
    # 1. Calculate per-input-feature importance from activations
    if activation.dim() == 3:
        activation = activation.reshape(-1, activation.shape[-1])
    
    # Calculate the mean absolute value for each input feature dimension
    importance_scores = torch.abs(activation).mean(dim=0)  # Shape: (in_features,)
    
    # Ensure importance scores are on the same device as weights
    importance_scores = importance_scores.to(weight.device)
    
    # 2. Scale weights UP by importance (W_scaled = W * s)
    # Weight matrix is (out_features, in_features)
    # Broadcast importance_scores across rows
    scaled_weight = weight * importance_scores.unsqueeze(0)
    
    # 3. Apply standard 2D BFP quantization to scaled weights
    quantized_scaled_weight = block_floating_point_quantize_2d(
        scaled_weight,
        block_height=block_height,
        block_width=block_width,
        mantissa_bits=mantissa_bits
    )
    
    # 4. Scale quantized weights back DOWN (W_final = Q(W_scaled) / s)
    reconstructed_weight = quantized_scaled_weight / (importance_scores.unsqueeze(0) + epsilon)
    
    return reconstructed_weight

def awq_mix_precision_quantize_2d(weight: torch.Tensor, activation: torch.Tensor, block_height: int, block_width: int, mantissa_bits: int, top_k: int = 16) -> torch.Tensor:
    """
    Performs mix-precision activation-aware weight quantization with 2D blocks.
    
    Mix-precision: Identifies the top_k most salient weights based on activation magnitudes
    and preserves them in full floating-point (FP32) precision. The remaining weights are
    quantized using standard BFP.
    
    Top-K Selection:
        - Per 2D block: Select top-k individual ENTRIES (elements) with highest salience
        - NOT per-row or per-column, but within the entire flattened block
        - These k entries can be scattered across different rows/columns in the block
        - Example: For a 16×16 block with top_k=16, we select the 16 most salient
          elements out of 256 total elements in that block
    
    Args:
        weight (torch.Tensor): The weight tensor (2D).
        activation (torch.Tensor): The input activations (batch, seq_len, in_features).
        block_height (int): The height of each 2D block.
        block_width (int): The width of each 2D block.
        mantissa_bits (int): The number of mantissa bits for BFP quantization.
        top_k (int): The number of salient weight ENTRIES to preserve in FP32 per block.
                     These entries can be in different rows/columns within the block.
                     Default: 16 (e.g., 16 out of 256 for a 16×16 block = 6.25%)
    
    Returns:
        torch.Tensor: The quantized weight tensor with mix-precision (FP32 + BFP).
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got {weight.dim()}D")
    
    epsilon = 1e-9
    
    # 1. Calculate per-input-feature importance from activations
    if activation.dim() == 3:
        # Reshape to (batch * seq_len, in_features)
        activation = activation.reshape(-1, activation.shape[-1])
    
    # Calculate the mean absolute value for each input feature dimension
    importance_scores = torch.abs(activation).mean(dim=0)  # Shape: (in_features,)
    
    # Ensure importance scores are on the same device as weights
    importance_scores = importance_scores.to(weight.device)
    
    # 2. Weight the weight matrix by importance
    # Weight matrix is (out_features, in_features)
    # Broadcast importance_scores across rows
    weighted_weight = weight * importance_scores.unsqueeze(0)
    
    # 3. Apply 2D block quantization
    original_shape = weight.shape
    out_features, in_features = original_shape
    
    # Padding
    pad_height = (block_height - (out_features % block_height)) % block_height
    pad_width = (block_width - (in_features % block_width)) % block_width
    
    if pad_height > 0 or pad_width > 0:
        weighted_weight = torch.nn.functional.pad(weighted_weight, (0, pad_width, 0, pad_height))
        weight_padded = torch.nn.functional.pad(weight, (0, pad_width, 0, pad_height))
    else:
        weight_padded = weight
    
    padded_height, padded_width = weighted_weight.shape
    num_blocks_height = padded_height // block_height
    num_blocks_width = padded_width // block_width
    
    # Reshape into blocks
    weighted_blocks = weighted_weight.reshape(num_blocks_height, block_height, num_blocks_width, block_width)
    weighted_blocks = weighted_blocks.permute(0, 2, 1, 3)
    total_blocks = num_blocks_height * num_blocks_width
    weighted_blocks = weighted_blocks.reshape(total_blocks, block_height * block_width)
    
    weight_blocks = weight_padded.reshape(num_blocks_height, block_height, num_blocks_width, block_width)
    weight_blocks = weight_blocks.permute(0, 2, 1, 3)
    weight_blocks = weight_blocks.reshape(total_blocks, block_height * block_width)
    
    # 4. For each block, identify top_k salient weights
    # Calculate salience: absolute value of weighted weights
    salience = torch.abs(weighted_blocks)  # Shape: (total_blocks, block_size)
    
    # Get top_k indices for each block
    # IMPORTANT: top_k is applied per block, selecting k individual ENTRIES
    # within the flattened block (not per row or per column)
    block_size = block_height * block_width
    k = min(top_k, block_size)  # Ensure k doesn't exceed block size
    
    # Get top-k values and indices (dim=1 means per-block)
    # topk_indices shape: (total_blocks, k)
    topk_salience, topk_indices = torch.topk(salience, k, dim=1)
    
    # 5. Apply standard BFP quantization to all weights
    # Determine the shared exponent (E) for each block
    max_abs_values = torch.amax(torch.abs(weight_blocks), dim=1, keepdim=True)
    E = torch.ceil(torch.log2(max_abs_values + epsilon))
    S = torch.pow(2.0, E)
    
    M = weight_blocks / S
    Q_max = (1 << (mantissa_bits - 1)) - 1
    M_quant = torch.round(M * Q_max)
    M_quant = torch.clamp(M_quant, -Q_max, Q_max)
    M_dequant = M_quant / Q_max
    reconstructed_blocks = M_dequant * S
    
    # 6. For salient weights, preserve with higher precision (skip quantization)
    # Create a mask for the top-k salient weights
    mask = torch.zeros_like(weight_blocks, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    
    # Replace quantized values with original values for salient weights
    reconstructed_blocks = torch.where(mask, weight_blocks, reconstructed_blocks)
    
    # 7. Reshape back to 2D array
    reconstructed_blocks = reconstructed_blocks.reshape(num_blocks_height, num_blocks_width, block_height, block_width)
    reconstructed_blocks = reconstructed_blocks.permute(0, 2, 1, 3)
    reconstructed_weight = reconstructed_blocks.reshape(padded_height, padded_width)
    
    # 8. Remove padding
    if pad_height > 0 or pad_width > 0:
        reconstructed_weight = reconstructed_weight[:out_features, :in_features]
    
    return reconstructed_weight
