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
