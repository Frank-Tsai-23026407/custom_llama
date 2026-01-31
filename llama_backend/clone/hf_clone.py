import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import torch
from typing import Tuple, Optional
from ..custom.llama_backend import rmsnorm, input_embedding, lm_head


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=torch.float32):
    """
    Builds the RoPE (Rotary Position Embedding) cache of sine and cosine values.

    This function pre-computes the sine and cosine values for RoPE, which can be
    cached and reused for efficiency.

    Args:
        seq_len (int): The sequence length.
        head_dim (int): The dimension of each attention head.
        base (float, optional): The base value for the inverse frequency calculation.
            Defaults to 10000.0.
        device (torch.device, optional): The device to create the cache on. Defaults to None.
        dtype (torch.dtype, optional): The data type for the cache. Defaults to torch.float32.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the cosine and sine caches.
    """
    positions = torch.arange(0, seq_len, dtype=dtype, device=device)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    freqs = torch.outer(positions, inv_freq)  # (S, D/2)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,S,D/2)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,S,D/2)
    return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding (RoPE) to the query and key tensors.

    This function uses the 'rotate_half' method to apply RoPE, which is consistent
    with the Hugging Face implementation. It splits the tensors at the middle and
    swaps the halves with negation.

    Args:
        q (torch.Tensor): The query tensor, with shape (B, H, S, D).
        k (torch.Tensor): The key tensor, with shape (B, H, S, D).
        cos (torch.Tensor): The cosine cache for RoPE.
        sin (torch.Tensor): The sine cache for RoPE.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the rotated query and key tensors.
    """
    # Unsqueeze cos/sin if they're (1,1,S,D/2) to match (B,H,S,D)
    # HF's cos/sin are actually (seq_len, head_dim) but get unsqueezed to (1, 1, seq_len, head_dim)
    if cos.shape[-1] != q.shape[-1]:
        # cos/sin are half-sized, need to duplicate like HF does with torch.cat((freqs, freqs), dim=-1)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    
    # HF's rotate_half: split at middle and swap with negation
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Apply rotation: q*cos + rotate_half(q)*sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def clone_attention_ffn(
    x: torch.Tensor,
    params: dict,
    num_heads: int,
    num_kv_heads: int,
    rms_eps: float,
    rope_cache: Tuple[torch.Tensor, torch.Tensor],
    return_attn: bool = False,
    compute_dtype: torch.dtype = torch.bfloat16,
    softmax_fp32: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Replicates a single Llama decoder layer, including attention and MLP with residuals.

    This function performs a full forward pass of a single Llama decoder layer,
    including RMS normalization, attention, and the feed-forward network (FFN). It
    supports configurable computation and softmax data types for precision control.

    Args:
        x (torch.Tensor): The input tensor.
        params (dict): A dictionary of layer parameters, including weights for normalization,
            attention, and MLP.
        num_heads (int): The number of attention heads.
        num_kv_heads (int): The number of key/value heads for GQA.
        rms_eps (float): The epsilon value for RMS normalization.
        rope_cache (tuple[torch.Tensor, torch.Tensor]): The RoPE cache.
        return_attn (bool, optional): Whether to return the attention weights. Defaults to False.
        compute_dtype (torch.dtype, optional): The data type for computation.
            Defaults to torch.bfloat16.
        softmax_fp32 (bool, optional): Whether to use float32 for softmax. Defaults to True.

    Returns:
        tuple[torch.Tensor, torch.Tensor or None]: A tuple containing the output tensor and,
            optionally, the attention weights.
    """
    B, S, D = x.shape
    # Cast working tensor to compute dtype (default bf16) while keeping weights already provided in that dtype.
    # This reduces memory bandwidth vs full fp32; attention softmax still uses fp32 for stability.
    x = x.to(compute_dtype)
    head_dim = D // num_heads
    # LayerNorm (RMSNorm) 1
    x_norm = rmsnorm(x, params['norm1_weight'].to(compute_dtype), eps=rms_eps)
    # Projections - ensure weights match compute_dtype for matmul compatibility
    q_proj = torch.matmul(x_norm, params['wq'].to(compute_dtype).t())
    k_proj = torch.matmul(x_norm, params['wk'].to(compute_dtype).t())
    v_proj = torch.matmul(x_norm, params['wv'].to(compute_dtype).t())
    # Reshape
    q = q_proj.view(B, S, num_heads, head_dim).transpose(1, 2)  # (B,H,S,Dh)
    kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
    k = k_proj.view(B, S, kv_heads, head_dim).transpose(1, 2)
    v = v_proj.view(B, S, kv_heads, head_dim).transpose(1, 2)
    # Rotary
    cos, sin = rope_cache
    q, k = apply_rotary_pos_emb(q, k, cos[..., :S, :], sin[..., :S, :])
    # Repeat k,v for grouped query attention
    if num_heads != kv_heads:
        repeat_factor = num_heads // kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
    # Attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    # Causal mask (lower triangular keep, upper masked to -inf)
    causal_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
    neg_inf = torch.tensor(float('-inf'), device=x.device, dtype=attn_scores.dtype)
    attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), neg_inf)
    # Softmax precision control: upcast to float32 for stability if requested
    if softmax_fp32:
        attn_weights = torch.softmax(attn_scores.to(torch.float32), dim=-1).to(q.dtype)
    else:
        attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_out = torch.matmul(attn_weights.to(v.dtype), v)  # (B,H,S,Dh)
    attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
    attn_out = torch.matmul(attn_out, params['wo'].to(compute_dtype).t())
    x = x + attn_out
    # LayerNorm 2
    x_norm2 = rmsnorm(x, params['norm2_weight'].to(compute_dtype), eps=rms_eps)
    # MLP (SwiGLU style)
    gate = torch.matmul(x_norm2, params['w_gate'].to(compute_dtype).t())
    up = torch.matmul(x_norm2, params['w_up'].to(compute_dtype).t())
    gate = torch.nn.functional.silu(gate)
    hidden = gate * up
    ffn_out = torch.matmul(hidden, params['w_down'].to(compute_dtype).t())
    x = x + ffn_out
    return (x, attn_weights) if return_attn else (x, None)


def clone_forward_all(
    x: torch.Tensor,
    layers_params: list,
    num_heads: int,
    num_kv_heads: int,
    rms_eps: float,
    rope_cache_dtype: torch.dtype = torch.bfloat16,
    compute_dtype: torch.dtype = torch.bfloat16,
    softmax_fp32: bool = True,
) -> Tuple[torch.Tensor, list]:
    """
    Performs a functional forward pass of all decoder layers.

    This function iterates through a list of layer parameters, applying the
    `clone_attention_ffn` function for each layer to compute the full forward
    pass of the model. It uses a shared RoPE cache for efficiency.

    Args:
        x (torch.Tensor): The input tensor.
        layers_params (list): A list of dictionaries, where each dictionary
            contains the parameters for a single decoder layer.
        num_heads (int): The number of attention heads.
        num_kv_heads (int): The number of key/value heads for GQA.
        rms_eps (float): The epsilon value for RMS normalization.
        rope_cache_dtype (torch.dtype, optional): The data type for the RoPE
            cache. Defaults to torch.bfloat16.
        compute_dtype (torch.dtype, optional): The data type for computation.
            Defaults to torch.bfloat16.
        softmax_fp32 (bool, optional): Whether to use float32 for softmax.
            Defaults to True.

    Returns:
        tuple[torch.Tensor, list]: A tuple containing the final output tensor
            and a list of latent states from each layer.
    """
    latents = [x]
    S = x.shape[1]
    D = x.shape[2]
    head_dim = D // num_heads
    rope_cache = build_rope_cache(S, head_dim, device=x.device, dtype=rope_cache_dtype)
    for lp in layers_params:
        x, _ = clone_attention_ffn(
            x, lp, num_heads, num_kv_heads, rms_eps, rope_cache,
            return_attn=False, compute_dtype=compute_dtype, softmax_fp32=softmax_fp32
        )
        latents.append(x)
    return x, latents
