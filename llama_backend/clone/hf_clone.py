import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import torch
from typing import Tuple, Optional
from .llama_backend.custom.plain_script.plain_script import rmsnorm, input_embedding, lm_head


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=torch.float32):
    positions = torch.arange(0, seq_len, dtype=dtype, device=device)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    freqs = torch.outer(positions, inv_freq)  # (S, D/2)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,S,D/2)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,S,D/2)
    return cos, sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE using HF's rotate_half method (split at middle, not interleaved).
    
    HF implementation: rotate_half splits tensor into [first_half, second_half]
    and returns cat([-second_half, first_half]).
    This is different from interleaved even/odd indexing.
    
    Args:
        q, k: (B, H, S, D) query and key tensors
        cos, sin: (1, 1, S, D/2) or (1, 1, S, D) cosine and sine tables
        
    Returns:
        q_embed, k_embed: rotated query and key
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
    """Replicate a single Llama decoder layer (attention + MLP) with residuals.
    params expects keys: norm1_weight, norm2_weight, wq,wk,wv,wo,w_gate,w_up,w_down
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
    """Functional forward of all layers using a shared RoPE cache.

    Added `rope_cache_dtype` so users can experiment with lower precision
    sin/cos tables (e.g. float16/bfloat16). This may change numerical parity
    very slightly; keep bfloat16 for bit-exact comparisons.
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
