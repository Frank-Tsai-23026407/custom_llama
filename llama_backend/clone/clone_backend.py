"""Clone backend: HF-exact transformer layer implementation.

This module provides transformer layer operations that exactly replicate
HuggingFace LlamaDecoderLayer behavior for numerical parity verification.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import math
from llama_backend.clone.hf_rope import HFRotaryEmbedding, apply_rotary_pos_emb, repeat_kv


def clone_rmsnorm(hidden_states, weight, eps=1e-6):
    """RMSNorm matching HF LlamaRMSNorm exactly.
    
    Args:
        hidden_states: input tensor (batch, seq_len, hidden_size)
        weight: learnable scale (hidden_size,)
        eps: epsilon for numerical stability
        
    Returns:
        normalized: (batch, seq_len, hidden_size)
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def clone_attention(
    hidden_states,
    wq, wk, wv, wo,
    num_heads,
    num_kv_heads,
    head_dim,
    rope_emb,
    position_ids=None,
    attention_mask=None
):
    """Llama attention exactly matching HF LlamaAttention.
    
    Args:
        hidden_states: (batch, seq_len, hidden_size)
        wq, wk, wv, wo: weight tensors
        num_heads: number of query heads
        num_kv_heads: number of key/value heads (for GQA)
        head_dim: dimension per head
        rope_emb: HFRotaryEmbedding instance
        position_ids: optional position indices (defaults to [0..seq_len-1])
        attention_mask: optional mask (None for causal)
        
    Returns:
        attn_output: (batch, seq_len, hidden_size)
    """
    bsz, q_len, hidden_size = hidden_states.shape
    
    # Q, K, V projections
    query_states = hidden_states @ wq.t()
    key_states = hidden_states @ wk.t()
    value_states = hidden_states @ wv.t()
    
    # Reshape to separate heads: (batch, num_heads, seq_len, head_dim)
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # Apply RoPE
    cos, sin = rope_emb.forward(value_states, seq_len=q_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # Repeat K/V for GQA
    key_states = repeat_kv(key_states, num_heads // num_kv_heads)
    value_states = repeat_kv(value_states, num_heads // num_kv_heads)
    
    # Scaled dot-product attention
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    
    # Causal mask
    if attention_mask is None and q_len > 1:
        # Build causal mask: (1, 1, seq_len, seq_len)
        causal_mask = torch.triu(
            torch.full((q_len, q_len), float('-inf'), device=hidden_states.device, dtype=attn_weights.dtype),
            diagonal=1
        )
        attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Attention @ V
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Reshape back: (batch, seq_len, hidden_size)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    
    # Output projection
    attn_output = attn_output @ wo.t()
    
    return attn_output


def clone_mlp(hidden_states, w_gate, w_up, w_down):
    """Llama MLP (SwiGLU) matching HF LlamaMLP exactly.
    
    Args:
        hidden_states: (batch, seq_len, hidden_size)
        w_gate, w_up, w_down: weight tensors
        
    Returns:
        output: (batch, seq_len, hidden_size)
    """
    gate = hidden_states @ w_gate.t()
    up = hidden_states @ w_up.t()
    gate = F.silu(gate)
    intermediate = gate * up
    output = intermediate @ w_down.t()
    return output


def clone_decoder_layer(
    hidden_states,
    norm1_weight,
    wq, wk, wv, wo,
    norm2_weight,
    w_gate, w_up, w_down,
    num_heads,
    num_kv_heads,
    head_dim,
    rope_emb,
    rms_eps=1e-6,
    attention_mask=None
):
    """Single decoder layer matching HF LlamaDecoderLayer exactly.
    
    Args:
        hidden_states: (batch, seq_len, hidden_size)
        norm1_weight, norm2_weight: RMSNorm weights
        wq, wk, wv, wo: attention weights
        w_gate, w_up, w_down: MLP weights
        num_heads, num_kv_heads, head_dim: attention config
        rope_emb: RoPE embedding instance
        rms_eps: RMSNorm epsilon
        attention_mask: optional mask
        
    Returns:
        output: (batch, seq_len, hidden_size)
    """
    # Self-attention block
    residual = hidden_states
    hidden_states = clone_rmsnorm(hidden_states, norm1_weight, eps=rms_eps)
    hidden_states = clone_attention(
        hidden_states,
        wq, wk, wv, wo,
        num_heads, num_kv_heads, head_dim,
        rope_emb,
        attention_mask=attention_mask
    )
    hidden_states = residual + hidden_states
    
    # MLP block
    residual = hidden_states
    hidden_states = clone_rmsnorm(hidden_states, norm2_weight, eps=rms_eps)
    hidden_states = clone_mlp(hidden_states, w_gate, w_up, w_down)
    hidden_states = residual + hidden_states
    
    return hidden_states
