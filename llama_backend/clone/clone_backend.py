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
    """
    Performs the attention mechanism, exactly matching the Hugging Face LlamaAttention implementation.

    This function computes the scaled dot-product attention, including query, key, and value
    projections, rotary position embeddings (RoPE), and grouped-query attention (GQA).

    Args:
        hidden_states (torch.Tensor): The input hidden states, with shape
            (batch_size, seq_len, hidden_size).
        wq (torch.Tensor): The weight tensor for the query projection.
        wk (torch.Tensor): The weight tensor for the key projection.
        wv (torch.Tensor): The weight tensor for the value projection.
        wo (torch.Tensor): The weight tensor for the output projection.
        num_heads (int): The number of attention heads.
        num_kv_heads (int): The number of key/value heads for GQA.
        head_dim (int): The dimension of each attention head.
        rope_emb (HFRotaryEmbedding): The rotary position embedding layer.
        position_ids (torch.Tensor, optional): The position IDs for RoPE. Defaults to None.
        attention_mask (torch.Tensor, optional): The attention mask. Defaults to None.

    Returns:
        torch.Tensor: The output of the attention mechanism, with shape
            (batch_size, seq_len, hidden_size).
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
    """
    Performs the MLP (SwiGLU) forward pass, exactly matching the Hugging Face LlamaMLP implementation.

    This function computes the SwiGLU activation, which involves a gated linear unit
    with the SiLU (Swish) activation function.

    Args:
        hidden_states (torch.Tensor): The input hidden states, with shape
            (batch_size, seq_len, hidden_size).
        w_gate (torch.Tensor): The weight tensor for the gate projection.
        w_up (torch.Tensor): The weight tensor for the up projection.
        w_down (torch.Tensor): The weight tensor for the down projection.

    Returns:
        torch.Tensor: The output of the MLP, with shape (batch_size, seq_len, hidden_size).
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
    """
    Performs a single decoder layer forward pass, exactly matching the Hugging Face LlamaDecoderLayer.

    This function combines the self-attention block and the MLP block, with RMS normalization
    and residual connections, to form a complete Llama decoder layer.

    Args:
        hidden_states (torch.Tensor): The input hidden states.
        norm1_weight (torch.Tensor): The weight for the first RMS normalization.
        wq (torch.Tensor): The weight for the query projection in attention.
        wk (torch.Tensor): The weight for the key projection in attention.
        wv (torch.Tensor): The weight for the value projection in attention.
        wo (torch.Tensor): The weight for the output projection in attention.
        norm2_weight (torch.Tensor): The weight for the second RMS normalization.
        w_gate (torch.Tensor): The weight for the gate projection in the MLP.
        w_up (torch.Tensor): The weight for the up projection in the MLP.
        w_down (torch.Tensor): The weight for the down projection in the MLP.
        num_heads (int): The number of attention heads.
        num_kv_heads (int): The number of key/value heads for GQA.
        head_dim (int): The dimension of each attention head.
        rope_emb (HFRotaryEmbedding): The rotary position embedding layer.
        rms_eps (float, optional): The epsilon value for RMS normalization. Defaults to 1e-6.
        attention_mask (torch.Tensor, optional): The attention mask. Defaults to None.

    Returns:
        torch.Tensor: The output hidden states of the decoder layer.
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
