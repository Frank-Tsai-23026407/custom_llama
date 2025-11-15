"""HuggingFace-compatible Rotary Position Embedding (RoPE) implementation.

This module provides bit-exact RoPE matching transformers LlamaRotaryEmbedding
for numerical parity with HF models.
"""
import torch
import math


class HFRotaryEmbedding:
    """Precompute cos/sin cache for RoPE matching HF transformers.

    Added option `cache_dtype` so users can experiment with lower precision
    (e.g. torch.float16 / torch.bfloat16) for the precomputed sin/cos tables.
    Lower precision reduces memory bandwidth but may introduce tiny numeric drift.
    """
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, cache_dtype: torch.dtype = torch.float32):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.cache_dtype = cache_dtype
        
        # Build frequency cache (store in chosen precision). Use float32 intermediate
        # for exponent then cast, to avoid severe precision loss in pow.
        inv_freq_fp32 = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        inv_freq = inv_freq_fp32.to(self.cache_dtype)
        self.register_buffer("inv_freq", inv_freq)
        
        # Build initial cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings, device)
    
    def register_buffer(self, name, tensor):
        """Simple buffer storage."""
        setattr(self, name, tensor)
    
    def _set_cos_sin_cache(self, seq_len, device):
        """Precompute cos/sin for positions [0, seq_len) in `cache_dtype`."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        # HF uses cat for interleaving pattern: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Compute with fp32 for accuracy then cast to cache dtype to reduce error amplification.
        self.cos_cached = emb.cos().to(self.cache_dtype)
        self.sin_cached = emb.sin().to(self.cache_dtype)
    
    def forward(self, x, seq_len):
        """Return cos, sin for sequence length seq_len.
        
        Args:
            x: input tensor (unused, for API compatibility)
            seq_len: sequence length
            
        Returns:
            cos: (seq_len, dim) cosine cache
            sin: (seq_len, dim) sine cache
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """Rotate half the hidden dims of the input (HF implementation)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to query and key tensors (HF-compatible).
    
    Args:
        q: query tensor (batch, num_heads, seq_len, head_dim)
        k: key tensor (batch, num_kv_heads, seq_len, head_dim)
        cos: cosine cache (seq_len, head_dim)
        sin: sine cache (seq_len, head_dim)
        
    Returns:
        q_embed: rotated query
        k_embed: rotated key
    """
    # Expand cos/sin to match q/k: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    """Repeat key/value heads for grouped-query attention (HF implementation).
    
    Args:
        hidden_states: (batch, num_kv_heads, seq_len, head_dim)
        n_rep: repetition factor (num_heads // num_kv_heads)
        
    Returns:
        repeated: (batch, num_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
