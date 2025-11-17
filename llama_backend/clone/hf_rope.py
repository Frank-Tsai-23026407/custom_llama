"""HuggingFace-compatible Rotary Position Embedding (RoPE) implementation.

This module provides bit-exact RoPE matching transformers LlamaRotaryEmbedding
for numerical parity with HF models.
"""
import torch
import math


class HFRotaryEmbedding:
    """
    Implements Rotary Position Embedding (RoPE) compatible with Hugging Face's transformers.

    This class pre-computes the sine and cosine values for RoPE, which can be cached
    and reused for efficiency. It includes an option to specify the data type of the
    cache to experiment with lower precision.

    Args:
        dim (int): The dimension of the embeddings.
        max_position_embeddings (int, optional): The maximum sequence length.
            Defaults to 2048.
        base (int, optional): The base value for the inverse frequency calculation.
            Defaults to 10000.
        device (torch.device, optional): The device to create the cache on.
            Defaults to None.
        cache_dtype (torch.dtype, optional): The data type for the cache.
            Defaults to torch.float32.
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
        """
        Registers a buffer to the object.

        This method provides a simple way to store a tensor as a buffer, making it
        part of the object's state.

        Args:
            name (str): The name of the buffer.
            tensor (torch.Tensor): The tensor to be registered.
        """
        setattr(self, name, tensor)
    
    def _set_cos_sin_cache(self, seq_len, device):
        """
        Pre-computes and caches the cosine and sine values for RoPE.

        This method calculates the cosine and sine values for a given sequence length
        and stores them in the cache.

        Args:
            seq_len (int): The sequence length.
            device (torch.device): The device to create the cache on.
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)
        # HF uses cat for interleaving pattern: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Compute with fp32 for accuracy then cast to cache dtype to reduce error amplification.
        self.cos_cached = emb.cos().to(self.cache_dtype)
        self.sin_cached = emb.sin().to(self.cache_dtype)
    
    def forward(self, x, seq_len):
        """
        Retrieves the cosine and sine caches for a given sequence length.

        If the requested sequence length is greater than the cached length, this method
        will first expand the cache.

        Args:
            x (torch.Tensor): The input tensor (unused, for API compatibility).
            seq_len (int): The sequence length.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the cosine and sine caches.
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    """
    Rotates half of the hidden dimensions of the input tensor.

    This function is a key component of RoPE, splitting the input tensor into two
    halves and concatenating them in reverse order with the second half negated.

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The tensor with half of its dimensions rotated.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Applies Rotary Position Embedding (RoPE) to the query and key tensors.

    This function applies the pre-computed cosine and sine values to the query and
    key tensors to incorporate positional information.

    Args:
        q (torch.Tensor): The query tensor.
        k (torch.Tensor): The key tensor.
        cos (torch.Tensor): The cosine cache.
        sin (torch.Tensor): The sine cache.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the rotated query and key tensors.
    """
    # Expand cos/sin to match q/k: (1, 1, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    """
    Repeats the key/value heads for grouped-query attention.

    This function expands the key/value heads to match the number of query heads,
    which is a key feature of grouped-query attention (GQA).

    Args:
        hidden_states (torch.Tensor): The hidden states of the key or value.
        n_rep (int): The number of times to repeat the heads.

    Returns:
        torch.Tensor: The repeated hidden states.
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
