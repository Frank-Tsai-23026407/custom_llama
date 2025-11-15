"""Precision policy for controlling compute dtypes in transformer operations."""
import torch


class PrecisionPolicy:
    """Configuration for controlling compute dtypes in attention and FFN.
    
    Parameters
    ----------
    attn_matmul_dtype : torch.dtype or None
        Dtype for Q/K/V projections and attention matmuls (QK^T, AV).
        If None, uses the tensor's intrinsic dtype.
    attn_softmax_dtype : torch.dtype or None
        Dtype for softmax computation (scores -> probabilities).
        If None, uses attention scores dtype.
    ffn_matmul_dtype : torch.dtype or None
        Dtype for feed-forward network linear layers.
        If None, uses the tensor's intrinsic dtype.
    rope_compute_dtype : torch.dtype or None
        Dtype used internally in RoPE (sin/cos computation).
        If None, uses float32.
    stable_softmax : bool
        If True, subtracts max before softmax for numerical stability.
    name : str
        Label for this policy.
    """
    
    def __init__(self,
                 attn_matmul_dtype=None,
                 attn_softmax_dtype=None,
                 ffn_matmul_dtype=None,
                 rope_compute_dtype=None,
                 stable_softmax=True,
                 name="custom"):
        self.attn_matmul_dtype = attn_matmul_dtype
        self.attn_softmax_dtype = attn_softmax_dtype
        self.ffn_matmul_dtype = ffn_matmul_dtype
        self.rope_compute_dtype = rope_compute_dtype if rope_compute_dtype is not None else torch.bfloat16
        self.stable_softmax = stable_softmax
        self.name = name
    
    @staticmethod
    def default():
        """Preserve existing behavior: rely on tensor intrinsic dtypes."""
        return PrecisionPolicy(
            attn_matmul_dtype=None,
            attn_softmax_dtype=None,
            ffn_matmul_dtype=None,
            rope_compute_dtype=torch.bfloat16,
            stable_softmax=True,
            name="default"
        )
    
    @staticmethod
    def match_hf():
        """Match HuggingFace behavior: compute attention/FFN in bfloat16."""
        return PrecisionPolicy(
            attn_matmul_dtype=torch.bfloat16,
            attn_softmax_dtype=torch.bfloat16,
            ffn_matmul_dtype=torch.bfloat16,
            rope_compute_dtype=torch.bfloat16,
            stable_softmax=True,
            name="match_hf"
        )
    
    @staticmethod
    def bf16_end_to_end():
        """Force all operations to bfloat16."""
        return PrecisionPolicy(
            attn_matmul_dtype=torch.bfloat16,
            attn_softmax_dtype=torch.bfloat16,
            ffn_matmul_dtype=torch.bfloat16,
            rope_compute_dtype=torch.bfloat16,
            stable_softmax=True,
            name="bf16"
        )
    
    def __repr__(self):
        return (f"PrecisionPolicy(name='{self.name}', "
                f"attn_matmul={self.attn_matmul_dtype}, "
                f"attn_softmax={self.attn_softmax_dtype}, "
                f"ffn_matmul={self.ffn_matmul_dtype}, "
                f"rope={self.rope_compute_dtype})")


def resolve_policy(policy):
    """Convert policy specification to PrecisionPolicy instance.
    
    Parameters
    ----------
    policy : None, str, or PrecisionPolicy
        - None or "default": use default policy
        - "match_hf" or "hf" or "float32": use HF-matching policy
        - "bf16" or "bfloat16": use bf16 end-to-end policy
        - PrecisionPolicy instance: return as-is
    
    Returns
    -------
    PrecisionPolicy
    """
    if policy is None:
        return PrecisionPolicy.default()
    if isinstance(policy, PrecisionPolicy):
        return policy
    if isinstance(policy, str):
        key = policy.lower().strip()
        if key in ("default", "orig", "original"):
            return PrecisionPolicy.default()
        if key in ("match_hf", "hf", "float32", "f32"):
            return PrecisionPolicy.match_hf()
        if key in ("bf16", "bfloat16"):
            return PrecisionPolicy.bf16_end_to_end()
        raise ValueError(f"Unknown precision policy string: {policy}")
    raise TypeError(f"policy must be None, str, or PrecisionPolicy instance, got {type(policy)}")
