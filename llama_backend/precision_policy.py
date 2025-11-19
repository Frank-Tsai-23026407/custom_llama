"""Precision policy for controlling compute dtypes in transformer operations."""
import torch


class PrecisionPolicy:
    """A configuration for controlling compute dtypes in transformer operations.

    This class defines a policy for specifying the `torch.dtype` to be used for
    different parts of a transformer block, such as attention matrix multiplications,
    softmax, and feed-forward network computations. This allows for fine-grained
    control over performance and numerical precision.

    Attributes:
        attn_matmul_dtype (torch.dtype or None): The dtype for Q/K/V projections
            and attention matrix multiplications (QK^T, AV). If None, the tensor's
            intrinsic dtype is used.
        attn_softmax_dtype (torch.dtype or None): The dtype for the softmax
            computation. If None, the attention scores' dtype is used.
        ffn_matmul_dtype (torch.dtype or None): The dtype for the linear layers
            in the feed-forward network. If None, the tensor's intrinsic dtype
            is used.
        rope_compute_dtype (torch.dtype or None): The dtype for internal RoPE
            (sin/cos) computations. Defaults to `torch.bfloat16`.
        stable_softmax (bool): If True, subtracts the maximum value from the
            attention scores before the softmax for numerical stability.
        name (str): A descriptive name for the policy.
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
        """Creates a policy that preserves the intrinsic dtypes of tensors.

        This policy does not force any dtype conversions, making it suitable for
        debugging or maintaining the original behavior of a model.

        Returns:
            PrecisionPolicy: A policy with all dtype settings set to None.
        """
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
        """Creates a policy to match Hugging Face's bfloat16 behavior.

        This policy sets all major computation dtypes to `torch.bfloat16`, which
        is a common configuration for modern Hugging Face models to achieve good
        performance on compatible hardware.

        Returns:
            PrecisionPolicy: A policy configured for bfloat16 computation.
        """
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
        """Creates a policy that forces all operations to bfloat16.

        This is a strict policy that ensures all transformer computations are
        performed in `torch.bfloat16` for maximum performance on supported GPUs.

        Returns:
            PrecisionPolicy: A policy with all dtypes set to `torch.bfloat16`.
        """
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
    """Resolves a flexible policy input into a PrecisionPolicy instance.

    This utility function allows users to specify a precision policy in multiple
    ways: as a string alias (e.g., "match_hf", "bf16"), as a `PrecisionPolicy`
    object, or as `None` to get the default policy.

    Args:
        policy (None, str, or PrecisionPolicy): The policy to resolve.
            - If `None` or "default", returns `PrecisionPolicy.default()`.
            - If a string alias, returns the corresponding static policy.
            - If a `PrecisionPolicy` instance, returns it directly.

    Returns:
        PrecisionPolicy: The resolved `PrecisionPolicy` object.

    Raises:
        ValueError: If the policy string is unknown.
        TypeError: If the policy is of an unsupported type.
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
