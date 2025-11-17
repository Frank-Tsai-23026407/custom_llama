# Utility helpers for task scripts to reduce duplication
from __future__ import annotations

import os
from typing import Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer

# my libraries
from llama_backend.utils import StopOnTokens
from llama_backend.llama_my import LlamaMyModel


def setup_model_and_tokenizer(
    model_path: str,
    backend: str = "custom",
    apply_bfp: bool = False,
    bfp_block_size: int = 16,
    bfp_mantissa_bits: int = 4,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    extra_model_kwargs: Optional[Dict] = None,
) -> Tuple[LlamaMyModel, AutoTokenizer, torch.device, torch.dtype]:
    """
    Create device, tokenizer, and LlamaMyModel with common defaults.
    extra_model_kwargs: forwarded to LlamaMyModel (e.g., clone/huggingface precision knobs).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_kwargs = dict(
        model_name=model_path,
        device=device,
        stop_criteria=StopOnTokens(),
        dtype=dtype,
        apply_bfp=apply_bfp,
        bfp_block_size=bfp_block_size,
        bfp_mantissa_bits=bfp_mantissa_bits,
        backend=backend,
    )
    if extra_model_kwargs:
        model_kwargs.update(extra_model_kwargs)

    model = LlamaMyModel(**model_kwargs)
    return model, tokenizer, device, dtype


def tokenize_to_device(tokenizer: AutoTokenizer, text: str, device: torch.device,
                       add_special_tokens: bool = False, truncation: bool = True) -> Dict[str, torch.Tensor]:
    enc = tokenizer(text, return_tensors="pt", truncation=truncation, add_special_tokens=add_special_tokens)
    return {k: v.to(device) for k, v in enc.items()}

def token_count(tokenizer: AutoTokenizer, text: str, truncation: bool = True) -> int:
    """Return number of tokens without adding special tokens."""
    return tokenizer(text, return_tensors="pt", truncation=truncation, add_special_tokens=False).input_ids.shape[1]


def get_shifted(outputs: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (shift_logits, shift_labels) aligned for next-token loss."""
    shift_logits = outputs[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    return shift_logits, shift_labels


def per_token_cross_entropy(shift_logits: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
    """Return per-token cross entropy loss (no reduction)."""
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def count_tokens(tokenizer: AutoTokenizer, text: str, add_special_tokens: bool = False) -> int:
    """Return the number of tokens for the given text (without creating tensors on device)."""
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def evaluate_choice_likelihood(
    model,
    tokenizer: AutoTokenizer,
    context: str,
    choice: str,
    device: torch.device,
    add_space: bool = True,
) -> Tuple[float, float]:
    """
    Evaluate log-likelihood for a single choice continuation.
    Returns both total and mean negative log-likelihood.
    
    Args:
        model: LlamaMyModel instance
        tokenizer: tokenizer instance
        context: context string
        choice: choice string to evaluate
        device: torch device
        add_space: whether to add a space between context and choice
    
    Returns:
        Tuple of (total_nll, mean_nll):
        - total_nll: sum of negative log-likelihoods (for acc - unnormalized)
        - mean_nll: average negative log-likelihood (for acc_norm - normalized by length)
    """
    full_text = context + (" " if add_space else "") + choice
    
    # Tokenize and move to device
    tensor_inputs = tokenize_to_device(tokenizer, full_text, device, add_special_tokens=False)
    input_ids = tensor_inputs['input_ids']
    
    # Get model outputs
    outputs = model.single_step(tensor_inputs)
    model.reset_kv_cache()
    
    # Compute loss
    shift_logits, shift_labels = get_shifted(outputs, input_ids)
    loss = per_token_cross_entropy(shift_logits, shift_labels)
    
    # Get context token count (without special tokens)
    context_tokens_len = count_tokens(tokenizer, context, add_special_tokens=False)
    
    # Extract continuation loss (loss[k] corresponds to predicting token k+1)
    choice_loss_continuation = loss[context_tokens_len-1:]
    
    # Return both total and mean negative log-likelihood
    if choice_loss_continuation.numel() > 0:
        total_nll = -choice_loss_continuation.sum().item()
        mean_nll = -choice_loss_continuation.mean().item()
        return total_nll, mean_nll
    else:
        return float('-inf'), float('-inf')
