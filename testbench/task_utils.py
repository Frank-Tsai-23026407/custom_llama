# Utility helpers for task scripts to reduce duplication
from __future__ import annotations

import os
from typing import Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer

# my libraries
from llama_backend.utils import StopOnTokens
from llama_backend.llama_custom import CustomLlamaModel


def setup_model_and_tokenizer(
    model_path: str,
    backend: str = "custom",
    apply_bfp: bool = False,
    bfp_block_size: int = 16,
    bfp_mantissa_bits: int = 4,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    extra_model_kwargs: Optional[Dict] = None,
) -> Tuple[CustomLlamaModel, AutoTokenizer, torch.device, torch.dtype]:
    """Initializes and configures a model, tokenizer, and device for evaluation.

    This is a convenience function that streamlines the setup process for running
    evaluation tasks. It handles device selection, tokenizer loading, and the
    instantiation of the `CustomLlamaModel` with specified quantization and backend
    configurations.

    Args:
        model_path (str): The path to the pretrained model.
        backend (str, optional): The execution backend ('custom', 'clone', 'huggingface').
            Defaults to "custom".
        apply_bfp (bool, optional): Whether to apply BFP quantization. Defaults to False.
        bfp_block_size (int, optional): The block size for BFP. Defaults to 16.
        bfp_mantissa_bits (int, optional): The number of mantissa bits for BFP. Defaults to 4.
        dtype (torch.dtype, optional): The primary data type for the model. Defaults to torch.float32.
        device (Optional[torch.device], optional): The device to use. If None, it is
            auto-detected. Defaults to None.
        extra_model_kwargs (Optional[Dict], optional): Additional keyword arguments to pass
            to the `CustomLlamaModel` constructor, useful for backend-specific settings.
            Defaults to None.

    Returns:
        Tuple[CustomLlamaModel, AutoTokenizer, torch.device, torch.dtype]: A tuple containing
            the initialized model, tokenizer, device, and dtype.
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

    model = CustomLlamaModel(**model_kwargs)
    return model, tokenizer, device, dtype


def tokenize_to_device(tokenizer: AutoTokenizer, text: str, device: torch.device,
                       add_special_tokens: bool = False, truncation: bool = True) -> Dict[str, torch.Tensor]:
    """Tokenizes text and moves the resulting tensors to a specified device.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use.
        text (str): The text to tokenize.
        device (torch.device): The device to move the tensors to.
        add_special_tokens (bool, optional): Whether to add special tokens. Defaults to False.
        truncation (bool, optional): Whether to truncate the input. Defaults to True.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of tensors (e.g., 'input_ids', 'attention_mask').
    """
    enc = tokenizer(text, return_tensors="pt", truncation=truncation, add_special_tokens=add_special_tokens)
    return {k: v.to(device) for k, v in enc.items()}

def token_count(tokenizer: AutoTokenizer, text: str, truncation: bool = True) -> int:
    """Counts the number of tokens in a string without adding special tokens.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use.
        text (str): The text to count the tokens of.
        truncation (bool, optional): Whether to truncate the input. Defaults to True.

    Returns:
        int: The number of tokens.
    """
    return tokenizer(text, return_tensors="pt", truncation=truncation, add_special_tokens=False).input_ids.shape[1]


def get_shifted(outputs: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aligns logits and labels for calculating next-token prediction loss.

    This function shifts the logits and labels by one position to create a causal
    language modeling objective. `logits[i]` will correspond to the prediction
    for `labels[i]`.

    Args:
        outputs (torch.Tensor): The raw logits from the model of shape (batch, seq_len, vocab_size).
        input_ids (torch.Tensor): The input token IDs of shape (batch, seq_len).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of (shifted_logits, shifted_labels).
    """
    shift_logits = outputs[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    return shift_logits, shift_labels


def per_token_cross_entropy(shift_logits: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
    """Calculates the cross-entropy loss for each token without reduction.

    Args:
        shift_logits (torch.Tensor): The model's predicted logits, already shifted.
        shift_labels (torch.Tensor): The ground truth labels, already shifted.

    Returns:
        torch.Tensor: A 1D tensor where each element is the cross-entropy loss
            for the corresponding token.
    """
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def count_tokens(tokenizer: AutoTokenizer, text: str, add_special_tokens: bool = False) -> int:
    """Efficiently counts tokens without creating tensors.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use.
        text (str): The text to encode.
        add_special_tokens (bool, optional): Whether to include special tokens in the count.
            Defaults to False.

    Returns:
        int: The number of tokens.
    """
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def evaluate_choice_likelihood(
    model,
    tokenizer: AutoTokenizer,
    context: str,
    choice: str,
    device: torch.device,
    add_space: bool = True,
) -> Tuple[float, float]:
    """Evaluates the log-likelihood of a given text continuation (choice).

    This function is central to multiple-choice evaluation tasks. It computes the
    negative log-likelihood (NLL) of a `choice` string, given a `context`. This
    value is used to determine which of several choices the model finds most plausible.

    It returns both the total NLL (sum of losses) and the mean NLL (loss normalized
    by the number of tokens in the choice), which are used for different accuracy metrics.

    Args:
        model (CustomLlamaModel): The model instance to evaluate with.
        tokenizer (AutoTokenizer): The corresponding tokenizer.
        context (str): The preceding text or question.
        choice (str): The continuation or answer to be evaluated.
        device (torch.device): The device to run the computation on.
        add_space (bool, optional): If True, adds a space between context and choice.
            Defaults to True.

    Returns:
        Tuple[float, float]: A tuple containing the total negative log-likelihood
            and the mean negative log-likelihood. Returns (-inf, -inf) if the
            choice has zero tokens.
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
