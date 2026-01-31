# Utility helpers for task scripts to reduce duplication
from __future__ import annotations

import os
import re
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# my libraries
from llama_backend.utils import StopOnTokens
from llama_backend.llama_custom import CustomLlamaModel

# ==============================================================================
# 1. Setup & Preprocessing Helpers
# ==============================================================================

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


# ==============================================================================
# 2. Tokenization & Logit Helpers
# ==============================================================================

def tokenize_to_device(tokenizer: AutoTokenizer, text: str, device: torch.device,
                       add_special_tokens: bool = False, truncation: bool = True) -> Dict[str, torch.Tensor]:
    enc = tokenizer(text, return_tensors="pt", truncation=truncation, add_special_tokens=add_special_tokens)
    return {k: v.to(device) for k, v in enc.items()}

def count_tokens(tokenizer: AutoTokenizer, text: str, add_special_tokens: bool = False) -> int:
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens))

def get_shifted(outputs: torch.Tensor, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = outputs[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    return shift_logits, shift_labels

def per_token_cross_entropy(shift_logits: torch.Tensor, shift_labels: torch.Tensor) -> torch.Tensor:
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


# ==============================================================================
# 3. Core Evaluation Logic (DRY)
# ==============================================================================

def evaluate_choice_likelihood(
    model,
    tokenizer: AutoTokenizer,
    context: str,
    choice: str,
    device: torch.device,
    add_space: bool = True,
) -> Tuple[float, float, int]:
    """Evaluates the log-likelihood of a given text continuation (choice).
    Returns (total_ll, token_mean_ll, byte_length).
    """
    full_text = context + (" " if add_space else "") + choice
    
    # Tokenize and move to device
    tensor_inputs = tokenize_to_device(tokenizer, full_text, device, add_special_tokens=False)
    input_ids = tensor_inputs['input_ids']
    
    # Get model outputs
    with torch.no_grad():
        if hasattr(model, "single_step"):
            logits = model.single_step(tensor_inputs)
            if hasattr(model, "reset_kv_cache"):
                model.reset_kv_cache()
        else:
            outputs = model(**tensor_inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
    
    # Compute loss
    shift_logits, shift_labels = get_shifted(logits, input_ids)
    loss = per_token_cross_entropy(shift_logits, shift_labels)
    
    # Get context token count
    context_tokens_len = count_tokens(tokenizer, context, add_special_tokens=False)
    
    # Extract continuation loss
    choice_loss_continuation = loss[context_tokens_len-1:]
    
    if choice_loss_continuation.numel() > 0:
        total_ll = -choice_loss_continuation.sum().item()
        mean_ll = -choice_loss_continuation.mean().item()
        return total_ll, mean_ll, len(choice.encode("utf-8"))
    else:
        return float('-inf'), float('-inf'), 0

def _generic_mc_eval(
    model, tokenizer, device, dataset, doc_to_mc_fn: Callable, task_name: str, 
    add_space: bool = True, use_byte_norm: bool = False
) -> Tuple[float, float]:
    """DRY core for multiple-choice evaluation loops."""
    correct, correct_norm, total = 0, 0, 0
    
    for doc in tqdm(dataset, desc=f"Evaluating {task_name}"):
        context, choices, gold = doc_to_mc_fn(doc)
        results = [evaluate_choice_likelihood(model, tokenizer, context, c, device, add_space) for c in choices]
        
        # total_ll = r[0], token_mean_ll = r[1], byte_len = r[2]
        lls = [r[0] for r in results]
        
        if use_byte_norm:
            # Normalization by byte length (e.g., HellaSwag style)
            lls_norm = [r[0] / r[2] if r[2] > 0 else -1e9 for r in results]
        else:
            # Normalization by token count (mean NLL)
            lls_norm = [r[1] for r in results]

        if np.argmax(lls) == gold:
            correct += 1
        if np.argmax(lls_norm) == gold:
            correct_norm += 1
        total += 1

    acc = correct / total
    acc_norm = correct_norm / total
    print(f"{task_name} Accuracy (acc): {acc:.4f} | (acc_norm): {acc_norm:.4f}")
    return acc, acc_norm


# ==============================================================================
# 4. Task Wrappers
# ==============================================================================

def evaluate_hellaswag(model, tokenizer, device, max_samples=None):
    dataset = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def process_doc(doc):
        # Preprocessing specific to HellaSwag
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        ctx = ctx.replace(" [title]", ". ")
        ctx = re.sub("\\[.*?\\]", "", ctx).replace("  ", " ").strip()
        query = f"{doc['activity_label']}: {ctx}"
        choices = [re.sub("\\[.*?\\]", "", c).replace("  ", " ").strip() for c in doc["endings"]]
        return query, choices, int(doc["label"])
    
    return _generic_mc_eval(model, tokenizer, device, dataset, process_doc, "HellaSwag", use_byte_norm=True)

def evaluate_arc(model, tokenizer, device, variant="Challenge", split="test", max_samples=None):
    dataset = load_dataset("ai2_arc", f"ARC-{variant}", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def process_doc(doc):
        return doc["question"], doc["choices"]["text"], doc["choices"]["label"].index(doc["answerKey"])
    
    return _generic_mc_eval(model, tokenizer, device, dataset, process_doc, f"ARC-{variant}")

def evaluate_obqa(model, tokenizer, device, split="test", max_samples=None):
    dataset = load_dataset("openbookqa", "main", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    
    def process_doc(doc):
        return doc["question_stem"], doc["choices"]["text"], label_map[doc["answerKey"]]
    
    return _generic_mc_eval(model, tokenizer, device, dataset, process_doc, "OpenBookQA")

def evaluate_boolq(model, tokenizer, device, split="validation", max_samples=None):
    dataset = load_dataset("super_glue", "boolq", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def process_doc(doc):
        return doc["passage"] + "\n" + doc["question"] + "?", ["no", "yes"], int(doc["label"])
        
    return _generic_mc_eval(model, tokenizer, device, dataset, process_doc, "BoolQ")

def evaluate_piqa(model, tokenizer, device, split="validation", max_samples=None):
    dataset = load_dataset("gimmaru/piqa", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        
    def process_doc(doc):
        return doc["goal"], [doc["sol1"], doc["sol2"]], int(doc["label"])
        
    return _generic_mc_eval(model, tokenizer, device, dataset, process_doc, "PIQA")

def evaluate_winogrande(model, tokenizer, device, split="validation", max_samples=None):
    dataset = load_dataset("winogrande", "winogrande_xl", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        
    def process_doc(doc):
        # Part before '_' is context
        return doc["sentence"].split("_")[0], [doc["option1"], doc["option2"]], int(doc["answer"]) - 1
        
    return _generic_mc_eval(model, tokenizer, device, dataset, process_doc, "Winogrande", add_space=False)


# ==============================================================================
# 5. Perplexity (PPL)
# ==============================================================================

def evaluate_ppl(model, tokenizer, device, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="test", limit_tokens=None):
    dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    
    max_length = 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)
    if limit_tokens: seq_len = min(seq_len, limit_tokens)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Evaluating PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            if hasattr(model, "single_step"):
                logits = model.single_step({"input_ids": input_ids})
                sl, sk = get_shifted(logits, input_ids)
                # Compute loss for all tokens, then mask
                loss_all = per_token_cross_entropy(sl, sk)
                shift_target_ids = target_ids[:, 1:]
                mask = (shift_target_ids != -100)
                neg_log_likelihood = loss_all[mask.view(-1)].mean()
            else:
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood * trg_len)
        prev_end_loc = end_loc
        if end_loc == seq_len: break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc).item()
    print(f"{dataset_name} PPL: {ppl:.4f}")
    return ppl
