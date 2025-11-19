#!/usr/bin/env python
import argparse
import os
import sys
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quantize_model_script.runtime_quantizer import apply_runtime_quantization
import testbench.task_utils as TU


def preprocess(text: str) -> str:
    import re
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset) -> "datasets.Dataset":
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)


def resolve_model_path(model: str) -> str:
    ws_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    presets = {
        "llama-3.2-1b": os.path.join(ws_root, "model/llama-3.2-1b/Llama-3.2-1B"),
        "tinyllama": os.path.join(ws_root, "model/tinyllama/TinyLlama_1.1v"),
    }
    return presets.get(model, model)


def evaluate_choice_likelihood_hf(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: str,
    choice: str,
    device: torch.device,
) -> Tuple[float, float]:
    # Build full text and tokenize
    full_text = context + " " + choice
    tensor_inputs = TU.tokenize_to_device(tokenizer, full_text, device, add_special_tokens=False)
    input_ids = tensor_inputs["input_ids"]

    with torch.no_grad():
        outputs = model(**tensor_inputs)
        logits = outputs.logits

    shift_logits, shift_labels = TU.get_shifted(logits, input_ids)
    loss = TU.per_token_cross_entropy(shift_logits, shift_labels)

    # Determine continuation region based on context token count
    context_tokens_len = TU.count_tokens(tokenizer, context, add_special_tokens=False)
    choice_loss_continuation = loss[context_tokens_len - 1 :]

    if choice_loss_continuation.numel() > 0:
        total_nll = -choice_loss_continuation.sum().item()
        mean_nll = -choice_loss_continuation.mean().item()
        return total_nll, mean_nll
    else:
        return float("-inf"), float("-inf")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TinyLlama HellaSwag with runtime 2D-BFP (no saving)"
    )
    parser.add_argument("--model", type=str, default="tinyllama", help="Model preset or path")
    parser.add_argument("--block-height", type=int, required=True, help="2D block height")
    parser.add_argument("--block-width", type=int, required=True, help="2D block width")
    parser.add_argument("--mantissa-bits", type=int, default=4, help="Mantissa bits for BFP")
    parser.add_argument("--load-dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Model load dtype")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit eval set for quick runs")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    print("=" * 80)
    print("HellaSwag Runtime 2D-BFP Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Load dtype: {args.load_dtype}")
    print(f"Block size: {args.block_height}x{args.block_width}")
    print(f"Mantissa bits: {args.mantissa_bits}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load HF model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype_map[args.load_dtype], low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    # Apply runtime 2D-BFP quantization in-place
    apply_runtime_quantization(
        model,
        method="bfp",
        block_height=args.block_height,
        block_width=args.block_width,
        mantissa_bits=args.mantissa_bits,
        verbose=True,
    )

    # Load HellaSwag and preprocess
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Using {len(dataset)} samples for quick testing")

    dataset = process_docs(dataset)

    correct = 0
    correct_norm = 0
    total = 0

    for item in dataset:
        context = item["query"]
        choices = item["choices"]
        gold = int(item["gold"]) if "gold" in item else int(item["label"])  # compatibility

        scores = []
        scores_norm = []
        for ch in choices:
            total_nll, mean_nll = evaluate_choice_likelihood_hf(model, tokenizer, context, ch, device)
            scores.append(total_nll)
            scores_norm.append(mean_nll)

        pred = int(torch.tensor(scores).argmax().item())
        pred_norm = int(torch.tensor(scores_norm).argmax().item())

        if pred == gold:
            correct += 1
        if pred_norm == gold:
            correct_norm += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    acc_norm = correct_norm / total if total > 0 else 0.0

    print("-" * 80)
    print(f"Accuracy:      {acc:.4f} ({correct}/{total})")
    print(f"Accuracy_norm: {acc_norm:.4f} ({correct_norm}/{total})")
    print("-" * 80)


if __name__ == "__main__":
    main()
