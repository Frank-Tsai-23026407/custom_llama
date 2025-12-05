import argparse
import torch
import os
import sys
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import re
import datasets
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_my import LlamaMyModel
from llama_backend.utils import StopOnTokens
import testbench.task_utils as TU

def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc
    return dataset.map(_process_doc)

def collect_activation_scales(model, tokenizer, num_samples=128, device="cuda"):
    print(f"Collecting activation scales from {num_samples} samples of WikiText...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    dataset = dataset.select(range(num_samples))
    
    # Concatenate text
    text = "\n\n".join(dataset["text"])
    # Tokenize
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    # We might need to chunk if tokens are too long, but let's try a smaller chunk if needed.
    # Usually calibration is done on a few batches of sequence length 2048.
    # Let's just take the first 2048 tokens * num_batches.
    seq_len = 2048
    num_batches = num_samples // 4 # heuristic
    if num_batches < 1: num_batches = 1
    
    # Actually, let's just use a simple loop over chunks
    
    scales = {}
    
    def get_activation_hook(name):
        def hook(module, input, output):
            # input is a tuple (x,), x shape [batch, seq, features]
            x = input[0].detach()
            # Reshape to [batch*seq, features]
            x = x.view(-1, x.shape[-1])
            # Calculate mean absolute value per channel
            # We want to accumulate this. For simplicity, let's just take the mean of the first batch 
            # or moving average. 
            # Better: accumulate sum and count.
            
            if name not in scales:
                scales[name] = torch.abs(x).mean(dim=0)
            else:
                # Simple moving average or just overwrite? 
                # Standard Wanda uses calibration set. Let's average over batches.
                # For simplicity in this script, let's just use one large batch or average.
                # Let's do a running average.
                scales[name] = (scales[name] + torch.abs(x).mean(dim=0)) / 2
        return hook

    hooks = []
    for name, module in model.model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(get_activation_hook(name)))
            
    print("Running forward pass for calibration...")
    # Just run one pass with a reasonable context length
    # If tokens are longer than model max length, truncate.
    max_len = 2048
    if tokens.shape[1] > max_len:
        tokens = tokens[:, :max_len]
        
    with torch.no_grad():
        model.model(tokens) # Use the underlying HF model for forward pass to trigger hooks
        
    for hook in hooks:
        hook.remove()
        
    print(f"Collected scales for {len(scales)} layers.")
    return scales

def main():
    parser = argparse.ArgumentParser(description="Run HellaSwag evaluation with Wanda Pruning")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-quantized model")
    parser.add_argument("--sparsity_ratio", type=float, default=0.5, help="Sparsity ratio")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bar")
    
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    device = torch.device(args.device)
    dtype = torch.float16

    model = LlamaMyModel(
        model_name=args.model_path,
        device=args.device,
        dtype=dtype,
        apply_bfp=False,
        sparsity_ratio=0.0 # Do not apply naive pruning
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Collect scales
    scales = collect_activation_scales(model, tokenizer, device=args.device)
    
    # Apply Wanda Pruning
    model.apply_wanda_pruning(args.sparsity_ratio, scales)

    # Load HellaSwag dataset
    print("Loading HellaSwag dataset...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Using {len(dataset)} samples for testing")
    
    dataset = process_docs(dataset)

    def format_question(example):
        context = example["query"]
        endings = example["choices"]
        return {
            "context": context,
            "choices": endings,
            "label": example["gold"]
        }
    
    formatted_dataset = dataset.map(format_question)
    data_loader = DataLoader(formatted_dataset, batch_size=1)

    correct_predictions_acc = 0
    correct_predictions_acc_norm = 0
    total_questions = 0

    tqdm_file = sys.stderr
    if not args.disable_tqdm:
        try:
            tqdm_file = open('/dev/tty', 'w')
        except Exception:
            pass

    print("Starting evaluation...")
    for batch in tqdm(data_loader, desc="Evaluating HellaSwag", disable=args.disable_tqdm, file=tqdm_file):
        contexts = batch["context"]
        choices_list = batch["choices"]
        true_labels = batch["label"]

        with torch.no_grad():
            for i in range(len(contexts)):
                context = contexts[i]
                choices = [c for c in choices_list]
                true_label = true_labels[i].item()

                choice_log_likelihoods_acc = []
                choice_log_likelihoods_acc_norm = []

                for choice_text in choices:
                    full_text = context + " " + choice_text[0]
                    if full_text[-1] == ".":
                        full_text = full_text[:-1]
                    
                    encoding = tokenizer(full_text, return_tensors="pt", truncation=True, add_special_tokens=False)
                    tensor_inputs = TU.tokenize_to_device(tokenizer, full_text, device, add_special_tokens=False)
                    input_ids = tensor_inputs['input_ids']

                    outputs = model.single_step(tensor_inputs)
                    model.reset_kv_cache()
                    
                    shift_logits, shift_labels = TU.get_shifted(outputs, input_ids)
                    loss = TU.per_token_cross_entropy(shift_logits, shift_labels)
                    
                    context_len_chars = len(context)
                    choice_start_char_idx = context_len_chars + 1
                    continuation_start_token_idx = encoding.char_to_token(0, choice_start_char_idx)

                    if continuation_start_token_idx is not None:
                        continuation_loss = loss[continuation_start_token_idx - 1:]
                    else:
                        continuation_loss = torch.tensor([], device=loss.device)

                    if continuation_loss.numel() > 0:
                        sum_loss = continuation_loss.sum().item()
                    else:
                        sum_loss = float('inf')
                    
                    choice_text_str = choice_text[0]
                    choice_byte_length = len(choice_text_str.encode('utf-8'))
                    norm_loss = sum_loss / choice_byte_length if choice_byte_length > 0 else float('inf')
                    
                    choice_log_likelihoods_acc.append(-sum_loss)
                    choice_log_likelihoods_acc_norm.append(-norm_loss)
                
                predicted_choice_idx_acc = np.argmax(choice_log_likelihoods_acc)
                predicted_choice_idx_acc_norm = np.argmax(choice_log_likelihoods_acc_norm)

                if predicted_choice_idx_acc == true_label:
                    correct_predictions_acc += 1
                if predicted_choice_idx_acc_norm == true_label:
                    correct_predictions_acc_norm += 1
                total_questions += 1

    accuracy_acc = correct_predictions_acc / total_questions
    accuracy_acc_norm = correct_predictions_acc_norm / total_questions
    
    print(f"\n--- HellaSwag Evaluation Results ---")
    print(f"Total questions: {total_questions}")
    print(f"Correct predictions (acc): {correct_predictions_acc}")
    print(f"Correct predictions (acc_norm): {correct_predictions_acc_norm}")
    print(f"Accuracy (acc): {accuracy_acc:.4f}")
    print(f"Accuracy (acc_norm): {accuracy_acc_norm:.4f}")

if __name__ == "__main__":
    main()
