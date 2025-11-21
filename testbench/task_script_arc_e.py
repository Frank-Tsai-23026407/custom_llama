# standard libraries
from datasets import load_dataset
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer

import re
import datasets

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# my libraries
from llama_backend.custom.plain_script import *
from llama_backend.utils import *
from llama_backend.llama_my import LlamaMyModel
import task_utils as TU


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on ARC-Easy with optional BFP quantization.")
    parser.add_argument("--bft", action="store_true", help="Apply Block Floating Point quantization.")
    parser.add_argument("--m_bit", type=int, default=4, help="Mantissa bits for BFP quantization.")
    parser.add_argument("--b_size", type=int, default=16, help="Block size for BFP quantization.")
    parser.add_argument("--model_path", type=str, default="TinyLlama/TinyLlama_v1.1", help="Path or hub ID for the model to evaluate (e.g., model/llama-3.2-1b/Llama-3.2-1B)")
    parser.add_argument("--backend", type=str, default="custom", choices=["custom","huggingface","clone"], help="Execution backend for LlamaMyModel.")
    
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bar.")
    
    args = parser.parse_args()

    apply_bfp = args.bft
    bfp_mantissa_bits = args.m_bit
    bfp_block_size = args.b_size

    if apply_bfp:
        print(f"Applying BFP quantization with block_size={bfp_block_size} and mantissa_bits={bfp_mantissa_bits}")
    else:
        print("Running without BFP quantization.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dtype = torch.float32 
    my_model, tokenizer, device, dtype = TU.setup_model_and_tokenizer(
        model_path=args.model_path,
        backend=args.backend,
        apply_bfp=apply_bfp,
        bfp_block_size=bfp_block_size,
        bfp_mantissa_bits=bfp_mantissa_bits,
        dtype=dtype,
    )

    # Load ARC-Easy dataset
    dataset = load_dataset("ai2_arc", "ARC-Easy", split="test")

    def format_question(example):
        context = example["question"]
        choices = example["choices"]["text"]
        choice_labels = example["choices"]["label"]
        
        # Find the index of the correct answer key in the choice labels
        gold = choice_labels.index(example["answerKey"])
        
        return {
            "context": context,
            "choices": choices,
            "label": gold
        }
    formatted_dataset = dataset.map(format_question)

    data_loader = DataLoader(formatted_dataset, batch_size=1)

    correct_predictions = 0
    correct_predictions_norm = 0
    total_questions = 0
    
    # model created above via TU.setup_model_and_tokenizer

    # Determine where to write tqdm output
    tqdm_file = sys.stderr
    if not args.disable_tqdm:
        try:
            # Try to open /dev/tty to write progress bar directly to terminal
            # This bypasses 'tee' so it doesn't end up in the log file
            tqdm_file = open('/dev/tty', 'w')
        except Exception:
            # Fallback to stderr if /dev/tty is not available (e.g. in some non-interactive environments)
            pass

    for batch in tqdm(data_loader, desc="Evaluating ARC-Easy", disable=args.disable_tqdm, file=tqdm_file):
        context = batch["context"][0]
        choices = [c[0] for c in batch["choices"]]
        true_label = batch["label"].item()

        with torch.no_grad():
            choice_total_nll = []
            choice_mean_nll = []
            for choice_text in choices:
                total_nll, mean_nll = TU.evaluate_choice_likelihood(
                    my_model, tokenizer, context, choice_text, device, add_space=True
                )
                choice_total_nll.append(total_nll)
                choice_mean_nll.append(mean_nll)

            predicted_choice_idx = np.argmax(choice_total_nll)
            predicted_choice_idx_norm = np.argmax(choice_mean_nll)
            
            if predicted_choice_idx == true_label:
                correct_predictions += 1
            if predicted_choice_idx_norm == true_label:
                correct_predictions_norm += 1
            total_questions += 1
            
            if total_questions <= 5:
                predicted_answer_char = chr(65 + predicted_choice_idx)
                true_answer_char = chr(65 + true_label)
                print(f"\nContext:\n{context}")
                for idx, ch in enumerate(choices):
                    print(f"{chr(65+idx)}. {ch} (Total NLL: {choice_total_nll[idx]:.2f}, Mean NLL: {choice_mean_nll[idx]:.2f})")
                print(f"True Answer: {true_answer_char}")
                print(f"Predicted Answer (acc): {chr(65 + predicted_choice_idx)}")
                print(f"Predicted Answer (acc_norm): {chr(65 + predicted_choice_idx_norm)}")
                print("-" * 30)

    accuracy = correct_predictions / total_questions
    accuracy_norm = correct_predictions_norm / total_questions
    print(f"\n--- ARC-Easy Evaluation Results ---")
    print(f"Total questions: {total_questions}")
    print(f"Correct predictions (acc): {correct_predictions}")
    print(f"Accuracy (acc): {accuracy:.4f}")
    print(f"Correct predictions (acc_norm): {correct_predictions_norm}")
    print(f"Accuracy (acc_norm): {accuracy_norm:.4f}")


if __name__ == "__main__":
    main()
