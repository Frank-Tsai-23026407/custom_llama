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
    parser = argparse.ArgumentParser(description="Evaluate a model on WinoGrande with optional BFP quantization.")
    parser.add_argument("--bft", action="store_true", help="Apply Block Floating Point quantization.")
    parser.add_argument("--m_bit", type=int, default=4, help="Mantissa bits for BFP quantization.")
    parser.add_argument("--b_size", type=int, default=16, help="Block size for BFP quantization.")
    parser.add_argument("--model_path", type=str, default="TinyLlama/TinyLlama_v1.1", help="Path or hub ID for the model to evaluate (e.g., model/llama-3.2-1b/Llama-3.2-1B)")
    parser.add_argument("--backend", type=str, default="custom", choices=["custom","huggingface","clone"], help="Execution backend for LlamaMyModel.")
    
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

    # Load WinoGrande dataset
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation")

    def format_question(example):
        sentence = example["sentence"]
        option1 = example["option1"]
        option2 = example["option2"]
        
        # The placeholder is "_"
        context = sentence.split("_")[0]
        
        choices = [option1, option2]
        gold = int(example["answer"]) - 1 # "1" or "2" -> 0 or 1
        
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

    for batch in tqdm(data_loader, desc="Evaluating WinoGrande"):
        context = batch["context"][0]
        choices = [c[0] for c in batch["choices"]]
        true_label = batch["label"].item()

        with torch.no_grad():
            choice_total_nll = []
            choice_mean_nll = []
            for choice_text in choices:
                # For WinoGrande, no space between context and choice
                full_text = context + choice_text
                
                tensor_inputs = TU.tokenize_to_device(tokenizer, full_text, device, add_special_tokens=False)
                input_ids = tensor_inputs['input_ids']
                outputs = my_model.single_step(tensor_inputs)
                my_model.reset_kv_cache()
                
                shift_logits, shift_labels = TU.get_shifted(outputs, input_ids)
                loss = TU.per_token_cross_entropy(shift_logits, shift_labels)
                
                # Use context[:-1] to exclude the trailing underscore for token counting
                context_tokens_len = TU.count_tokens(tokenizer, context[:-1], add_special_tokens=False)
                
                choice_loss_continuation = loss[context_tokens_len-1:]
                
                # Calculate both total and mean negative log-likelihood
                if choice_loss_continuation.numel() > 0:
                    total_nll = -choice_loss_continuation.sum().item()
                    mean_nll = -choice_loss_continuation.mean().item()
                else:
                    total_nll = float('-inf')
                    mean_nll = float('-inf')
                
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
                print(f"\nContext:\n'{context}'")
                for idx, ch in enumerate(choices):
                    print(f"{chr(65+idx)}. {ch} (Total NLL: {choice_total_nll[idx]:.2f}, Mean NLL: {choice_mean_nll[idx]:.2f})")
                print(f"True Answer: {chr(65 + true_label)}")
                print(f"Predicted Answer (acc): {chr(65 + predicted_choice_idx)}")
                print(f"Predicted Answer (acc_norm): {chr(65 + predicted_choice_idx_norm)}")
                print("-" * 30)

    accuracy = correct_predictions / total_questions
    accuracy_norm = correct_predictions_norm / total_questions
    print(f"\n--- WinoGrande Evaluation Results ---")
    print(f"Total questions: {total_questions}")
    print(f"Correct predictions (acc): {correct_predictions}")
    print(f"Accuracy (acc): {accuracy:.4f}")
    print(f"Correct predictions (acc_norm): {correct_predictions_norm}")
    print(f"Accuracy (acc_norm): {accuracy_norm:.4f}")


if __name__ == "__main__":
    main()
