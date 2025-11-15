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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on WinoGrande with optional BFP quantization.")
    parser.add_argument("--bft", action="store_true", help="Apply Block Floating Point quantization.")
    parser.add_argument("--m_bit", type=int, default=4, help="Mantissa bits for BFP quantization.")
    parser.add_argument("--b_size", type=int, default=16, help="Block size for BFP quantization.")
    parser.add_argument("--model_path", type=str, default="TinyLlama/TinyLlama_v1.1", help="Path or hub ID for the model to evaluate (e.g., model/llama-3.2-1b/Llama-3.2-1B)")
    
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

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
    total_questions = 0
    
    my_model = LlamaMyModel(
        model_name=args.model_path,
        device=device,
        stop_criteria=StopOnTokens(),
        dtype=dtype,
        apply_bfp=apply_bfp,
        bfp_block_size=bfp_block_size,
        bfp_mantissa_bits=bfp_mantissa_bits
    )

    for batch in tqdm(data_loader, desc="Evaluating WinoGrande"):
        context = batch["context"][0]
        choices = [c[0] for c in batch["choices"]]
        true_label = batch["label"].item()

        with torch.no_grad():
            choice_log_likelihoods = []
            for choice_text in choices:
                # For WinoGrande, the choice text itself is the continuation
                full_text = context + choice_text
                
                tokenized_input = tokenizer(full_text, return_tensors="pt", truncation=True, add_special_tokens=False)
                input_ids = tokenized_input.input_ids

                outputs = my_model.single_step(tokenized_input)
                my_model.reset_kv_cache()
                
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                context_tokens_len = tokenizer(context[:-1], return_tensors="pt", truncation=True).input_ids.shape[1]
                
                choice_loss_continuation = loss[context_tokens_len-1:]
                
                # print the context, the tokenized context, the tokenized context input, and the length of both
                if len(choice_loss_continuation) == 2:
                    print(f"Context: '{context}'")
                    print(f"Full Text: '{full_text}'")
                    print(f"Tokenized Context: {tokenizer(context, return_tensors='pt', truncation=True).input_ids}")
                    print(f"Tokenized Context Input: {tokenized_input.input_ids}")
                    print(f"Length of Context: {context_tokens_len}")
                
                # This is equivalent to acc_norm in lm-eval-harness for this task
                avg_log_likelihood = -choice_loss_continuation.mean().item()
                
                if torch.isnan(torch.tensor(avg_log_likelihood)):
                    print("\n--- NaN Detected ---")
                    print(f"Context: '{context}'")
                    print(f"Problematic Choice: '{choice_text}'")
                    print(f"Loss tensor (all tokens): {loss}")
                    print(f"Choice loss continuation: {choice_loss_continuation}")
                    
                    # Find which token in the choice is causing the issue
                    nan_indices = torch.where(torch.isnan(choice_loss_continuation))[0]
                    if len(nan_indices) > 0:
                        problem_index_in_continuation = nan_indices[0].item()
                        problem_index_in_input = context_tokens_len - 1 + problem_index_in_continuation
                        
                        problematic_logits = shift_logits.view(-1, shift_logits.size(-1))[problem_index_in_input]
                        problematic_label = shift_labels.view(-1)[problem_index_in_input]
                        
                        print(f"Problematic token index in continuation: {problem_index_in_continuation}")
                        print(f"Problematic token index in full input: {problem_index_in_input}")
                        print(f"Problematic Logits (shape {problematic_logits.shape}): {problematic_logits}")
                        print(f"Correct next token ID (label): {problematic_label}")
                        
                        # Check the probability of the correct token
                        probs = torch.softmax(problematic_logits, dim=-1)
                        correct_token_prob = probs[problematic_label]
                        print(f"Probability of correct token: {correct_token_prob.item()}")
                    print("--------------------\n")

                choice_log_likelihoods.append(avg_log_likelihood)

            predicted_choice_idx = np.argmax(choice_log_likelihoods)
            
            if predicted_choice_idx == true_label:
                correct_predictions += 1
            total_questions += 1
            
            if total_questions <= 5:
                predicted_answer_char = chr(65 + predicted_choice_idx)
                true_answer_char = chr(65 + true_label)
                print(f"\nContext:\n'{context}'")
                for idx, ch in enumerate(choices):
                    print(f"{chr(65+idx)}. {ch} (Log-likelihood: {choice_log_likelihoods[idx]:.2f})")
                print(f"True Answer: {true_answer_char}")
                print(f"Predicted Answer: {predicted_answer_char}")
                print("-" * 30)

    accuracy = correct_predictions / total_questions
    print(f"\n--- WinoGrande Evaluation Results ---")
    print(f"Total questions: {total_questions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
