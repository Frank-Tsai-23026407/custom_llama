# standard libraries
from datasets import load_dataset
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import re
import datasets

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# my libraries
from llama_backend.custom.plain_script import *
from llama_backend.utils import *
from llama_backend.llama_my import LlamaMyModel


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on HellaSwag with optional BFP quantization.")
    parser.add_argument("--bft", action="store_true", help="Apply Block Floating Point quantization.")
    parser.add_argument("--m_bit", type=int, default=4, help="Mantissa bits for BFP quantization.")
    parser.add_argument("--b_size", type=int, default=16, help="Block size for BFP quantization.")
    parser.add_argument("--awq", action="store_true", help="Use a pre-quantized AWQ model.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the quantized model for evaluation mode.")
    parser.add_argument("--backend", type=str, default="huggingface", choices=["custom","huggingface","clone"], help="Execution backend for LlamaMyModel.")
    
    # Precision control arguments for clone backend
    parser.add_argument("--compute_dtype", type=str, default=None, choices=["fp32", "bf16", "fp16"], 
                        help="Compute dtype for clone backend (None, fp32, bf16, or fp16)")
    parser.add_argument("--rope_cache_dtype", type=str, default=None, choices=["fp32", "bf16", "fp16"],
                        help="RoPE cache dtype (None, fp32, bf16, or fp16)")
    parser.add_argument("--softmax_fp32", action="store_true", 
                        help="Use FP32 for attention softmax in clone backend")
    
    # Precision policy for custom backend
    parser.add_argument("--precision_policy", type=str, default=None, choices=["default", "match_hf", "bf16"],
                        help="Precision policy for custom backend (default, match_hf, or bf16)")
    
    # Sample size for quick testing
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for quick testing)")
    
    args = parser.parse_args()

    apply_bfp = args.bft
    bfp_mantissa_bits = args.m_bit
    bfp_block_size = args.b_size

    # Parse dtype arguments
    dtype_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        None: None
    }
    compute_dtype = dtype_map.get(args.compute_dtype, None)
    rope_cache_dtype = dtype_map.get(args.rope_cache_dtype, None)
    
    # Print configuration
    print("\n" + "="*60)
    print("Configuration:")
    print(f"  Backend: {args.backend}")
    if args.backend == "clone":
        print(f"  Compute dtype: {args.compute_dtype if args.compute_dtype else 'default (bf16)'}")
        print(f"  RoPE cache dtype: {args.rope_cache_dtype if args.rope_cache_dtype else 'default (fp32)'}")
        print(f"  Softmax FP32: {args.softmax_fp32}")
    elif args.backend == "custom":
        print(f"  Precision policy: {args.precision_policy if args.precision_policy else 'default'}")
    print("="*60 + "\n")

    # Corrected logic to prioritize --model_path
    if args.model_path != None:
        print(f"Providing model path {args.model_path}")
        model_path = args.model_path
        apply_bfp = False
    elif args.awq:
        print("Using pre-quantized AWQ model.")
        model_path = "model/tinyllama/TinyLlmam_1.1v-awq-quantized"
        # When using a pre-quantized model, we should not apply BFP on top.
        apply_bfp = False
    elif apply_bfp:
        print(f"Applying BFP quantization with block_size={bfp_block_size} and mantissa_bits={bfp_mantissa_bits}")
        model_path = "TinyLlama/TinyLlama_v1.1" # BFP is applied on the original model
    else:
        print("Running without BFP quantization.")
        model_path = "TinyLlama/TinyLlama_v1.1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load HellaSwag dataset
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    
    # Limit dataset size if max_samples is specified
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Using {len(dataset)} samples for quick testing")
    
    dataset_sample = dataset
    dataset_sample = process_docs(dataset_sample)

    # Function to format HellaSwag questions for TinyLlama
    def format_question(example):
        context = example["query"]
        endings = example["choices"]
        
        return {
            "context": context,
            "choices": endings,
            "label": example["gold"]}
    formatted_dataset = dataset_sample.map(format_question)


    # DataLoader for batching
    batch_size = 1 # Process one question at a time for simplicity
    data_loader = DataLoader(formatted_dataset, batch_size=batch_size)

    correct_predictions_acc = 0
    correct_predictions_acc_norm = 0
    total_questions = 0
    
    # use my model for evaluation
    # Build kwargs for LlamaMyModel based on backend
    model_kwargs = {
        "model_name": model_path,
        "device": device,
        "stop_criteria": StopOnTokens(),
        "dtype": dtype,
        "apply_bfp": apply_bfp,
        "bfp_block_size": bfp_block_size,
        "bfp_mantissa_bits": bfp_mantissa_bits,
        "backend": args.backend
    }
    
    # Add precision parameters for clone backend
    if args.backend == "clone":
        if compute_dtype is not None:
            model_kwargs["clone_compute_dtype"] = compute_dtype
        if rope_cache_dtype is not None:
            model_kwargs["rope_cache_dtype"] = rope_cache_dtype
        model_kwargs["softmax_fp32"] = args.softmax_fp32
    
    # Add precision policy for custom backend
    if args.backend == "custom" and args.precision_policy is not None:
        model_kwargs["precision_policy"] = args.precision_policy
    
    my_model = LlamaMyModel(**model_kwargs)


    for batch in tqdm(data_loader, desc="Evaluating HellaSwag"):
        contexts = batch["context"]
        choices_list = batch["choices"] # This will be a list of lists of strings
        true_labels = batch["label"]

        # Get ground truth logits from the original model
        with torch.no_grad():
            for i in range(len(contexts)): # Iterate through batch (batch_size is 1 here)
                context = contexts[i]
                choices = [c for c in choices_list] # Flatten choices
                true_label = true_labels[i].item()

                # Calculate log-likelihood for each choice
                choice_log_likelihoods_acc = []
                choice_log_likelihoods_acc_norm = []
                for choice_text in choices:
                    full_text = context + " " + choice_text[0]
                    if full_text[-1] == ".":
                        full_text = full_text[:-1]
                    
                    # Tokenize the full text (keep the BatchEncoding for char->token mapping)
                    encoding = tokenizer(full_text, return_tensors="pt", truncation=True, add_special_tokens=False)
                    # Move tensor items to device but keep integer dtypes (input_ids/attention_mask should remain long)
                    tensor_inputs = {k: v.to(device) for k, v in encoding.items()}

                    # Log input ids (from tensor_inputs) and keep a reference to input_ids tensor
                    with open("mine.txt", "a") as f:
                        f.write(f"{tensor_inputs['input_ids']}\n")
                    input_ids = tensor_inputs['input_ids']

                    # Get model outputs
                    # pass the tensor inputs (moved to device) into the model
                    outputs = my_model.single_step(tensor_inputs)
                    my_model.reset_kv_cache()
                    
                    t_values, t_indices = outputs[0].max(dim=1)  # Get both values and indices
                    t_values = t_values.squeeze()
                    arr = t_values.cpu().to(torch.float32).numpy()  # Use the values
                    with open('mine.txt', 'a') as f:      # 'a' = append text mode
                        f.write('\n### tensor shape: {}\n'.format(arr.shape))  # optional separator/header
                        np.savetxt(f, arr, fmt='%.6f', delimiter=' ')
                        
                    
                    # Calculate the negative log-likelihood (NLL)
                    # Shift logits and labels for language modeling
                    shift_logits = outputs[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()
                    
                    # Calculate loss for each token
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    context_len_chars = len(context)
                    choice_start_char_idx = context_len_chars + 1 # account for the space

                    # Find the token index corresponding to the start of the choice.
                    # `tokenized_input` has a batch size of 1, so we use index 0.
                    # Use the original encoding (BatchEncoding) for char->token mapping
                    continuation_start_token_idx = encoding.char_to_token(0, choice_start_char_idx)

                    # The loss tensor is shifted by one from the input_ids. loss[k] is for token k+1.
                    # We want the loss for tokens from continuation_start_token_idx onwards.
                    # So, the slice starts at continuation_start_token_idx - 1.
                    if continuation_start_token_idx is not None:
                        continuation_loss = loss[continuation_start_token_idx - 1:]
                    else:
                        # This case means the choice is empty or doesn't produce tokens.
                        continuation_loss = torch.tensor([], device=loss.device)

                    # Score for 'acc' (Unnormalized Log-Likelihood)
                    if continuation_loss.numel() > 0:
                        sum_loss = continuation_loss.sum().item()
                    else:
                        sum_loss = float('inf')
                    
                    # Calculate the byte length of the choice text for normalization
                    choice_text_str = choice_text[0]
                    choice_byte_length = len(choice_text_str.encode('utf-8'))

                    # Score for 'acc_norm' (Byte-Normalized Log-Likelihood)
                    norm_loss = sum_loss / choice_byte_length if choice_byte_length > 0 else float('inf')
                    
                    # Append Log-Likelihoods (Log-Likelihood = -NLL)
                    choice_log_likelihoods_acc.append(-sum_loss)
                    choice_log_likelihoods_acc_norm.append(-norm_loss)
                    
                # Predict the choice with the highest log-likelihood
                predicted_choice_idx_acc = np.argmax(choice_log_likelihoods_acc)
                predicted_choice_idx_acc_norm = np.argmax(choice_log_likelihoods_acc_norm)
                predicted_answer_char_acc = chr(65 + predicted_choice_idx_acc)
                predicted_answer_char_acc_norm = chr(65 + predicted_choice_idx_acc_norm)
                
                true_answer_char = chr(65 + true_label)

                if predicted_choice_idx_acc == true_label:
                    correct_predictions_acc += 1
                if predicted_choice_idx_acc_norm == true_label:
                    correct_predictions_acc_norm += 1
                total_questions += 1
                
                if total_questions <= 5: # Print first 5 examples
                    print(f"\nContext:\n{context}")
                    for idx, ch in enumerate(choices):
                        print(f"{chr(65+idx)}. {ch} (Log-likelihood: {choice_log_likelihoods_acc[idx]:.2f} (acc) {choice_log_likelihoods_acc_norm[idx]:.2f} (acc_norm))")
                    print(f"True Answer: {true_answer_char}")
                    print(f"Predicted Answer (acc): {predicted_answer_char_acc}")
                    print(f"Predicted Answer (acc_norm): {predicted_answer_char_acc_norm}")
                    print("-" * 30)

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