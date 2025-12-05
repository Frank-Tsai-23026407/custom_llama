# standard libraries
from datasets import load_dataset
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

import re
import datasets

import os
import sys
# Add parent directory to path for imports
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))
# Add testbench directory to path for task_utils
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), '../testbench'))))
# Add current directory to path
sys.path.insert(0, str(os.path.abspath(os.path.dirname(__file__))))

# my libraries
from plain_script import *
from utils import *
from llama_my import LlamaMyModel
import task_utils as TU


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

def get_model_path(mantissa_bits, block_size):
    base_dir = "model/tinyllama/TinyLlama_1.1v-comprehensive/awq_fix"
    return os.path.join(base_dir, f"block_{block_size}_mantissa_{mantissa_bits}")

def apply_weights(my_model, path):
    print(f"Loading weights from {path}...")
    try:
        # Load quantized model to CPU to save GPU memory
        quantized_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=my_model.dtype, device_map="cpu")
        
        # Copy weights
        # We need to copy: self_attn (q,k,v,o), mlp (gate,up,down), input_layernorm, post_attention_layernorm
        # And embeddings/head if they are quantized (usually they are not in awq_fix unless specified, but let's assume standard awq_fix structure)
        # Actually, awq_fix usually only quantizes linear layers in blocks.
        
        # Embeddings
        my_model.model.get_input_embeddings().weight.data = quantized_model.get_input_embeddings().weight.data.to(my_model.device)
        my_model.model.get_output_embeddings().weight.data = quantized_model.get_output_embeddings().weight.data.to(my_model.device)
        
        for layer_idx in range(my_model.num_layers):
            src_layer = quantized_model.model.layers[layer_idx]
            tgt_layer = my_model.model.model.layers[layer_idx]
            
            # Layer Norms
            tgt_layer.input_layernorm.weight.data = src_layer.input_layernorm.weight.data.to(my_model.device)
            tgt_layer.post_attention_layernorm.weight.data = src_layer.post_attention_layernorm.weight.data.to(my_model.device)
            
            # Attention
            tgt_layer.self_attn.q_proj.weight.data = src_layer.self_attn.q_proj.weight.data.to(my_model.device)
            tgt_layer.self_attn.k_proj.weight.data = src_layer.self_attn.k_proj.weight.data.to(my_model.device)
            tgt_layer.self_attn.v_proj.weight.data = src_layer.self_attn.v_proj.weight.data.to(my_model.device)
            tgt_layer.self_attn.o_proj.weight.data = src_layer.self_attn.o_proj.weight.data.to(my_model.device)
            
            # MLP
            tgt_layer.mlp.gate_proj.weight.data = src_layer.mlp.gate_proj.weight.data.to(my_model.device)
            tgt_layer.mlp.up_proj.weight.data = src_layer.mlp.up_proj.weight.data.to(my_model.device)
            tgt_layer.mlp.down_proj.weight.data = src_layer.mlp.down_proj.weight.data.to(my_model.device)
            
            # Update params dict in attention_blocks
            block_params = my_model.attention_blocks[layer_idx].params
            block_params['norm1_weight'] = tgt_layer.input_layernorm.weight
            block_params['norm2_weight'] = tgt_layer.post_attention_layernorm.weight
            block_params['wq'] = tgt_layer.self_attn.q_proj.weight
            block_params['wk'] = tgt_layer.self_attn.k_proj.weight
            block_params['wv'] = tgt_layer.self_attn.v_proj.weight
            block_params['wo'] = tgt_layer.self_attn.o_proj.weight
            block_params['w_gate'] = tgt_layer.mlp.gate_proj.weight
            block_params['w_up'] = tgt_layer.mlp.up_proj.weight
            block_params['w_down'] = tgt_layer.mlp.down_proj.weight
            
            # IMPORTANT: We must also ensure the BF16 weights are correctly set in the params dict
            # LlamaMyModel init captured the ORIGINAL weights of the base model as BF16.
            # If we load a quantized model, we are replacing the 'wq', 'wk' etc. with quantized weights.
            # The 'wq_bf16' etc. in params dict should remain as the original high precision weights.
            # Since we initialized LlamaMyModel with the base model (TinyLlama_v1.1), 
            # the 'wq_bf16' should already be the correct original weights.
            # We just need to make sure we don't overwrite them with quantized weights.
            # We are NOT updating 'wq_bf16' here, so it should be fine.
        
        del quantized_model
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error loading weights from {path}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on HellaSwag with dynamic mix precision.")
    parser.add_argument("--block_size", type=str, required=True, help="Block size for quantization (e.g., 1x128).")
    parser.add_argument("--mantissa_bits", type=int, required=True, help="Mantissa bits (e.g., 5).")
    parser.add_argument("--dynamic_mix_ratio", type=float, default=0.0, help="Ratio for dynamic mix precision.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate.")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bar.")
    
    args = parser.parse_args()

    # Base model path (using the standard TinyLlama)
    base_model_path = "TinyLlama/TinyLlama_v1.1"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize LlamaMyModel with base model
    # We initialize with apply_bfp=True to trigger the capture of bf16 weights and setting up of params dict
    # But we will overwrite the quantized weights with the actual AWQ weights later.
    # We pass dynamic_mix_ratio here.
    print(f"Initializing base model: {base_model_path}")
    my_model = LlamaMyModel(
        model_name=base_model_path,
        device=device,
        backend='custom',
        stop_criteria=StopOnTokens(),
        dtype=torch.float32, # Use float32 for stability
        apply_bfp=True, # This ensures params dict has wq_bf16 etc.
        dynamic_mix_ratio=args.dynamic_mix_ratio
    )
    
    # Load AWQ weights
    model_path = get_model_path(args.mantissa_bits, args.block_size)
    apply_weights(my_model, model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

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
    
    # Determine where to write tqdm output
    tqdm_file = sys.stderr
    if not args.disable_tqdm:
        try:
            tqdm_file = open('/dev/tty', 'w')
        except Exception:
            pass

    # Reset computation stats before evaluation
    my_model.reset_computation_stats()

    for batch in tqdm(data_loader, desc="Evaluating HellaSwag", disable=args.disable_tqdm, file=tqdm_file):
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
                    # Move input tensors to device
                    tensor_inputs = TU.tokenize_to_device(tokenizer, full_text, device, add_special_tokens=False)
                    input_ids = tensor_inputs['input_ids']

                    # Get model outputs
                    outputs = my_model.single_step(tensor_inputs)
                    my_model.reset_kv_cache()
                    
                    # Calculate the negative log-likelihood (NLL)
                    # Shift logits and labels for language modeling
                    shift_logits, shift_labels = TU.get_shifted(outputs, input_ids)
                    loss = TU.per_token_cross_entropy(shift_logits, shift_labels)
                    
                    context_len_chars = len(context)
                    choice_start_char_idx = context_len_chars + 1 # account for the space

                    # Find the token index corresponding to the start of the choice.
                    continuation_start_token_idx = encoding.char_to_token(0, choice_start_char_idx)

                    if continuation_start_token_idx is not None:
                        continuation_loss = loss[continuation_start_token_idx - 1:]
                    else:
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
                
                if predicted_choice_idx_acc == true_label:
                    correct_predictions_acc += 1
                if predicted_choice_idx_acc_norm == true_label:
                    correct_predictions_acc_norm += 1
                total_questions += 1

    accuracy_acc = correct_predictions_acc / total_questions
    accuracy_acc_norm = correct_predictions_acc_norm / total_questions
    
    stats = my_model.get_computation_stats()
    
    print(f"\n--- HellaSwag Evaluation Results ---")
    print(f"Block Size: {args.block_size}")
    print(f"Mantissa Bits: {args.mantissa_bits}")
    print(f"Dynamic Mix Ratio: {args.dynamic_mix_ratio}")
    print(f"Total questions: {total_questions}")
    print(f"Correct predictions (acc): {correct_predictions_acc}")
    print(f"Correct predictions (acc_norm): {correct_predictions_acc_norm}")
    print(f"Accuracy (acc): {accuracy_acc:.4f}")
    print(f"Accuracy (acc_norm): {accuracy_acc_norm:.4f}")
    print(f"--- Computation Analysis ---")
    print(f"Total Ops: {stats['total_ops']:.2e}")
    print(f"High Precision Ops: {stats['high_prec_ops']:.2e}")
    print(f"Low Precision Ops: {stats['low_prec_ops']:.2e}")
    print(f"High Precision Ratio: {stats['high_prec_ratio']:.4%}")


if __name__ == "__main__":
    main()
