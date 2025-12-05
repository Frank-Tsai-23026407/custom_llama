import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from maddness import MaddnessLayer, MaddnessTrainer
import copy
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import testbench
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add testbench directory to path so task_script_hellaswag can import task_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'testbench')))
from testbench.task_script_hellaswag import task_script_hellaswag

class HellaSwagArgs:
    def __init__(self, max_samples=None):
        self.max_samples = max_samples


def get_calibration_data(model, tokenizer, dataset, num_samples=128, seq_len=512):
    print("Collecting calibration data...")
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            if len(activations[name]) < num_samples:
                # input is a tuple (x, ...), we want x
                # Detach and move to CPU to save GPU memory
                activations[name].append(input[0].detach().cpu())
        return hook
    
    hooks = []
    # Target all Linear layers in the model
    # For TinyLlama, layers are usually in model.layers
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Filter for main layers
            if "model.layers" in name:
                target_modules.append((name, module))
                
    # Register hooks
    for name, module in target_modules:
        hooks.append(module.register_forward_hook(hook_fn(name)))
        
    # Run inference
    model.eval()
    count = 0
    with torch.no_grad():
        for text in dataset:
            if count >= num_samples:
                break
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            if inputs.input_ids.shape[1] == 0:
                continue
                
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)
            count += 1
            
    # Remove hooks
    for h in hooks:
        h.remove()
        
    # Concatenate activations
    # Each item in list is [1, seq_len, dim]
    # We want [Total_Tokens, dim]
    final_activations = {}
    for name, data in activations.items():
        final_activations[name] = torch.cat(data, dim=1).squeeze(0).view(-1, data[0].shape[-1])
        
    print(f"Collected activations for {len(final_activations)} layers.")
    return final_activations

def evaluate_perplexity(model, tokenizer, dataset, num_samples=50, seq_len=512):
    print("Evaluating perplexity...")
    model.eval()
    nlls = []
    count = 0
    
    with torch.no_grad():
        for text in tqdm(dataset, total=num_samples, desc="Eval"):
            if count >= num_samples:
                break
                
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            if inputs.input_ids.shape[1] == 0:
                continue
                
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            count += 1
            
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def evaluate_hellaswag_wrapper(model, tokenizer, device="cuda"):
    print("Evaluating on HellaSwag...")
    # Use a subset for speed if needed, but user asked for evaluation. 
    # Let's use 500 samples for now to be faster than full validation but meaningful.
    # Or just None for full.
    # args = HellaSwagArgs(max_samples=100) 
    args = HellaSwagArgs(max_samples=None) 
    
    def model_single_step(inputs):
        # inputs is dict of tensors on device
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits
        
    device_obj = torch.device(device)
    acc, acc_norm, _, _, _ = task_script_hellaswag(args, tokenizer, model_single_step, device_obj)
    print(f"HellaSwag Acc: {acc:.4f}, Acc Norm: {acc_norm:.4f}")
    return acc, acc_norm

def main():
    # Paths
    model_path = "/home/frank23026407/custom_llama/model/tinyllama/TinyLlama_1.1v"
    
    # Load Model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    
    # Load Dataset
    print("Loading wikitext dataset...")
    # Use wikitext-2-v1 for speed
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    # Filter empty lines
    dataset = [x["text"] for x in dataset if len(x["text"]) > 20]
    
    # Baseline Eval
    print("Evaluating baseline model...")
    baseline_ppl = evaluate_perplexity(model, tokenizer, dataset)
    print(f"Baseline Perplexity: {baseline_ppl:.2f}")
    
    # Baseline HellaSwag
    baseline_acc, baseline_acc_norm = evaluate_hellaswag_wrapper(model, tokenizer)
    
    # MADDNESS Parameters
    NUM_SUBSPACES = 4
    TREE_DEPTH = 10
    
    # MADDNESS Model Path
    maddness_model_path = f"maddness_tinyllama_{NUM_SUBSPACES}_{TREE_DEPTH}.pt"
    
    # Check if model exists
    if os.path.exists(maddness_model_path):
        print(f"Loading MADDNESS model from {maddness_model_path}...")
        # We need to replace layers first to match architecture
        # We assume all Linear layers in "model.layers" were replaced with default config
        # To be safe, we should save/load config too, but for now hardcode
        
        modules_to_replace = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "model.layers" in name:
                modules_to_replace[name] = module
                
        for name, original_layer in tqdm(modules_to_replace.items(), desc="Replacing Layers (Loading)"):
             maddness_layer = MaddnessLayer(original_layer, num_subspaces=NUM_SUBSPACES, tree_depth=TREE_DEPTH)
             maddness_layer.to(original_layer.weight.device)
             # Cast to correct dtype
             maddness_layer.lookup_table = maddness_layer.lookup_table.to(original_layer.weight.dtype)
             maddness_layer.split_thresholds = maddness_layer.split_thresholds.to(original_layer.weight.dtype)
             
             parent_name = ".".join(name.split(".")[:-1])
             child_name = name.split(".")[-1]
             if parent_name == "":
                 parent = model
             else:
                 parent = model.get_submodule(parent_name)
             setattr(parent, child_name, maddness_layer)
             
        # Load state dict
        model.load_state_dict(torch.load(maddness_model_path))
        print("Model loaded successfully.")
        
    else:
        # Calibration
        # Use a subset of dataset
        calib_data = get_calibration_data(model, tokenizer, dataset[:200], num_samples=50)
        
        # Train and Replace
        print("Training MADDNESS layers...")
        
        # We need to replace layers in-place.
        # It's tricky to iterate and replace. We can use named_modules but need to set attribute on parent.
        
        modules_to_replace = {}
        
        for name, module in model.named_modules():
            if name in calib_data:
                modules_to_replace[name] = module
                
        # For demonstration/testing speed, let's limit to first 5 layers if there are many
        # keys are sorted by default insertion order usually, but let's be safe
        all_keys = list(modules_to_replace.keys())
        # if len(all_keys) > 10:
        #     print(f"Limiting to first 2 layers for speed (out of {len(all_keys)})...")
        #     modules_to_replace = {k: modules_to_replace[k] for k in all_keys[:2]}
        
        quantization_errors = []
        
        for name, original_layer in tqdm(modules_to_replace.items(), desc="Replacing Layers"):
            # Train MADDNESS
            X_calib = calib_data[name].to(torch.float32) # Train in FP32
            # Limit calibration size if too big
            if X_calib.shape[0] > 10000:
                indices = torch.randperm(X_calib.shape[0])[:10000]
                X_calib = X_calib[indices]
                
            trainer = MaddnessTrainer(original_layer.in_features, num_subspaces=NUM_SUBSPACES, tree_depth=TREE_DEPTH)
            
            # Get weights
            W = original_layer.weight.data.to(torch.float32).t() # [in, out]
                
            split_indices, split_thresholds, lookup_table, _, mse = trainer.train(X_calib.to(model.device), W.to(model.device))
            
            quantization_errors.append(mse)
            
            # Create Layer
            maddness_layer = MaddnessLayer(original_layer, num_subspaces=NUM_SUBSPACES, tree_depth=TREE_DEPTH)
            maddness_layer.split_indices = split_indices
            maddness_layer.split_thresholds = split_thresholds
            maddness_layer.lookup_table = lookup_table
            
            # Move to device/dtype
            maddness_layer.to(original_layer.weight.device)
            # Note: MaddnessLayer implementation uses float by default. 
            # If model is fp16, we should cast lookup table.
            maddness_layer.lookup_table = maddness_layer.lookup_table.to(original_layer.weight.dtype)
            maddness_layer.split_thresholds = maddness_layer.split_thresholds.to(original_layer.weight.dtype)
            
            # Replace in model
            # name is like "model.layers.0.self_attn.q_proj"
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name == "":
                parent = model
            else:
                parent = model.get_submodule(parent_name)
                
            setattr(parent, child_name, maddness_layer)
            
            # Clean up memory
            del X_calib
            torch.cuda.empty_cache()
            
        print("Replacement complete.")
        
        if len(quantization_errors) > 0:
            avg_mse = sum(quantization_errors) / len(quantization_errors)
            print(f"Average Quantization Error (MSE): {avg_mse:.6f}")
        
        # Save Model
        print(f"Saving MADDNESS model to {maddness_model_path}...")
        torch.save(model.state_dict(), maddness_model_path)

    
    # Evaluate MADDNESS
    print("Evaluating MADDNESS model...")
    maddness_ppl = evaluate_perplexity(model, tokenizer, dataset)
    print(f"MADDNESS Perplexity: {maddness_ppl:.2f}")
    
    # MADDNESS HellaSwag
    maddness_acc, maddness_acc_norm = evaluate_hellaswag_wrapper(model, tokenizer)
    
    # Save results
    with open("maddness_results.txt", "w") as f:
        f.write(f"Baseline PPL: {baseline_ppl}\n")
        f.write(f"Baseline HellaSwag Acc: {baseline_acc}\n")
        f.write(f"Baseline HellaSwag Acc Norm: {baseline_acc_norm}\n")
        f.write(f"MADDNESS PPL: {maddness_ppl}\n")
        f.write(f"MADDNESS HellaSwag Acc: {maddness_acc}\n")
        f.write(f"MADDNESS HellaSwag Acc Norm: {maddness_acc_norm}\n")

if __name__ == "__main__":
    main()
