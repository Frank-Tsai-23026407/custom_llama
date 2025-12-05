import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
from tqdm import tqdm
import numpy as np
from maddness_hadamard import MaddnessHadamardLayer
import copy
import sys
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
                activations[name].append(input[0].detach().cpu())
        return hook
    
    hooks = []
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "model.layers" in name:
            target_modules.append((name, module))
            
    for name, module in target_modules:
        hooks.append(module.register_forward_hook(hook_fn(name)))
        
    model.eval()
    count = 0
    with torch.no_grad():
        for text in dataset:
            if count >= num_samples:
                break
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            if inputs.input_ids.shape[1] == 0: continue
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)
            count += 1
            
    for h in hooks: h.remove()
    
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
            if inputs.input_ids.shape[1] == 0: continue
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)
            count += 1
            
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def evaluate_hellaswag_wrapper(model, tokenizer, device="cuda"):
    print("Evaluating on HellaSwag...")
    args = HellaSwagArgs(max_samples=None) 
    
    def model_single_step(inputs):
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits
        
    device_obj = torch.device(device)
    acc, acc_norm, _, _, _ = task_script_hellaswag(args, tokenizer, model_single_step, device_obj)
    print(f"HellaSwag Acc: {acc:.4f}, Acc Norm: {acc_norm:.4f}")
    return acc, acc_norm

def main():
    model_path = "model/tinyllama/TinyLlama_1.1v"
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load dataset once
    print("Loading wikitext dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    dataset = [x["text"] for x in dataset if len(x["text"]) > 20]
    
    # MADDNESS Parameters
    subspace_dim = 512
    tree_depth = 4
    num_levels_list = [1, 2]
    
    results_file = "maddness_hadamard_evaluation_results_levels.txt"
    with open(results_file, "w") as f:
        f.write("Levels | Depth | Perplexity | HellaSwag Acc\n")
        f.write("---|---|---|---\n")
    
    for num_levels in num_levels_list:
        print(f"\n==========================================")
        print(f"Evaluating Num Levels: {num_levels}, Tree Depth: {tree_depth}")
        print(f"==========================================")
        
        # Reload model fresh for each configuration
        print("Reloading base model...")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        
        maddness_model_path = f"maddness_hadamard_tinyllama_{subspace_dim}_d{tree_depth}_l{num_levels}.pt"
        
        if os.path.exists(maddness_model_path):
            print(f"Loading MADDNESS model from {maddness_model_path}...")
            state_dict = torch.load(maddness_model_path)
            
            # Identify which layers are Maddness layers based on state_dict keys
            maddness_layers = set()
            for key in state_dict.keys():
                if "lookup_table" in key:
                    layer_name = ".".join(key.split(".")[:-1])
                    maddness_layers.add(layer_name)
            
            print(f"Found {len(maddness_layers)} Maddness layers in checkpoint.")
            
            for name, original_layer in tqdm(model.named_modules(), desc="Replacing Layers (Loading)"):
                 if name not in maddness_layers:
                     continue
                     
                 if not isinstance(original_layer, nn.Linear):
                     continue
    
                 maddness_layer = MaddnessHadamardLayer(original_layer, subspace_dim=subspace_dim, tree_depth=tree_depth, num_levels=num_levels)
                 maddness_layer.to(device=original_layer.weight.device, dtype=original_layer.weight.dtype)
                 
                 parent_name = ".".join(name.split(".")[:-1])
                 child_name = name.split(".")[-1]
                 if parent_name == "":
                     parent = model
                 else:
                     parent = model.get_submodule(parent_name)
                 setattr(parent, child_name, maddness_layer)
                 
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully.")
            
        else:
            calib_data = get_calibration_data(model, tokenizer, dataset[:200], num_samples=50)
            print("Training MADDNESS layers...")
            modules_to_replace = {}
            for name, module in model.named_modules():
                if name in calib_data:
                    modules_to_replace[name] = module
                    
            # Train all layers for valid evaluation
            # This might take a while (~10-15 mins per depth for training + eval)
            
            quantization_errors = []
            max_diffs = []
            
            for name, original_layer in tqdm(modules_to_replace.items(), desc="Replacing Layers"):
                X_calib = calib_data[name].to(torch.float32)
                if X_calib.shape[0] > 10000:
                    indices = torch.randperm(X_calib.shape[0])[:10000]
                    X_calib = X_calib[indices]
                    
                # Create and train MaddnessHadamardLayer
                # Create and configure the layer
                maddness_layer = MaddnessHadamardLayer(original_layer, subspace_dim=subspace_dim, tree_depth=tree_depth, num_levels=num_levels)
                maddness_layer.to(device=original_layer.weight.device, dtype=original_layer.weight.dtype)
                
                W = original_layer.weight.data.to(torch.float32)
                
                # Train and configure
                mse, max_diff = maddness_layer.train_and_configure(X_calib.to(model.device), W.to(model.device))
                
                quantization_errors.append(mse)
                max_diffs.append(max_diff)
                
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                if parent_name == "":
                    parent = model
                else:
                    parent = model.get_submodule(parent_name)
                setattr(parent, child_name, maddness_layer)
                
                del X_calib
                del W
                torch.cuda.empty_cache()
            
            print(f"Average MSE: {np.mean(quantization_errors):.6f}")
            print(f"Average Max Diff: {np.mean(max_diffs):.6f}")
            
            print(f"Saving model to {maddness_model_path}...")
            torch.save(model.state_dict(), maddness_model_path)
        
        # Evaluation
        print("Evaluating Perplexity...")
        ppl = evaluate_perplexity(model, tokenizer, dataset)
        print(f"Perplexity: {ppl:.2f}")
        
        print("Evaluating HellaSwag...")
        hellaswag_acc, _ = evaluate_hellaswag_wrapper(model, tokenizer)
        print(f"HellaSwag Accuracy: {hellaswag_acc:.4f}")
        
        with open(results_file, "a") as f:
            f.write(f"{num_levels} | {tree_depth} | {ppl:.2f} | {hellaswag_acc:.4f}\n")
            
    print(f"All evaluations complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()
