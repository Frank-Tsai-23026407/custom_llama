import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import copy
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import testbench and other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add testbench directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'testbench')))

from maddness_remove_outlier.maddness_outlier import MaddnessOutlierLayer, MaddnessOutlierTrainer
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
        if isinstance(module, nn.Linear):
            if "model.layers" in name:
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
            if inputs.input_ids.shape[1] == 0:
                continue
                
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            model(**inputs)
            count += 1
            
    for h in hooks:
        h.remove()
        
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
    # Paths
    model_path = "/home/frank23026407/custom_llama/model/tinyllama/TinyLlama_1.1v"
    
    # Load Model
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    
    # Load Dataset
    print("Loading wikitext dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    dataset = [x["text"] for x in dataset if len(x["text"]) > 20]
    
    # Baseline Eval
    print("Evaluating baseline model...")
    baseline_ppl = evaluate_perplexity(model, tokenizer, dataset)
    print(f"Baseline Perplexity: {baseline_ppl:.2f}")
    
    # MADDNESS Parameters
    NUM_SUBSPACES = 4
    TREE_DEPTH = 4
    OUTLIER_K = 32
    OUTLIER_PERCENT = 0.05
    
    # MADDNESS Model Path
    maddness_model_path = "maddness_remove_outlier/maddness_tinyllama_outlier.pt"
    
    if os.path.exists(maddness_model_path):
        print(f"Loading MADDNESS model from {maddness_model_path}...")
        
        modules_to_replace = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "model.layers" in name:
                modules_to_replace[name] = module
                
        for name, original_layer in tqdm(modules_to_replace.items(), desc="Replacing Layers (Loading)"):
             maddness_layer = MaddnessOutlierLayer(original_layer, num_subspaces=NUM_SUBSPACES, tree_depth=TREE_DEPTH, outlier_k=OUTLIER_K)
             maddness_layer.to(original_layer.weight.device)
             maddness_layer.lookup_table = maddness_layer.lookup_table.to(original_layer.weight.dtype)
             maddness_layer.split_thresholds = maddness_layer.split_thresholds.to(original_layer.weight.dtype)
             
             parent_name = ".".join(name.split(".")[:-1])
             child_name = name.split(".")[-1]
             if parent_name == "":
                 parent = model
             else:
                 parent = model.get_submodule(parent_name)
             setattr(parent, child_name, maddness_layer)
             
        model.load_state_dict(torch.load(maddness_model_path))
        print("Model loaded successfully.")
        
    else:
        # Calibration
        calib_data = get_calibration_data(model, tokenizer, dataset[:200], num_samples=50)
        
        # Train and Replace
        print("Training MADDNESS layers with Outlier Removal...")
        
        modules_to_replace = {}
        for name, module in model.named_modules():
            if name in calib_data:
                modules_to_replace[name] = module
        
        quantization_errors = []
        
        for name, original_layer in tqdm(modules_to_replace.items(), desc="Replacing Layers"):
            # Train MADDNESS
            X_calib = calib_data[name].to(torch.float32)
            if X_calib.shape[0] > 10000:
                indices = torch.randperm(X_calib.shape[0])[:10000]
                X_calib = X_calib[indices]
                
            trainer = MaddnessOutlierTrainer(original_layer.in_features, num_subspaces=NUM_SUBSPACES, tree_depth=TREE_DEPTH, outlier_percent=OUTLIER_PERCENT)
            
            W = original_layer.weight.data.to(torch.float32).t()
                
            split_indices, split_thresholds, lookup_table, mse = trainer.train(X_calib.to(model.device), W.to(model.device))
            
            quantization_errors.append(mse)
            
            # Create Layer
            maddness_layer = MaddnessOutlierLayer(original_layer, num_subspaces=NUM_SUBSPACES, tree_depth=TREE_DEPTH, outlier_k=OUTLIER_K)
            maddness_layer.split_indices = split_indices
            maddness_layer.split_thresholds = split_thresholds
            maddness_layer.lookup_table = lookup_table
            
            maddness_layer.to(original_layer.weight.device)
            maddness_layer.lookup_table = maddness_layer.lookup_table.to(original_layer.weight.dtype)
            maddness_layer.split_thresholds = maddness_layer.split_thresholds.to(original_layer.weight.dtype)
            
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name == "":
                parent = model
            else:
                parent = model.get_submodule(parent_name)
                
            setattr(parent, child_name, maddness_layer)
            
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
    print("Evaluating MADDNESS Outlier model...")
    maddness_ppl = evaluate_perplexity(model, tokenizer, dataset)
    print(f"MADDNESS Outlier Perplexity: {maddness_ppl:.2f}")
    
    # MADDNESS HellaSwag
    maddness_acc, maddness_acc_norm = evaluate_hellaswag_wrapper(model, tokenizer)
    
    # Save results
    with open("maddness_remove_outlier/results.txt", "w") as f:
        f.write(f"Baseline PPL: {baseline_ppl}\n")
        f.write(f"MADDNESS Outlier PPL: {maddness_ppl}\n")
        f.write(f"MADDNESS Outlier HellaSwag Acc: {maddness_acc}\n")
        f.write(f"MADDNESS Outlier HellaSwag Acc Norm: {maddness_acc_norm}\n")

if __name__ == "__main__":
    main()
