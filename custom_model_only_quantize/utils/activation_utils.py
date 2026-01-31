import torch
import gc
from datasets import load_dataset
from tqdm import tqdm

def prepare_calibration_data(tokenizer, dataset_name="wikitext2", num_samples=128, seq_len=512):
    """
    Loads dataset, tokenizes, and packs into fixed-length sequences.
    
    Args:
        tokenizer: Model tokenizer
        dataset_name: "wikitext2", "c4", "ptb"
        num_samples: Number of packed sequences to generate
        seq_len: Length of each sequence
        
    Returns:
        list[torch.Tensor]: List of input tensors (1, seq_len)
    """
    print(f"Loading and preparing calibration data: {dataset_name}...")
    
    # 1. Load Data
    if dataset_name == "wikitext2" or dataset_name == "Salesforce/wikitext":
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        text_column = "text"
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        text_column = "text"
    elif dataset_name == "ptb":
        dataset = load_dataset("ptb_text_only", "penn_treebank", split="train")
        text_column = "sentence"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 2. Tokenize and Pack
    # We need enough text to fill num_samples * seq_len
    # Strategy: Tokenize stream until we have enough tokens
    
    all_tokens = []
    required_tokens = num_samples * seq_len
    
    print("Tokenizing and packing data...")
    iterator = iter(dataset)
    
    pbar = tqdm(total=required_tokens, desc="Collecting tokens")
    
    while len(all_tokens) < required_tokens:
        try:
            item = next(iterator)
            text = item[text_column]
            if not text.strip():
                continue
                
            # Tokenize
            tokens = tokenizer(text, add_special_tokens=False).input_ids
            all_tokens.extend(tokens)
            pbar.update(len(tokens))
            
        except StopIteration:
            print("Warning: Dataset exhausted before reaching requested token count.")
            break
            
    pbar.close()
    
    # 3. Chunk into sequences
    packed_samples = []
    tokens_tensor = torch.tensor(all_tokens)
    
    for i in range(0, len(tokens_tensor), seq_len):
        if len(packed_samples) >= num_samples:
            break
            
        chunk = tokens_tensor[i : i + seq_len]
        if len(chunk) == seq_len:
            packed_samples.append(chunk.unsqueeze(0)) # [1, seq_len]
            
    print(f"Prepared {len(packed_samples)} packed samples of length {seq_len}.")
    return packed_samples


def get_layer_activations(model, samples, layer_name, device="cuda"):
    """
    Captures activations for a specific SINGLE layer using pre-processed samples.
    Recommended to avoid OOM by processing layer-by-layer.
    
    Args:
        model: The model
        samples: List of input tensors
        layer_name: Name of module to hook
        device: Execution device
        
    Returns:
        torch.Tensor: Aggregated activations for the layer [N_tokens, Features]
    """
    return get_multiple_layers_activations(model, samples, [layer_name], device)[layer_name]

def get_multiple_layers_activations(model, samples, layer_names, device="cuda"):
    """
    Captures activations for multiple layers simultaneously in a single forward pass.
    Useful for capturing all linear layers within a single transformer block at once,
    reducing the total number of forward passes required.
    
    Args:
        model: The model
        samples: List of input tensors
        layer_names: List of module names to hook
        device: Execution device
        
    Returns:
        dict[str, torch.Tensor]: Dictionary of aggregated activations {layer_name: tensor}
    """
    # Initialize storage: {layer_name: [list of batch activs]}
    activations_storage = {name: [] for name in layer_names}
    
    # Helper to create hook for specific layer name
    def make_hook(name):
        def hook(model, input, output):
            # input is tuple (tensor, )
            # Move immediately to CPU
            activations_storage[name].append(input[0].detach().cpu())
        return hook

    # Register hooks
    handles = []
    registered_count = 0
    modules_dict = dict(model.named_modules())
    
    for name in layer_names:
        if name in modules_dict:
            module = modules_dict[name]
            handles.append(module.register_forward_hook(make_hook(name)))
            registered_count += 1
        else:
             print(f"Warning: Layer {name} not found in model.")
             
    if registered_count == 0:
        raise ValueError("No valid layers found to hook.")
    
    # Run Inference
    # Assumed model is in eval mode
    with torch.no_grad():
        for batch in samples:
            batch = batch.to(device)
            model(batch)
            
    # Cleanup hooks
    for h in handles:
        h.remove()
    
    # Concatenate and Process Results
    final_results = {}
    
    for name, acts_list in activations_storage.items():
        if not acts_list:
            final_results[name] = None
            continue
            
        try:
            # act in acts_list is [1, seq_len, features]
            # Cat dim=1 -> [1, total_tokens, features]
            # View -> [total_tokens, features]
            full_seq = torch.cat(acts_list, dim=1)
            final_results[name] = full_seq.view(-1, full_seq.shape[-1])
        except Exception as e:
            print(f"Error merging activations for {name}: {e}")
            final_results[name] = None
            
        # Clear list from memory
        activations_storage[name] = None
        
    del activations_storage
    gc.collect()
    
    return final_results
