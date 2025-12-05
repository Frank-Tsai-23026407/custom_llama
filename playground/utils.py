import torch
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

# Add parent directory to path for imports to allow accessing llama_backend
# We do this here so any file importing utils gets the path set up
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_backend.llama_my import LlamaMyModel
from maddness_hadamard_multi_level.maddness_hadamard import MaddnessHadamardLayer
import torch.nn as nn
import re
import os
from tqdm import tqdm

# --- Global State ---
GENERATION_FUNCTIONS = {}
LOADED_MODELS = {} # (model_path, backend_type) -> (model, tokenizer)
CURRENT_LOADED_KEY = None # (model_path, backend_type)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Registry Decorator ---
def register_generation_function(name):
    """Decorator to register a generation function."""
    def decorator(func):
        GENERATION_FUNCTIONS[name] = func
        return func
    return decorator

# --- Helper Classes ---
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# --- Model Management ---
def load_maddness_model(checkpoint_path):
    """
    Loads a Maddness-Hadamard quantized model.
    Expects checkpoint_path to follow naming convention:
    ..._{subspace_dim}_d{tree_depth}_l{num_levels}.pt
    """
    print(f"Loading Maddness model from {checkpoint_path}")
    
    # Parse config from filename
    # Example: maddness_hadamard_tinyllama_512_d4_l2.pt
    try:
        match = re.search(r'_(\d+)_d(\d+)_l(\d+)\.pt$', checkpoint_path)
        if match:
            subspace_dim = int(match.group(1))
            tree_depth = int(match.group(2))
            num_levels = int(match.group(3))
        else:
            # Fallback for old naming convention or manual override
            print("Warning: Could not parse config from filename. Using defaults.")
            subspace_dim = 512
            tree_depth = 4
            num_levels = 1
            
        print(f"Config: subspace_dim={subspace_dim}, tree_depth={tree_depth}, num_levels={num_levels}")
        
    except Exception as e:
        print(f"Error parsing filename: {e}")
        raise ValueError("Invalid filename format for Maddness model")

    # Load Base Model
    base_model_path = "model/tinyllama/TinyLlama_1.1v" # Hardcoded for now as per user context
    print(f"Loading base model: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load State Dict
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path)
    
    # Identify Maddness layers
    maddness_layers = set()
    for key in state_dict.keys():
        if "lookup_table" in key:
            layer_name = ".".join(key.split(".")[:-1])
            maddness_layers.add(layer_name)
            
    print(f"Found {len(maddness_layers)} Maddness layers to replace.")
    
    # Replace Layers
    for name, original_layer in tqdm(model.named_modules(), desc="Replacing Layers"):
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
         
    # Load Weights
    model.load_state_dict(state_dict, strict=False)
    print("Maddness model loaded successfully.")
    
    return model, tokenizer

def get_or_load_model(model_path, backend_type):
    """
    Helper to manage model loading/unloading.
    Ensures only one model is loaded at a time to save memory.
    """
    global CURRENT_LOADED_KEY, LOADED_MODELS
    
    key = (model_path, backend_type)
    
    if CURRENT_LOADED_KEY == key and key in LOADED_MODELS:
        return LOADED_MODELS[key]
    
    # Unload previous if exists
    if CURRENT_LOADED_KEY is not None and CURRENT_LOADED_KEY in LOADED_MODELS:
        print(f"Unloading {CURRENT_LOADED_KEY}")
        del LOADED_MODELS[CURRENT_LOADED_KEY]
        torch.cuda.empty_cache()
        CURRENT_LOADED_KEY = None

    print(f"Loading {model_path} for {backend_type}")
    
    if backend_type == "hf":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    elif backend_type == "custom":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LlamaMyModel(
            model_name=model_path,
            device=device,
            backend='custom',
            stop_criteria=StopOnTokens(),
            dtype=torch.float32
        )
    elif backend_type == "maddness_hadamard":
        model, tokenizer = load_maddness_model(model_path)
    else:
        raise ValueError(f"Unknown backend type for loading: {backend_type}")
        
    LOADED_MODELS[key] = (model, tokenizer)
    CURRENT_LOADED_KEY = key
    return model, tokenizer
