import torch
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

# Add parent directory to path for imports to allow accessing llama_backend
# We do this here so any file importing utils gets the path set up
sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_backend.llama_custom import CustomLlamaModel
from custom_model_only_quantize.runtime_quantize import apply_runtime_quantization, QuantizationConfig

try:
    from maddness_hadamard_multi_level.maddness_hadamard import MaddnessHadamardLayer
    MADDNESS_AVAILABLE = True
except ImportError:
    MaddnessHadamardLayer = None
    MADDNESS_AVAILABLE = False
    print("Warning: maddness_hadamard_multi_level not found. Maddness backend will be disabled.")

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
    if not MADDNESS_AVAILABLE:
        raise ImportError("Maddness module is not available.")
        
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
             
         if isinstance(original_layer, nn.Linear):
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

    return model, tokenizer

def get_or_load_model(model_path, backend_type, quantization_config=None):
    """
    Helper to manage model loading/unloading.
    Ensures only one model is loaded at a time to save memory.
    """
    global CURRENT_LOADED_KEY, LOADED_MODELS
    
    # Include quantization config in the key so we reload if config changes
    config_repr = str(quantization_config) if quantization_config else "None"
    key = (model_path, backend_type, config_repr)
    
    if CURRENT_LOADED_KEY == key and key in LOADED_MODELS:
        return LOADED_MODELS[key]
    
    # Unload previous if exists
    if CURRENT_LOADED_KEY is not None and CURRENT_LOADED_KEY in LOADED_MODELS:
        print(f"Unloading {CURRENT_LOADED_KEY}")
        del LOADED_MODELS[CURRENT_LOADED_KEY]
        torch.cuda.empty_cache()
        CURRENT_LOADED_KEY = None

    print(f"Loading {model_path} for {backend_type} with config {config_repr}")
    
    if backend_type == "hf":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        # Check for calibration requirement
        if quantization_config and quantization_config.calibration_dataset and quantization_config.calibration_dataset != "Local File":
            if quantization_config.activations is None and quantization_config.calibration_data is None: 
                from activation_utils import prepare_calibration_data
                print(f"Running on-the-fly calibration preparation with {quantization_config.calibration_dataset}...")
                # Pack data (CPU)
                packed_data = prepare_calibration_data(tokenizer, quantization_config.calibration_dataset)
                quantization_config.calibration_data = packed_data

        # Apply runtime quantization if configured (for dryrun)
        if quantization_config:
            print(f"Applying runtime quantization: {quantization_config}")
            apply_runtime_quantization(
                model, 
                method=quantization_config.method,
                block_height=quantization_config.block_height,
                block_width=quantization_config.block_width,
                mantissa_bits=quantization_config.mantissa_bits,
                top_k=quantization_config.top_k,
                activations=quantization_config.activations,
                calibration_data=quantization_config.calibration_data # NEW
            )

    elif backend_type == "custom":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Note: Custom backend might handle loading differently, but if we need calibration we typically need the base model first
        # CustomLlamaModel applies quantization in __init__.
        # If we need calibration, we might need to intercept it there or here.
        # But CustomLlamaModel helper logic inside __init__ is simple.
        # Let's see llama_custom.py again. It applies quantization if config is passed.
        # So we probably need to handle calibration HERE before creating CustomLlamaModel if possible,
        # OR CustomLlamaModel needs to handle it.
        # Since CustomLlamaModel loads its own HF model inside, we can't easily calibrate "before" passing config unless we mod CustomLlamaModel.
        # Let's stick to HF backend support for this feature request primarily, or advise user it's for HF backend.
        # BUT, waiting... if CustomLlamaModel uses `apply_runtime_quantization`, we can just ensure `quantization_config.activations` is populated.
        # BUT CustomLlamaModel creates the model inside __init__. We don't have access to it before then.
        # So for CustomLlamaModel, on-the-fly calibration is harder without refactoring CustomLlamaModel to accept 'activations' or doing it inside.
        
        # Simplest fix for CustomLlamaModel:
        # 1. Instantiate CustomLlamaModel.
        # 2. It has self.model (the hf model) and self.tokenizer.
        # 3. WE DO NOT pass quantization_config to init.
        # 4. Instead we manually apply quantization AFTER init here in utils.py.
        
        model = CustomLlamaModel(
            model_name=model_path,
            device=device,
            backend='custom',
            stop_criteria=StopOnTokens(),
            dtype=torch.float32,
            quantization_config=None # Don't quantize yet
        )
        
        if quantization_config:
             # Handle Calibration
             if quantization_config.calibration_dataset and quantization_config.calibration_dataset != "Local File" and quantization_config.activations is None and quantization_config.calibration_data is None:
                  from activation_utils import prepare_calibration_data
                  print(f"Running on-the-fly calibration with {quantization_config.calibration_dataset} for Custom Model...")
                  # Use tokenizer from model
                  packed_data = prepare_calibration_data(model.tokenizer, quantization_config.calibration_dataset)
                  quantization_config.calibration_data = packed_data
            
             print(f"Applying runtime quantization to CustomLlamaModel: {quantization_config}")
             apply_runtime_quantization(
                model.model, # The inner HF model
                method=quantization_config.method,
                block_height=quantization_config.block_height,
                block_width=quantization_config.block_width,
                mantissa_bits=quantization_config.mantissa_bits,
                top_k=quantization_config.top_k,
                activations=quantization_config.activations,
                calibration_data=quantization_config.calibration_data
            )

    elif backend_type == "maddness_hadamard":
        model, tokenizer = load_maddness_model(model_path)
    else:
        raise ValueError(f"Unknown backend type for loading: {backend_type}")
        
    LOADED_MODELS[key] = (model, tokenizer)
    CURRENT_LOADED_KEY = key
    return model, tokenizer
