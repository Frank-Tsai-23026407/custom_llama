import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from llama_backend.llama_my import LlamaMyModel

model_path = "/home/frank23026407/custom_llama/model/tinyllama/TinyLlama_1.1v-comprehensive/awq_fix/block_1x128_mantissa_3"

try:
    print(f"Loading LlamaMyModel from {model_path}")
    model = LlamaMyModel(
        model_name=model_path,
        device='cpu', # Use CPU for quick check
        backend='custom',
        dtype=torch.float32
    )
    print("Successfully loaded LlamaMyModel!")
except Exception as e:
    print(f"Error loading LlamaMyModel: {e}")
