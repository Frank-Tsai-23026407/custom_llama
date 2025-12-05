
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add current directory to path so we can import llama_my
sys.path.insert(0, str(Path(__file__).parent))

from llama_my import LlamaMyModel
from llama_backend.utils import StopOnTokens

def test_dynamic_mix():
    print("Testing Dynamic Mix Precision...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32
    
    # Initialize model with dynamic mix precision
    # We enable apply_bfp to simulate fix-precision
    # And set dynamic_mix_ratio to something > 0
    my_model = LlamaMyModel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=device,
        backend='custom',
        stop_criteria=StopOnTokens(),
        dtype=dtype,
        apply_bfp=True,
        bfp_block_size=128,
        bfp_mantissa_bits=4,
        dynamic_mix_ratio=0.01 # 1% important channels
    )
    
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    input_text = "Hello, how are you?"
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    
    print("Running single step...")
    logits = my_model.single_step(inputs)
    print("Logits shape:", logits.shape)
    print("Logits (first 5):", logits[0, -1, :5])
    
    # Run generation
    print("Running generation...")
    outputs = my_model.generate(input_text, max_new_tokens=10)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:", generated_text)
    print("Test passed!")

if __name__ == "__main__":
    test_dynamic_mix()
