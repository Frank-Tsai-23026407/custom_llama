"""Quick test to verify precision policy works and compare with HF."""
import torch
from transformers import AutoTokenizer

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# customed code
from lm_eval.models.huggingface import HFLM
from llama_backend.llama_my import LlamaMyModel

def test_precision_policies():
    model_path = "TinyLlama/TinyLlama_v1.1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    print("="*60)
    
    # Test text
    text = "Hello, how are you?"
    
    # Load HF reference model
    print("Loading HF model (bfloat16)...")
    hflm = HFLM(pretrained=model_path, device=device, dtype="bfloat16")
    tokenizer = hflm.tokenizer
    
    # Tokenize
    encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoding["input_ids"].to(device)
    print(f"Input: {text}")
    print(f"Tokens: {input_ids.shape}")
    
    # HF forward
    with torch.no_grad():
        hf_out = hflm.model(input_ids=input_ids)
        hf_logits = hf_out.logits
    
    print(f"\nHF logits shape: {hf_logits.shape}, dtype: {hf_logits.dtype}")
    print(f"HF logits [0, -1, :5]: {hf_logits[0, -1, :5].cpu().to(torch.float32).numpy()}")
    
    # Test 0: Exact HF backend path inside LlamaMyModel
    print("\n" + "="*60)
    print("Test 0: LlamaMyModel with backend='huggingface' (exact parity path)")
    my_model_hf_backend = LlamaMyModel(
        model_name=model_path,
        device=device,
        dtype=torch.bfloat16,
        precision_policy="default",
        backend="huggingface"
    )
    with torch.no_grad():
        tensor_inputs = {k: v.to(device) for k, v in encoding.items()}
        my_logits_hf_backend = my_model_hf_backend.single_step(tensor_inputs)
    diff_hf_backend = (hf_logits.cpu().to(torch.float32) - my_logits_hf_backend.cpu().to(torch.float32)).abs()
    print(f"Backend HF parity diff: max={diff_hf_backend.max():.6f}, mean={diff_hf_backend.mean():.6f}")

    # Test 0b: Clone backend path (HF-compatible clone implementation)
    print("\n" + "="*60)
    print("Test 0b: LlamaMyModel with backend='clone' (HF-clone path)")
    my_model_clone = LlamaMyModel(
        model_name=model_path,
        device=device,
        dtype=torch.bfloat16,
        precision_policy="default",
        backend="clone"
    )
    with torch.no_grad():
        my_logits_clone = my_model_clone.single_step(tensor_inputs)
    diff_clone = (hf_logits.cpu().to(torch.float32) - my_logits_clone.cpu().to(torch.float32)).abs()
    print(f"Clone backend diff: max={diff_clone.max():.6f}, mean={diff_clone.mean():.6f}")

    # Test 1: Default policy (bfloat16 weights, default compute)
    print("\n" + "="*60)
    print("Test 1: LlamaMyModel with default precision policy")
    my_model_default = LlamaMyModel(
        model_name=model_path,
        device=device,
        dtype=torch.bfloat16,
        precision_policy="default"
    )
    print(f"Policy: {my_model_default.precision_policy}")
    
    with torch.no_grad():
        tensor_inputs = {k: v.to(device) for k, v in encoding.items()}
        my_logits_default = my_model_default.single_step(tensor_inputs)
    
    print(f"My logits shape: {my_logits_default.shape}, dtype: {my_logits_default.dtype}")
    print(f"My logits [0, -1, :5]: {my_logits_default[0, -1, :5].cpu().to(torch.float32).numpy()}")
    
    diff_default = (hf_logits.cpu().to(torch.float32) - my_logits_default.cpu().to(torch.float32)).abs()
    print(f"Diff vs HF: max={diff_default.max():.6f}, mean={diff_default.mean():.6f}")
    
    # Test 2: match_hf policy (bfloat16 weights, float32 compute)
    print("\n" + "="*60)
    print("Test 2: LlamaMyModel with match_hf precision policy")
    my_model_hf = LlamaMyModel(
        model_name=model_path,
        device=device,
        dtype=torch.bfloat16,
        precision_policy="match_hf"
    )
    print(f"Policy: {my_model_hf.precision_policy}")
    
    my_model_hf.reset_kv_cache()
    with torch.no_grad():
        my_logits_hf = my_model_hf.single_step(tensor_inputs)
    
    print(f"My logits shape: {my_logits_hf.shape}, dtype: {my_logits_hf.dtype}")
    print(f"My logits [0, -1, :5]: {my_logits_hf[0, -1, :5].cpu().to(torch.float32).numpy()}")
    
    diff_hf = (hf_logits.cpu().to(torch.float32) - my_logits_hf.cpu().to(torch.float32)).abs()
    print(f"Diff vs HF: max={diff_hf.max():.6f}, mean={diff_hf.mean():.6f}")
    
    # Test 3: bf16 end-to-end policy
    print("\n" + "="*60)
    print("Test 3: LlamaMyModel with bf16 end-to-end policy")
    my_model_bf16 = LlamaMyModel(
        model_name=model_path,
        device=device,
        dtype=torch.bfloat16,
        precision_policy="bf16"
    )
    print(f"Policy: {my_model_bf16.precision_policy}")
    
    my_model_bf16.reset_kv_cache()
    with torch.no_grad():
        my_logits_bf16 = my_model_bf16.single_step(tensor_inputs)
    
    print(f"My logits shape: {my_logits_bf16.shape}, dtype: {my_logits_bf16.dtype}")
    print(f"My logits [0, -1, :5]: {my_logits_bf16[0, -1, :5].cpu().to(torch.float32).numpy()}")
    
    diff_bf16 = (hf_logits.cpu().to(torch.float32) - my_logits_bf16.cpu().to(torch.float32)).abs()
    print(f"Diff vs HF: max={diff_bf16.max():.6f}, mean={diff_bf16.mean():.6f}")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"  HF backend parity diff: max={diff_hf_backend.max():.6f}, mean={diff_hf_backend.mean():.6f}")
    print(f"  Clone backend diff: max={diff_clone.max():.6f}, mean={diff_clone.mean():.6f}")
    print(f"  Default policy diff: max={diff_default.max():.6f}, mean={diff_default.mean():.6f}")
    print(f"  match_hf policy diff: max={diff_hf.max():.6f}, mean={diff_hf.mean():.6f}")
    print(f"  bf16 policy diff: max={diff_bf16.max():.6f}, mean={diff_bf16.mean():.6f}")
    
    if diff_hf.max() < diff_default.max():
        print("\n✓ match_hf policy reduces divergence from HF!")
    else:
        print("\n⚠ match_hf policy did not reduce divergence (may need investigation)")

if __name__ == "__main__":
    test_precision_policies()
