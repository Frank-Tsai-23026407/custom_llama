import argparse
import os
import sys
import torch
from transformers import AutoTokenizer

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_backend.llama_custom import CustomLlamaModel
from llama_backend.utils import StopOnTokens
import testbench.task_utils as TU

def main():
    parser = argparse.ArgumentParser(description="Lightweight Winogrande Evaluation Script")
    parser.add_argument("--model_path", type=str, default="TinyLlama/TinyLlama_v1.1", help="Path to HF model or quantized directory")
    parser.add_argument("--backend", type=str, default="custom", choices=["custom","huggingface","clone"], help="Execution backend")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation samples")
    parser.add_argument("--bft", action="store_true", help="Apply BFP on-the-fly")
    parser.add_argument("--m_bit", type=int, default=4, help="Mantissa bits")
    parser.add_argument("--b_size", type=int, default=16, help="Block size")
    
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading model: {args.model_path} on {device}")
    
    model, tokenizer, device, _ = TU.setup_model_and_tokenizer(
        model_path=args.model_path,
        backend=args.backend,
        apply_bfp=args.bft,
        bfp_block_size=args.b_size,
        bfp_mantissa_bits=args.m_bit,
        device=device
    )
    
    print("\nStarting evaluation...")
    TU.evaluate_winogrande(model, tokenizer, device, max_samples=args.max_samples)
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
