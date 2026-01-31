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
    parser = argparse.ArgumentParser(description="Lightweight HellaSwag Evaluation Script")
    parser.add_argument("--model_path", type=str, default="TinyLlama/TinyLlama_v1.1", help="Path to HF model or quantized directory")
    parser.add_argument("--backend", type=str, default="huggingface", choices=["custom","huggingface","clone"], help="Execution backend")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit evaluation samples")
    
    # BFP arguments (if using custom backend with on-the-fly quantization)
    parser.add_argument("--bft", action="store_true", help="Apply BFP on-the-fly")
    parser.add_argument("--m_bit", type=int, default=4, help="Mantissa bits")
    parser.add_argument("--b_size", type=int, default=16, help="Block size")
    
    # Precision control for clone/custom backends
    parser.add_argument("--precision_policy", type=str, default=None, help="Precision policy for custom backend")
    
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading model: {args.model_path} on {device}")
    
    # Initialization for CustomLlamaModel
    model_kwargs = {
        "model_name": args.model_path,
        "device": device,
        "dtype": torch.bfloat16,
        "backend": args.backend,
        "stop_criteria": StopOnTokens(),
    }
    
    if args.bft:
        model_kwargs.update({
            "apply_bfp": True,
            "bfp_block_size": args.b_size,
            "bfp_mantissa_bits": args.m_bit
        })
    
    if args.precision_policy:
        model_kwargs["precision_policy"] = args.precision_policy

    # Initialize model and tokenizer
    model = CustomLlamaModel(**model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print("\nStarting evaluation...")
    TU.evaluate_hellaswag(model, tokenizer, device, max_samples=args.max_samples)
    print("Evaluation complete.")

if __name__ == "__main__":
    main()