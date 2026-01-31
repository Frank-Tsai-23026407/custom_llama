# python3 aqlm/evaluate_quantized_model.py --model_path aqlm/output/tinyllama-2/tinyllama_quantized.pt
import argparse
import os
import sys
import torch

# Add project root and testbench to path
# __file__ is custom_model_quantize_inference/custom_aqlm/inference/evaluate_quantized_model.py
submodule_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # custom_model_quantize_inference
project_root = os.path.dirname(submodule_root) # custom_llama_clean

if submodule_root not in sys.path:
    sys.path.append(submodule_root)
if project_root not in sys.path:
    sys.path.append(project_root)

from transformers import AutoTokenizer
from llama_backend.llama_aqlm import LlamaAQLM
from custom_aqlm.quantize.datautils import get_wikitext2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the quantized model (.pt file)")
    parser.add_argument("--model_seqlen", type=int, default=2048)
    parser.add_argument("--eval_hellaswag", action="store_true", help="Evaluate on HellaSwag")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"Loading custom LlamaAQLM from {args.model_path} for evaluation...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        return

    # Load Model
    llama_model = LlamaAQLM(args.model_path, device=args.device)
    
    # Load Tokenizer (usually same as original model)
    print("Loading tokenizer...")
    try:
        # Try to load from the same directory as model, then fall back to TinyLlama
        model_dir = os.path.dirname(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except:
        print("Could not find tokenizer in model directory, falling back to TinyLlama/TinyLlama_v1.1")
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    # Custom PPL Evaluation
    print("Evaluating WikiText2 Perplexity (Custom Backend)...")
    try:
        # Note: calling get_wikitext2 with eval_mode=True returns the test set tokens
        test_data = get_wikitext2(0, args.model_seqlen, tokenizer, eval_mode=True)
        # Ensure we have the input_ids tensor
        if hasattr(test_data, "input_ids"): 
            test_data = test_data.input_ids
        
        def evaluate_perplexity_custom(model, test_ids, seqlen, device):
            print(f"Test data shape: {test_ids.shape}")
            nsamples = test_ids.numel() // seqlen
            nlls = []
            loss_fct = torch.nn.CrossEntropyLoss()
            
            for i in range(nsamples):
                batch = test_ids[:, i*seqlen : (i+1)*seqlen].to(device)
                if batch.size(1) < 1: continue
                
                # Input to single_step is expected to be a dict or tensor. 
                # LlamaAQLM.single_step now handles dict, so we can pass dict to be safe or tensor.
                # Passing dict to match typical usage.
                inputs = {'input_ids': batch}
                logits = model.single_step(inputs)
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch[..., 1:].contiguous()
                
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
                print(f"Sample {i+1}/{nsamples} | Loss: {loss.item():.4f}", end='\r')
                
            print("")
            if not nlls: return float('inf')
            return torch.exp(torch.stack(nlls).mean())

        ppl = evaluate_perplexity_custom(llama_model, test_data, args.model_seqlen, device=args.device)
        print(f"WikiText2 PPL: {ppl:.4f}")
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        return
    except Exception as e:
        print(f"WikiText2 Evaluation Failed: {e}")
        import traceback
        traceback.print_exc()

    # Custom HellaSwag Evaluation
    if args.eval_hellaswag:
        print("\n--- Evaluating HellaSwag Accuracy ---")
        try:
            from testbench import task_utils as TU
            
            # Define a wrapper class to match the expected interface of task_utils
            class ModelWrapper:
                def __init__(self, model):
                    self.model = model
                def single_step(self, inputs):
                    return self.model.single_step(inputs)

            wrapped_model = ModelWrapper(llama_model)
            
            print("Running Hellaswag Accuracy evaluation...")
            acc, acc_norm = TU.evaluate_hellaswag(wrapped_model, tokenizer, args.device, max_samples=None)
            print(f"HellaSwag Acc: {acc:.4f}, Acc Norm: {acc_norm:.4f}")
        except Exception as e:
            import traceback
            print(f"HellaSwag evaluation failed: {e}")
            traceback.print_exc()
if __name__ == "__main__":
    main()
