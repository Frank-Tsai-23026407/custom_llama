import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root and testbench to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
parent_root = os.path.dirname(project_root)
if parent_root not in sys.path:
    sys.path.append(parent_root)

def evaluate_baseline(model_id="TinyLlama/TinyLlama_v1.1", device="cuda"):
    print(f"Loading baseline model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map=device
    )
    model.eval()

    # 1. PPL Evaluation (WikiText2)
    print("\n--- Evaluating WikiText2 Perplexity (Baseline) ---")
    from custom_aqlm.quantize.datautils import get_wikitext2
    seqlen = 2048
    test_data = get_wikitext2(0, seqlen, tokenizer, eval_mode=True)
    if hasattr(test_data, "input_ids"): test_data = test_data.input_ids
    
    test_ids = test_data.to(device)
    nsamples = test_ids.numel() // seqlen
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i in range(nsamples):
            batch = test_ids[:, i*seqlen : (i+1)*seqlen]
            outputs = model(batch)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch[..., 1:].contiguous()
            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss)
            if i % 20 == 0:
                print(f"Sample {i}/{nsamples}, loss: {loss.item():.4f}")
                
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Baseline WikiText2 PPL: {ppl.item():.4f}")

    # 2. HellaSwag Evaluation
    print("\n--- Evaluating HellaSwag Accuracy (Baseline) ---")
    try:
        from testbench import task_utils as TU
        
        class ModelWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device
            def single_step(self, inputs):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.no_grad():
                    return self.model(**inputs).logits

        wrapped_model = ModelWrapper(model, device)
        acc, acc_norm = TU.evaluate_hellaswag(wrapped_model, tokenizer, device, max_samples=None)
        print(f"Baseline HellaSwag Acc: {acc:.4f}, Acc Norm: {acc_norm:.4f}")
    except Exception as e:
        print(f"HellaSwag evaluation failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama_v1.1")
    args = parser.parse_args()
    
    evaluate_baseline(args.model_id)
