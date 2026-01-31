import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

# Add project root and custom_aqlm to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Set path to the downloaded model
model_path = "./custom_aqlm/model/TinyLlama-v1.0-hf-aqlm"
device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate():
    print(f"Loading HF AQLM model from {model_path}...")
    
    # 1. Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()

    # 2. PPL Evaluation (WikiText2)
    print("\n--- Evaluating WikiText2 Perplexity ---")
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
            if i % 10 == 0:
                print(f"Sample {i}/{nsamples}, current loss: {loss.item():.4f}")
                
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"\nWikiText2 PPL: {ppl.item():.4f}")

    # 3. HellaSwag Evaluation
    print("\n--- Evaluating HellaSwag Accuracy ---")
    try:
        # Get the absolute root of the project (one level up from current dir)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        actual_root = os.path.dirname(script_dir)
        if actual_root not in sys.path:
            sys.path.append(actual_root)
            
        print(f"Adding {actual_root} to sys.path for testbench import...")
        from testbench import task_utils as TU
        
        # Define a wrapper class to match the expected interface of task_utils
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
        
        print("Running Hellaswag Accuracy evaluation...")
        acc, acc_norm = TU.evaluate_hellaswag(wrapped_model, tokenizer, device, max_samples=None)
        print(f"HellaSwag Acc: {acc:.4f}, Acc Norm: {acc_norm:.4f}")
    except Exception as e:
        import traceback
        print(f"HellaSwag evaluation failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    evaluate()
