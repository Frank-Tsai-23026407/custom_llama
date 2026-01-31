import torch
import torch.nn as nn
from tqdm import trange
from argparse import Namespace
from transformers import PreTrainedModel, AutoTokenizer
from typing import Optional

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from custom_aqlm.utils.modelutils import get_layers, get_model_head_with_norm, get_lm_logits
from custom_aqlm.inference.inference_utils import get_inps, update_outs

@torch.no_grad()
def perplexity_eval(model: PreTrainedModel, testenc: torch.LongTensor, args: Namespace) -> float:
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")

    nsamples = testenc.numel() // args.model_seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    device = args.devices[0]
    inps, forward_args = get_inps(model, testenc, args.model_seqlen, device, args.offload_activations)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
    
    for k, v in forward_args.items():
        forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    layers = get_layers(model)
    for i in trange(len(layers), desc="processing eval data by layer"):
        layer = layers[i].to(device)
        assert len(inps) == len(outs) == 1
        update_outs(layer, inps[0], outs[0], compute_mse=False, **forward_args)
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    get_model_head_with_norm(model).to(device)
    testenc = testenc.to(device)
    nsamples_per_device = len(inps[0])
    
    nlls = []
    for i in range(nsamples):
        inp = inps[0][i].to(device, non_blocking=True)
        lm_logits = get_lm_logits(inp.to(device), model)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * args.model_seqlen) : ((i + 1) * args.model_seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * args.model_seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.model_seqlen)).item()
    print(f"\n{args.dataset_name} perplexity = {ppl:.4f}\n")

    get_model_head_with_norm(model).to(torch.device("cpu"))

    if args.wandb and wandb is not None:
        wandb.log({args.dataset_name: ppl})

    model.config.use_cache = use_cache
    return ppl

def evaluate_hellaswag(model: PreTrainedModel, tokenizer: AutoTokenizer, args: Namespace):
    print("\n============ Evaluating HellaSwag... ============")
    try:
        from testbench.task_script_hellaswag import task_script_hellaswag
        
        device = args.devices[0]
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Wrapper for single step inference
        def model_single_step(inputs):
                # Move inputs to device
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(device)
                with torch.no_grad():
                    return model(**inputs).logits
        
        # Arguments wrapper for Hellaswag script
        class HellaSwagArgs:
            max_samples = args.hellaswag_max_samples
        
        acc, acc_norm, _, _, _ = task_script_hellaswag(
            HellaSwagArgs(), 
            tokenizer, 
            model_single_step, 
            device, 
            print_first_5_examples=False
        )
        print(f"HellaSwag Result: Acc={acc:.4f}, Acc Norm={acc_norm:.4f}")
        if args.wandb and wandb is not None:
            wandb.log({"hellaswag_acc": acc, "hellaswag_acc_norm": acc_norm})
            
    except Exception as e:
        print(f"Failed to run HellaSwag evaluation: {e}")
        import traceback
        traceback.print_exc()
