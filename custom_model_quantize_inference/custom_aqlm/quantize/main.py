import os
import sys
import json
import torch
import torch.nn as nn
from argparse import Namespace
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root and testbench to path to enable importing testbench scripts
# The actual root is 4 levels up from this file: main.py -> quantize -> aqlm -> custom_model_quantize_inference -> custom_llama_clean
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Also add the submodule root for internal imports
submodule_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if submodule_root not in sys.path:
    sys.path.append(submodule_root)

testbench_dir = os.path.join(project_root, 'testbench')
if testbench_dir not in sys.path:
    sys.path.append(testbench_dir)

try:
    import wandb
    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False
    
from custom_aqlm.utils.layer import AqlmLayer
from custom_aqlm.quantize.datautils import get_loaders
from custom_aqlm.utils.modelutils import find_sublayers

def get_calibration_inputs(model, dataloader, seqlen, device):
    """
    Capture inputs to the first transformer block including kwargs.
    """
    inps = []
    all_kwargs = []  # Store kwargs for each sample
    
    # Create a hook module to capture inputs
    # Captured inputs: inp, ['attention_mask', 'position_ids', 'past_key_value', 'output_attentions', 'use_cache', 'cache_position'] (kwargs)
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, inp, **kwargs):
            inps.append(inp)
            all_kwargs.append(kwargs)  # Save kwargs for each sample
            raise ValueError("Stop Forward")
    
    # Get the first attention block inputs
    layers = model.model.layers
    original_layer0 = layers[0]
    layers[0] = Catcher(original_layer0)
    
    print(f"Calibrating with {len(dataloader)} samples...")
    
    for batch in dataloader:
        batch = batch.to(device)
        try:
            # perform the whole forward pass to capture inputs and kwargs
            model(batch)
            # Or ONLY perform embedding + layer[0] to save time
            # hidden_states = model.model.embed_tokens(batch)
            # model.model.layers[0](hidden_states)
        except ValueError as e:
            if "Stop Forward" in str(e):
                pass
            else:
                raise e
        
        # dummy break condition in case dataloader is infinite
        if len(inps) >= len(dataloader): 
            break
            
    layers[0] = original_layer0
    
    res = [i.detach().cpu() for i in inps]
    
    return res, all_kwargs

def main():
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name [c4, pajama] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )

    parser.add_argument("--save", type=str, default=None, help="Path to save quantized model (should be a .pt file now).")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda:0)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    
    # Quantization Config
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=2,
        help="#Number of codebooks per layer",
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=8,
        help="each codebook will contain 2 ** nbits_per_codebook vectors",
    )
    parser.add_argument(
        "--codebook_value_nbits",
        type=int,
        default=16,
        help="If below 16, quantize the values in each codebook with the specified number of bits",
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=8,
        help="How many input features are quantized together",
    )
    parser.add_argument(
        "--out_group_size",
        type=int,
        default=1,
        help="How many output units are quantized together",
    )
    parser.add_argument(
         "--scale_nbits",
         type=int,
         default=0,
         help="Number of bits for scale quantization (0 to disable).",
    )
    parser.add_argument(
        "--use_bfp",
        action="store_true",
        help="Use BFP for codebooks (Block Floating Point compression).",
    )

    # Optimization Config
    parser.add_argument(
        "--init_max_iter",
        type=int,
        default=100,
        help="Number of K-Means iterations used to initialize codebooks and codes",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Keep top-(this_many) best candidates for each codebook when finding optimal codes",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Maximum number of beam search rounds. (Reduced default to 10 for reasonable runtime)",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="Run (this many) Adam updates before every beam search round",
    )
    parser.add_argument(
        "--relative_mse_tolerance",
        type=float,
        default=None,
        help="Stop training when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)",
    )
    parser.add_argument(
        "--accum_dtype",
        type=str,
        default="float64",
        choices=["float64", "float32", "float16"],
        help="Dtype for XTX accumulation (float64 recommended for precision). switch to float32 if memory is extremely tight but results may degrade.",
    )
    
    # Evaluation Config
    parser.add_argument(
        "--run_calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run calibration and quantization.",
    )
    parser.add_argument(
        "--eval_ppl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to evaluate perplexity.",
    )
    parser.add_argument(
        "--eval_hellaswag",
        action="store_true",
        help="Whether to evaluate on HellaSwag task.",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Whether to skip evaluation",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to log to wandb",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code.",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Whether to use fast tokenizer (some models have only fast tokenizer).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print detailed logging during quantization.",
    )

    torch.set_num_threads(min(16, torch.get_num_threads()))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    args = parser.parse_args()
    
    # Resolve Model Path
    if os.path.exists(args.model_path):
        args.model_path = os.path.abspath(args.model_path)
    else:
        print(f"Local path {args.model_path} not found. Using as HF path/ID.")
    
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)

    if args.run_calibration:
        print(f"Loading calibration data ({args.dataset})...")
        dataloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=args.model_seqlen, model_path=args.model_path, use_fast_tokenizer=args.use_fast_tokenizer, trust_remote_code=args.trust_remote_code)
        
        # Process dataloader into list of tensors
        calib_data = []
        if isinstance(dataloader, list):
            calib_data = dataloader
        elif isinstance(dataloader, torch.Tensor):
            for i in range(0, dataloader.shape[1], args.model_seqlen):
                if i + args.model_seqlen <= dataloader.shape[1]:
                    calib_data.append(dataloader[:, i:i+args.model_seqlen])
        
        if args.nsamples:
            calib_data = calib_data[:args.nsamples]
        print(f"Using {len(calib_data)} calibration samples.")
        
        print("Collecting inputs for Layer 0...")
        device = torch.device("cuda")
        inps, all_kwargs = get_calibration_inputs(model, calib_data, args.model_seqlen, device)
        
        layers = model.model.layers
        
        # Log captured kwargs from first sample
        if all_kwargs:
            print(f"Captured kwargs: {list(all_kwargs[0].keys())}")
        
        if args.wandb:
            assert has_wandb, "`wandb` not installed, try pip install `wandb`"
            wandb.init(
                config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
            )

        # --- Checkpoint / Resume Logic ---
        checkpoint_dir = None
        if args.save:
            checkpoint_dir = os.path.join(os.path.dirname(args.save) if os.path.dirname(args.save) else ".", f"checkpoints_{os.path.basename(args.save)}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Checkpoints will be saved to: {checkpoint_dir}")

        start_layer = 0
        if checkpoint_dir:
            # Find the latest checkpoint
            existing_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_layer_") and f.endswith(".pt")]
            if existing_checkpoints:
                layer_indices = [int(f.split("_")[-1].split(".")[0]) for f in existing_checkpoints]
                latest_layer = max(layer_indices)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_layer_{latest_layer}.pt")
                print(f"Found checkpoint for layer {latest_layer}. Resuming from layer {latest_layer + 1}...")
                
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
                model = checkpoint_data["model"].to(device)
                inps = checkpoint_data["inps"]
                start_layer = latest_layer + 1
                layers = model.model.layers # Re-reference layers from the loaded model
                
                # Cleanup old unused tensors from checkpoint data
                del checkpoint_data
                torch.cuda.empty_cache()

        print("Starting Layer-wise Quantization...")
        for i in range(start_layer, len(layers)):
            layer = layers[i]
            print(f"Quantizing Layer {i}/{len(layers)}...")
            layer = layer.to(device)
            
            # 1. Identify sublayers (Linear modules) to quantize
            subset = find_sublayers(layer)
            subset_inputs = {name: [] for name in subset}
            
            # 2. Register hooks to capture inputs for each sublayer
            def get_hook(name):
                def hook(module, inp, out):
                    subset_inputs[name].append(inp[0].detach().cpu())
                return hook
                
            handles = []
            for name, sub in subset.items():
                handles.append(sub.register_forward_hook(get_hook(name)))
                
            # 3. Pass data through the layer to trigger hooks
            new_inps = []
            for j in range(len(inps)):
                inp = inps[j].to(device)
                
                # Use the kwargs captured for this specific sample
                sample_kwargs = all_kwargs[j] if j < len(all_kwargs) else {}
                
                # Extract and move position_ids to device
                position_ids = sample_kwargs.get("position_ids", None)
                if position_ids is None:
                    # Fallback: generate position_ids based on sequence length
                    position_ids = torch.arange(inp.shape[1], device=device).unsqueeze(0)
                elif isinstance(position_ids, torch.Tensor):
                    position_ids = position_ids.to(device)
                
                # Extract attention_mask and move to device
                attention_mask = sample_kwargs.get("attention_mask", None)
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.to(device)
                
                kwargs = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
                # RoPE Fix: Do not manually pass position_embeddings arguments
                
                with torch.no_grad():
                    try:
                        layer_out = layer(inp, **kwargs)[0]
                    except Exception as e:
                        print(f"DEBUG: Layer forward failed: {e}")
                        raise e
                
                new_inps.append(layer_out.cpu())
                
            for h in handles: h.remove()
            
            # 4. Quantize each sublayer using the captured inputs
            # 4. Quantize each sublayer using the captured inputs
            for name, sub in subset.items():
                # Keep on CPU: Do not move to device yet
                all_x = torch.cat(subset_inputs[name], dim=1).squeeze(0)
                print(f"  Quantizing {name} (Shape: {all_x.shape})...")
                
                aqlm_layer = AqlmLayer(
                    sub, 
                    num_codebooks=args.num_codebooks, 
                    nbits_per_codebook=args.nbits_per_codebook, 
                    in_group_size=args.in_group_size,
                    out_group_size=args.out_group_size,
                    scale_nbits=args.scale_nbits,
                    codebook_value_nbits=args.codebook_value_nbits,
                    use_bfp=args.use_bfp, 
                    init_max_iter=args.init_max_iter,
                    max_epochs=args.max_epochs,
                    lr=args.lr,
                    beam_size=args.beam_size,
                    steps_per_epoch=args.steps_per_epoch,
                    relative_mse_tolerance=args.relative_mse_tolerance,
                    accum_dtype=getattr(args, "accum_dtype", "float64")
                )
                aqlm_layer.fit(all_x, debug=args.verbose)
                
                # Replace the original linear layer with the quantized one
                name_parts = name.split('.')
                parent = layer
                for p in name_parts[:-1]: parent = getattr(parent, p)
                setattr(parent, name_parts[-1], aqlm_layer.quantized_linear)
                
                del all_x
                torch.cuda.empty_cache()
                
            inps = new_inps
            
            # Save Checkpoint after each layer
            if checkpoint_dir:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_layer_{i}.pt")
                print(f"  Saving checkpoint: {checkpoint_path}")
                torch.save({
                    "layer_index": i,
                    "model": model,
                    "inps": inps # These are inputs for layer i+1
                }, checkpoint_path)
                
                # Optional: remove previous checkpoint to save space
                if i > 0:
                    prev_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_layer_{i-1}.pt")
                    if os.path.exists(prev_checkpoint):
                        os.remove(prev_checkpoint)
        
        if args.save:
            print(f"Saving quantized model to {args.save}...")
            try:
                # Create a subdirectory for the model and its configs
                model_dir = args.save
                os.makedirs(model_dir, exist_ok=True)
                
                # 1. Save the full model inside the directory
                model_save_path = os.path.join(model_dir, "model.pt")
                torch.save(model, model_save_path)
                
                # 2. Combine Architecture and Quantitative configurations
                full_config = {
                    "architecture": model.config.to_dict(),
                    "quantization_settings": vars(args)
                }
                
                # 3. Save to JSON inside the directory
                config_json_path = os.path.join(model_dir, "config.json")
                with open(config_json_path, 'w', encoding='utf-8') as f:
                    serializable_config = json.loads(json.dumps(full_config, default=lambda o: str(o)))
                    json.dump(serializable_config, f, indent=4, ensure_ascii=False)
                
                # 4. Save a .pt version for legacy support
                config_pt_path = os.path.join(model_dir, "config.pt")
                torch.save(full_config, config_pt_path)
                
                print(f"Model saved in directory: {model_dir}")
                print(f"  - Weights: {model_save_path}")
                print(f"  - Config (JSON): {config_json_path}")
                
                # Update args.save to point to the actual model file for the evaluation step below
                args.model_file_path = model_save_path
                
            except Exception as e:
                print(f"Error during saving: {e}")

        # Cleanup checkpoints after successful completion
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            import shutil
            shutil.rmtree(checkpoint_dir)
            print(f"Cleaned up checkpoints in {checkpoint_dir}")

    if args.no_eval or not args.save:
        return

    print(f"Loading custom LlamaAQLM from {args.model_file_path} for evaluation...")
    from llama_backend.llama_aqlm import LlamaAQLM
    
    # Reload the quantized model into custom backend class
    llama_model = LlamaAQLM(args.model_file_path, device="cuda")
    
    # Custom PPL Evaluation
    if args.eval_ppl:
        print("Evaluating WikiText2 Perplexity (Custom Backend)...")
        from custom_aqlm.quantize.datautils import get_wikitext2
        test_data = get_wikitext2(0, args.model_seqlen, tokenizer, eval_mode=True)
        if hasattr(test_data, "input_ids"): test_data = test_data.input_ids
        
        def evaluate_perplexity_custom(model, test_ids, seqlen, device):
            print(f"Test data shape: {test_ids.shape}")
            nsamples = test_ids.numel() // seqlen
            nlls = []
            loss_fct = torch.nn.CrossEntropyLoss()
            
            for i in range(nsamples):
                batch = test_ids[:, i*seqlen : (i+1)*seqlen].to(device)
                if batch.size(1) < 1: continue
                
                inputs = {'input_ids': batch}
                logits = model.single_step(inputs)
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch[..., 1:].contiguous()
                
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
                
            if not nlls: return float('inf')
            return torch.exp(torch.stack(nlls).mean())

        ppl = evaluate_perplexity_custom(llama_model, test_data, args.model_seqlen, device="cuda")
        print(f"WikiText2 PPL: {ppl:.4f}")
        if args.wandb:
            wandb.log({"WikiText2 PPL": ppl})
    
    # Custom HellaSwag Evaluation
    if args.eval_hellaswag:
        def evaluate_hellaswag_custom(model, tokenizer, device):
            try:
                from testbench import task_script_hellaswag
                class Args:
                    max_samples = None
                
                def model_single_step(inputs):
                    for k,v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(device)
                    logits = model.single_step(inputs)
                    return logits
                    
                print("Running Hellaswag (Custom Backend)...")
                acc, acc_norm, _, _, _ = task_script_hellaswag.task_script_hellaswag(
                    Args(), tokenizer, model_single_step, device, print_first_5_examples=False
                )
                print(f"HellaSwag Accuracy: {acc:.4f}, Norm: {acc_norm:.4f}")
                if args.wandb:
                    wandb.log({"HellaSwag Acc": acc, "HellaSwag Acc Norm": acc_norm})
                return acc_norm
            except ImportError:
                print("Could not import task_script_hellaswag. Check path.")
                return 0.0
            except Exception as e:
                print(f"HellaSwag failed: {e}")
                return 0.0
                
        evaluate_hellaswag_custom(llama_model, tokenizer, device="cuda")

if __name__ == "__main__":
    main()
