import torch
import torch.nn as nn
from tqdm import trange
from tqdm.auto import trange
from transformers import PreTrainedModel
from argparse import Namespace
import time
import os
from typing import Sequence, Dict, Any, Optional

try:
    import wandb
    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from custom_aqlm.quantize.aq_engine import AQEngine
from custom_aqlm.utils.aq import QuantizedLinear
from custom_aqlm.quantize.finetune import finetune_groupwise
from custom_aqlm.quantize.datautils import get_loaders
from custom_aqlm.utils.modelutils import (
    find_sublayers,
    get_layers,
    get_sequential_groups,
    save_not_quantized_weights,
)
from custom_aqlm.utils.utils import using_tf32
from custom_aqlm.inference.inference_utils import get_inps, update_outs


class _LayerWrapperThatAccumulatesXTX(nn.Module):
    def __init__(self, layer: nn.Module, aq_handler: AQEngine):
        super().__init__()
        self.wrapped_layer, self.aq_handler = layer, aq_handler

    def forward(self, input, *args, **kwargs):
        self.aq_handler.add_batch(input)
        return self.wrapped_layer(input, *args, **kwargs)


@torch.no_grad()
def init_aq_engines(
    layer: nn.Module,
    names: Sequence[str],
    inps_tensor: torch.Tensor,
    outs_tensor: torch.Tensor,
    **forward_args: Dict[str, Any],
) -> Dict[str, AQEngine]:
    """
    Create a dictionary of AQUtil instances for each quantized layer;
    Run forward pass on each sample in inps_tensor; write output activations to outs_tensor (in-plance)
    Accumulate XTX to each one of aq_handlers
    :param layer: transformer layer with one or more linear layer to be quantized
    :param names: a list/tuple of string names for linear sub-layers inside :layer: that shall be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a dictionary where keys are full layer names and values are AQUtil instances ready to run .quantize
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    all_sublayers = find_sublayers(layer)
    subset = {name: all_sublayers[name] for name in names}
    assert len(subset) > 0
    aq_handlers = {}
    for sublayer_name in subset:
        aq_handlers[sublayer_name] = AQEngine(subset[sublayer_name])

    # wrap all quantized sub-layers with a wrapper that accumulates inputs on forward
    # note: the code below uses wrappers instead of hooks because hooks cause bugs in multi-gpu code
    wrapped_layer_to_hander = {aq_handler.layer: aq_handler for aq_handler in aq_handlers.values()}
    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if child in wrapped_layer_to_hander:
                setattr(module, child_name, _LayerWrapperThatAccumulatesXTX(child, wrapped_layer_to_hander[child]))

    # compute output activations and accumulate XTX
    for j in trange(len(inps_tensor), desc="calc outs before quantization", leave=False):
        outs_tensor[j].copy_(
            layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0].view_as(outs_tensor[j]), non_blocking=True
        )

    # remove wrappers
    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, _LayerWrapperThatAccumulatesXTX):
                setattr(module, child_name, child.wrapped_layer)
    return aq_handlers


@torch.no_grad()
def quantize_aq(model: PreTrainedModel, data: Sequence, val_data: Optional[Sequence], args: Namespace):
    """
    The core quantization loop.
    
    1. Gets initial input activations (via get_inps).
    2. Iterates over each layer (Transformer Block):
       a. Computes output activations (outs).
       b. Quantizes the Linear modules within the block using AQLM. (Linear -> QuantizedLinear)
       c. Finetunes the quantized layer end-to-end to minimize error.
       d. Updates 'inps' to be the output of this quantized layer (which becomes input for the next).
    """
    assert not torch.backends.cuda.matmul.allow_tf32
    print("\nStarting AQ quantization ...")
    
    device = args.devices[0]
    inps, forward_args = get_inps(model, data, args.model_seqlen, device, args.offload_activations)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]

    if val_data:
        run_validation = True
        val_inps, _ = get_inps(model, val_data, args.model_seqlen, device, args.offload_activations)
        val_outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in val_inps]
    else:
        run_validation = False
        val_inps, val_outs = None, None

    use_cache = model.config.use_cache
    num_codebooks = args.num_codebooks
    model.config.use_cache = False

    quantizers = {}
    overall_bits = 0
    number_of_quantized_params = 0
    layers = get_layers(model)

    # Use trange for global progress bar and ETA
    layer_iterator = trange(len(layers), desc="Quantizing layers", unit="layer")
    for layer_index in layer_iterator:
        print(f"\n---------------- Layer {layer_index} of {len(layers)} ----------------")
        stats_payload = {}
        start_time = time.time()

        # quantized layer will return there
        layer_device_original = next(layers[layer_index].parameters()).device
        # backup layer dtype
        layer_dtype_original = next(layers[layer_index].parameters()).dtype
        print(f"{layer_device_original=}")
        layer = layers[layer_index].to(device)
        for k, v in forward_args.items():
            forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v

        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(find_sublayers(layer).keys())]

        loaded_layer = False
        if args.resume:
            assert args.save is not None, "using --resume requires a --save path to resume from"
            layer_save_path = os.path.join(args.save, f"{layer_index}.pth")
            if os.path.exists(layer_save_path):
                print(f"Loading layer {layer_index} from {layer_save_path}")
                layer = torch.load(layer_save_path, map_location=device)
                loaded_layer = True

        # prepare validation  outputs
        if run_validation and not loaded_layer:  # note: if we skip validation, val_outs will still be updated later
            assert len(val_inps) == len(val_outs) == 1
            update_outs(layer, val_inps[0], val_outs[0], compute_mse=not args.skip_out_loss, **forward_args)

        for names in sequential:
            if loaded_layer:
                print("Skipping quantization: loaded a previously quantized layer")
                break
            
            assert len(inps) == len(outs) == 1  # number of per-device inputs/outputs
            # Filter gate and mixtral related layers logic removed
            aq_handlers = init_aq_engines(
                layer,
                names,
                inps[0],
                outs[0],
                **forward_args,
            )
            
            for sublayer_name in aq_handlers.keys():
                print(f"Quantizing module {sublayer_name} of layer {layer_index}")
                
                # Removed Mixtral/MoE logic here
                
                quantized_weight = aq_handlers[sublayer_name].quantize(args=args, verbose=True)

                with torch.no_grad():
                    assert aq_handlers[sublayer_name].layer.weight in set(
                        layer.parameters()
                    )  # test that this is not a replica

                    new_linear = QuantizedLinear(quantized_weight, aq_handlers[sublayer_name].layer.bias)
                    if args.use_checkpointing:
                        new_linear.use_checkpoint = True
                        print("ENABLED CHECKPOINTING FOR", sublayer_name)
                    found_original = False
                    for submodule in layer.modules():
                        for child_name, child_module in submodule.named_children():
                            if child_module is aq_handlers[sublayer_name].layer:
                                setattr(submodule, child_name, new_linear)
                                found_original = True  # note: do not break to handle tied layers

                    assert found_original, f"could not find {sublayer_name}"

                weight_avg_bits = quantized_weight.estimate_nbits_per_parameter()
                overall_bits += int(weight_avg_bits * torch.numel(aq_handlers[sublayer_name].layer.weight.data))
                number_of_quantized_params += torch.numel(aq_handlers[sublayer_name].layer.weight.data)
                print("curent_avg_bits", overall_bits / number_of_quantized_params)
                quantizers["model.layers.%d.%s" % (layer_index, sublayer_name)] = ()  # to be updated

            del aq_handlers
            assert not loaded_layer

            print("PREPARING TO FINETUNE")
            print(layer)
            layer = layer.to(dtype=torch.float32)
            with using_tf32(enabled=True):
                layer = finetune_groupwise(
                    layer=layer,
                    train_inps=inps,
                    train_outs=outs,
                    args=args,
                    valid_inps=val_inps,
                    valid_outs=val_outs,
                    **forward_args,
                )
            layer = layer.to(dtype=layer_dtype_original)
            print("FINISHED FINETUNING")

        if args.save and not loaded_layer:
            os.makedirs(args.save, exist_ok=True)
            layer_save_path = os.path.join(args.save, f"{layer_index}.pth")
            print(f"Saving layer {layer_index}... to {layer_save_path}")
            torch.save(layer, layer_save_path)
            if args.on_save:
                exec(args.on_save)  # a callback e.g. to save progress in slurm or similar distributed infrastructure

        should_compute_mse = not (args.skip_out_loss or loaded_layer)
        assert len(inps) == len(outs) == 1
        out_losses = update_outs(layer, inps[0], outs[0], compute_mse=should_compute_mse, **forward_args)
        stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()

        if run_validation:
            assert len(val_inps) == len(val_outs) == 1
            out_val_losses = update_outs(
                layer, val_inps[0], val_outs[0], compute_mse=should_compute_mse, **forward_args
            )
            stats_payload["out_val_loss"] = torch.mean(torch.Tensor(out_val_losses)).item()

        layers[layer_index] = layer.to(layer_device_original)
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        if run_validation:
            val_inps, val_outs = val_outs, val_inps

        # Logging
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["Step"] = layer_index
        if args.wandb and has_wandb:
            wandb.log(stats_payload, step=layer_index)
        if not loaded_layer:
            print(stats_payload)

    print("=====================\nFinal stats:")
    if args.save:
        torch.save(vars(args), os.path.join(args.save, "args.pt"))
        save_not_quantized_weights(model, args.save)
        if args.on_save:
            exec(args.on_save)  # a callback e.g. to save progress in slurm or similar distributed infrastructure

    if args.wandb and has_wandb:
        wandb.log({"max_cuda_mem_quantize": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        if number_of_quantized_params > 0:  # do not report avg bits if we load all pre-quantized layers via --resume
            wandb.log({"Avg_bits": overall_bits / number_of_quantized_params})
    model.config.use_cache = use_cache
    print(f"quantize: {torch.cuda.max_memory_allocated()=:,}")
    return quantizers

def quantize_model(model: PreTrainedModel, args: Namespace):
    """
    Main entry point to functions for model quantization.
    
    1. Loads calibration data (C4/WikiText2).
    2. Calls 'quantize_aq' to perform layer-by-layer quantization.
    3. Returns the quantization statistics.
    """
    tick = time.time()
    print("Loading data ...")
    data = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.model_path,
        seqlen=args.model_seqlen,
        use_fast_tokenizer=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if args.val_size > 0:
        all_ids = torch.randperm(len(data))
        train_ids, val_ids = all_ids[args.val_size :], all_ids[: args.val_size]
        train_data = [data[i] for i in train_ids]
        val_data = [data[i] for i in val_ids]
    else:
        train_data = data
        val_data = None

    results = quantize_aq(model, train_data, val_data, args)
    print(f"quantization time: {time.time() - tick:.1f}")
    return results
