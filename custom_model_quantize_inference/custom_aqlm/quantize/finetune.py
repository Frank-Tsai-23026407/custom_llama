"""Utilities for internal **block-wise** finetuning used during initial AQLM calibration"""
from __future__ import annotations

import warnings
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.scatter_gather import Gather

from custom_aqlm.quantize.aq_engine import replace_parameter_
from custom_aqlm.utils.utils import iterate_minibatches


@torch.enable_grad()
def finetune_groupwise(
    *,
    layer: nn.Module,
    train_inps: Sequence[torch.Tensor],
    train_outs: Sequence[torch.Tensor],
    args: Namespace,
    valid_inps: Sequence[torch.Tensor] = None,
    valid_outs: Sequence[torch.Tensor] = None,
    verbose: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Fine-tune a module with pre-quantized linear layers so as to minimize MSE between layer-wise inps/outs

    :param layer: a trainable module where linear layers are replaced by QuantizedLinear instances
    :param inps: a list of tensors of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs: a list of tensors of previous output activations, [nsamples_per_device, seq_len, hidden_size]
    :param args: quantization hyperparameters from main.py
    :param kwargs: additional keyword arguments to be passed into layer on each forward
    """
    assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
    assert isinstance(train_inps, (list, tuple)) and isinstance(train_inps, (list, tuple))
    # Simplify assertions for single device
    assert len(train_inps) == len(train_outs) == 1
    
    device = args.devices[0]
    
    assert isinstance(train_inps[0], torch.Tensor) and isinstance(train_outs[0], torch.Tensor)
    if not args.offload_activations:
        assert train_inps[0].device == train_outs[0].device == device
    else:
        assert train_inps[0].device == train_outs[0].device == torch.device("cpu")

    # initialize trainable parameters on main device
    differentiable_parameters_by_name = {name: param for name, param in layer.named_parameters() if param.requires_grad}
    differentiable_parameters = nn.ParameterList(differentiable_parameters_by_name.values())
    for param in differentiable_parameters:
        param.grad = torch.zeros_like(param)

    print(f"Fine-tuning {sum(param.numel() for param in differentiable_parameters)} parameters")
    opt = torch.optim.Adam(
        differentiable_parameters, lr=args.finetune_lr, betas=(args.finetune_adam_beta1, args.finetune_adam_beta2)
    )

    num_samples_per_device = len(train_inps[0])
    local_batch_size = args.local_batch_size
    if local_batch_size is None:
        local_batch_size = args.finetune_batch_size 

    assert all(len(inps_tensor) == num_samples_per_device for inps_tensor in train_inps)
    assert args.finetune_batch_size % local_batch_size == 0, ""
    num_accumulation_steps = args.finetune_batch_size // local_batch_size
    assert num_samples_per_device % local_batch_size * num_accumulation_steps == 0, (
        num_samples_per_device,
        local_batch_size,
    )
    train_batches_per_epoch = num_samples_per_device // local_batch_size
    # Single iterator
    train_batch_iterator = iterate_minibatches(train_inps[0], train_outs[0], batch_size=local_batch_size, device=device)

    run_validation = False
    if valid_inps and valid_outs:
        run_validation = True
        num_valid_samples_per_device = len(valid_inps[0])
        valid_batches_per_epoch = num_valid_samples_per_device // local_batch_size
        valid_batch_iterator = iterate_minibatches(valid_inps[0], valid_outs[0], batch_size=local_batch_size, device=device)

    if run_validation:
        # evaluate before training
        layer.eval()
        loss_numerator = loss_denominator = 0
        with torch.no_grad():
            for _ in range(valid_batches_per_epoch):
                loss = _compute_mse_on_batch(layer, valid_batch_iterator, **kwargs)
                loss_numerator += loss.item()
                loss_denominator += 1
        valid_loss_epoch = loss_numerator / loss_denominator
        print(f"Evaluation before training.")
        print(f"valid loss={valid_loss_epoch:.2e}\t")
        best_loss = valid_loss_epoch
        best_parameters_by_name = deepcopy(differentiable_parameters_by_name)
        worse_count = 0

    steps_accumulated = 0
    for epoch in range(args.finetune_max_epochs):
        layer.train()
        # train epoch
        loss_numerator = loss_denominator = 0
        for _ in range(train_batches_per_epoch):
            loss = _compute_mse_on_batch(layer, train_batch_iterator, **kwargs)

            (loss / num_accumulation_steps).backward()
            steps_accumulated += 1

            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")

            if steps_accumulated >= num_accumulation_steps:
                opt.step()
                opt.zero_grad()
                steps_accumulated = 0

            loss_numerator += loss.item()
            loss_denominator += 1
        train_loss_epoch = loss_numerator / loss_denominator
        if run_validation:
            layer.eval()
            # val epoch
            loss_numerator = loss_denominator = 0
            with torch.no_grad():
                for _ in range(valid_batches_per_epoch):
                    loss = _compute_mse_on_batch(layer, valid_batch_iterator, **kwargs)
                    loss_numerator += loss.item()
                    loss_denominator += 1
            valid_loss_epoch = loss_numerator / loss_denominator
        # log losses in the end of the epoch
        if verbose:
            print("-" * 10)
            print(f"epoch={epoch}")
            print(f"train loss={train_loss_epoch:.2e}\t")
            if run_validation:
                print(f"valid loss={valid_loss_epoch:.2e}\t")

        if run_validation:
            if valid_loss_epoch < best_loss:
                print(f"new best loss {valid_loss_epoch:.2e} on epoch {epoch}")
                best_loss = valid_loss_epoch
                best_parameters_by_name = deepcopy(differentiable_parameters_by_name)
                worse_count = 0
            else:
                worse_count += 1
                if worse_count >= args.finetune_early_stop:
                    break

    if run_validation:
        layer.load_state_dict(best_parameters_by_name, strict=False)

    return layer


def _compute_mse_on_batch(
    layer: nn.Module, batch_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]], **kwargs
) -> torch.Tensor:
    """
    Compute the activation MSE error between transformer layers
    :param
    """
    inps_batch, outs_batch = next(batch_iter)
    inps_batch = inps_batch.to(dtype=torch.float32)
    outs_batch = outs_batch.to(dtype=torch.float32)

    if inps_batch.shape[0] != 1:  # replicate kwargs to match the batch size
        for name, value in list(kwargs.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                if name not in ("attention_mask", "position_ids"):
                    warnings.warn(f"Tiling an unexpected kwarg {name} over batch size; make sure this is valid.")
                repeats = [len(inps_batch)] + [1 for _ in range(value.ndim - 1)]
                kwargs[name] = value.tile(*repeats)

    outs_prediction, *_unused = layer(inps_batch, **kwargs)
    assert outs_prediction.shape == outs_batch.shape
    return F.mse_loss(outs_prediction, outs_batch)
