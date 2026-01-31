from __future__ import annotations

import math
import random
from argparse import Namespace
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from custom_aqlm.utils.aq import QuantizedWeight
from custom_aqlm.utils.utils import ellipsis


class AQEngine(nn.Module):
    """A wrapper class that runs AQ training for a single linear layer. All the important math is in aq.py"""

    def __init__(self, layer: nn.Linear, accumulator_dtype: torch.dtype = torch.float64):
        super().__init__()
        self.layer = layer
        self.device = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.register_buffer(
            "XTX", torch.zeros((self.columns, self.columns), dtype=accumulator_dtype, device=self.device)
        )
        self.quantized_weight: Optional[QuantizedWeight] = None
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """
        Accumulate a minibatch of layer inputs and update the X.T @ X (aka half hessian).
        
        This constitutes the calibration phase where we learn the "importance" of different input features.
        The metric we want to minimize is the output reconstruction error: ||(W - W_q)X||^2.
        This expands to: tr((W - W_q) X X^T (W - W_q)^T).
        
        Here, we compute H = X X^T (or X^T X depending on shape conventions), which acts as the weighting matrix.
        """
        assert self.XTX is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        tmp = inp.shape[0] # Number of tokens in this batch
        inp = inp.t() # Shape: [in_features, num_tokens]

        # Update the running average of XTX
        # self.nsamples track total number of tokens seen so far
        self.XTX *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        
        # Normalize the new batch by total samples to keep the running average consistent
        inp = math.sqrt(1 / self.nsamples) * inp.to(self.XTX.dtype)
        
        # Update H += X_new @ X_new.T
        self.XTX += inp.matmul(inp.t())

    @torch.enable_grad()
    def quantize(self, *, args: Namespace, verbose: bool = True) -> QuantizedWeight:
        """
        Create a QuantizedLinear with specified args based on the collected hessian (XTX) data.
        
        This is the main optimization loop dealing with the "hard" problem of AQLM:
        finding the best discrete codes (indices) and continuous codebooks to approximate W.
        """
        assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
        assert args.devices[0] == self.device, (args.devices[0], self.XTX.device)
        
        # 1. Initialization: Create the QuantizedWeight object.
        #    This internally calls 'init_aq_kmeans' to run Residual K-Means for a good starting point.
        self.quantized_weight = QuantizedWeight(
            reference_weight=self.layer.weight.detach().to(device=self.device, dtype=torch.float32),
            out_group_size=args.out_group_size,
            in_group_size=args.in_group_size,
            num_codebooks=args.num_codebooks,
            nbits_per_codebook=args.nbits_per_codebook,
            codebook_value_nbits=args.codebook_value_nbits,
            codebook_value_num_groups=args.codebook_value_num_groups,
            scale_nbits=args.scale_nbits,
            max_iter=args.init_max_iter,
            max_points_per_centroid=args.init_max_points_per_centroid,
            devices=args.devices,
            verbose=True,
        )

        # 2. Setup Optimizer: We will optimize the 'codebooks' (vectors) values using Adam.
        #    Note: 'codes' (integers) are NOT differentiable and are optimized via Beam Search explicitly.
        differentiable_parameters = nn.ParameterDict(
            {name: param for name, param in self.quantized_weight.named_parameters() if param.requires_grad}
        )
        opt = torch.optim.Adam(differentiable_parameters.values(), lr=args.lr, betas=(0.0, 0.95), amsgrad=True)

        previous_best_loss = float("inf")  # for early stopping
        
        # 3. Main Loop: Coordinate Descent
        #    Alternates between updating Codebooks (Continuous opt) and Codes (Discrete opt)
        for epoch in range(args.max_epochs):
            # --- Step A: Optimize Codebooks (Continuous) ---
            # Fix the discrete codes, update the vector values to minimize MSE.
            # Add progress bar for Step A
            from tqdm import tqdm
            step_iterator = range(args.steps_per_epoch)
            if verbose:
                step_iterator = tqdm(step_iterator, desc=f"Optimize Codebooks (Epoch {epoch})", leave=False)
            
            for step in step_iterator:
                loss = self._compute_mse()

                if not torch.isfinite(loss).item():
                    raise ValueError(f"Quantization loss is {loss}")
                    
                # Early Stopping check
                if step == 0 and args.relative_mse_tolerance is not None:
                    if loss.item() / previous_best_loss > (1.0 - args.relative_mse_tolerance):
                        return self.quantized_weight  # early stopping; no updates after last epoch's beam search
                    previous_best_loss = min(previous_best_loss, loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()
                
                # Update progress bar description with loss if verbose
                if verbose and (epoch * args.steps_per_epoch + step) % args.print_frequency == 0:
                   if hasattr(step_iterator, "set_postfix"):
                       step_iterator.set_postfix(loss=f"{loss.item():.6f}")
                   else:
                       print(f"epoch={epoch}\tstep={step}\tloss={loss.item():.10f}\t")

            # --- Step B: Optimize Codes (Discrete) ---
            # Fix the codebooks, find the best set of indices (codes) using Beam Search.
            # This is done every 'steps_per_epoch' gradient updates.
            seed = random.getrandbits(256)
            self.beam_search_update_codes_(
                seed=seed,
                beam_size=args.beam_size,
                verbose=True,
            )
        return self.quantized_weight

    def _compute_mse(self, selection: Union[slice, ellipsis] = ...) -> torch.Tensor:
        """
        Compute the activation MSE error = ||X @ quantized_weight - X @ reference_weight||^2
        
        Using the trace trick and pre-computed XTX (which is X @ X.T or X.T @ X), this can be computed 
        efficiently without iterating over data samples:
        Error = || (W - W_q) X ||^2 
              = tr( (W - W_q) X X^T (W - W_q)^T )
              = tr( Delta @ XTX @ Delta.T )
        
        where Delta = (W_q - W) is the quantization error matrix.
        
        :param selection:  By default, compute MSE normally. If selection is specified, this method will instead
            compute MSE over a portion of output channels that align with the selected out_groups (for parallelism)
            The indices / slices must correspond to output channels (if out_group_size==1) or groups (if > 1).
            Formally, the indices must be in range [ 0 , self.out_features // self.out_group_size )
        """
        assert self.quantized_weight is not None, "must be called inside / after AQUtil.quantize"
        
        # 1. Reconstruct the approximate weights (W_q) from codes and codebooks
        #    This uses the 'forward' pass of QuantizedWeight which performs the summation.
        quantized_weight = self.quantized_weight(selection)

        if isinstance(selection, ellipsis):
            reference_weight = self.layer.weight.detach().to(quantized_weight.dtype)
        else:
            assert isinstance(selection, slice)
            out_channel_selection = slice(
                selection.start * self.quantized_weight.out_group_size,
                selection.stop * self.quantized_weight.out_group_size,
            )

            reference_weight = self.layer.weight.detach()[out_channel_selection].to(quantized_weight.dtype)
        
        # 2. Compute Delta = (W_q - W)
        delta_weight = (quantized_weight - reference_weight).to(self.XTX.dtype)
        
        # 3. Compute Loss = tr(Delta @ XTX @ Delta.T)
        #    Note: (A @ B).flatten() @ C.flatten() is equivalent to trace(A @ B @ C.T) 
        #    Here: (Delta @ XTX) . dot(Delta)  -> effective trace(Delta @ XTX @ Delta.T)
        return (delta_weight @ self.XTX).flatten() @ delta_weight.flatten() / self.quantized_weight.out_features

    @torch.no_grad()
    def beam_search_update_codes_(
        self,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Update quantized_weight codes in-place via beam search"""
        dtype = self.quantized_weight.codebooks.dtype
        self.quantized_weight.beam_search_update_codes_(
            XTX=self.XTX.to(dtype),
            reference_weight=self.layer.weight.detach().to(dtype),
            dim_rng=random.Random(seed),
            **kwargs,
        )


def replace_parameter_(module: nn.Module, name: str, new_value: torch.Tensor):
    """A hacky way to substitute an already registered parameter with a non-parameter tensor. Breaks future use."""
    if name in module._parameters:
        module._parameters[name] = new_value
    else:
        setattr(module, name, new_value)
