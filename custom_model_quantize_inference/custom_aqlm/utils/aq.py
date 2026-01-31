""" Core mathematics for Additive Quantization (AQ): initialization, reconstruction and beam search"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm.auto import trange

from custom_aqlm.quantize.beam_search_l2 import beam_search_optimal_codes as beam_search_minimize_weight_mse
from custom_aqlm.quantize.beam_search_xtx import beam_search_optimal_codes as beam_search_minimize_activation_mse
from custom_aqlm.quantize.kmeans import find_nearest_cluster, fit_faiss_kmeans, fit_kmeans, fit_kmeans_1d
from custom_aqlm.utils.utils import IntCodes, _dequantize_weight, ellipsis, is_signed


class QuantizedLinear(nn.Module):
    """
    The runtime drop-in replacement for nn.Linear.
    
    It maintains the QuantizedWeight object and performs the forward pass by:
    1. Reconstructing the weights (dequantization).
    2. Performing the standard linear matrix multiplication.
    """
    def __init__(self, quantized_weight: QuantizedWeight, bias: Optional[nn.Parameter]):
        super().__init__()
        self.out_features, self.in_features = quantized_weight.out_features, quantized_weight.in_features
        self.quantized_weight = quantized_weight
        self.bias = bias
        self.use_checkpoint = False

    def _forward(self, input: torch.Tensor):
        # self.quantized_weight() calls QuantizedWeight.forward(), returning the reconstructed float matrix
        weight = self.quantized_weight()
        if weight.dtype != input.dtype:
             weight = weight.to(input.dtype)
        return F.linear(input, weight, self.bias)

    def forward(self, input: torch.Tensor):
        if getattr(self, "use_checkpoint", False) and torch.is_grad_enabled():
            return checkpoint(
                self._forward, input, use_reentrant=False, preserve_rng_state=False, determinism_check="none"
            )
        return self._forward(input)


class QuantizedWeight(nn.Module):
    """
    The main container for AQLM parameters.
    
    It stores:
    - codebooks: The dictionary of vectors.
    - codes (indices): Which vector to pick from each codebook.
    - scales: Optional scaling factors.
    """
    EPS = 1e-9

    def __init__(
        self,
        *,
        reference_weight: torch.Tensor,
        in_group_size: int,
        out_group_size: int,
        num_codebooks: int,
        nbits_per_codebook: int = 8,
        codebook_value_nbits: int = 16,
        codebook_value_num_groups: int = 1,
        scale_nbits: int = 0,
        straight_through_gradient: Optional[bool] = None,
        code_dtype: torch.dtype = torch.int32,
        **init_kwargs,
    ):
        super().__init__()
        self.out_features, self.in_features = reference_weight.shape
        # ... validation checks ...
        assert self.in_features % in_group_size == 0
        assert self.out_features % out_group_size == 0
        if nbits_per_codebook > torch.iinfo(code_dtype).bits - is_signed(code_dtype):
            raise ValueError(f"Code dtype cannot store {nbits_per_codebook} bits; please specify code_dtype manually")

        self.out_group_size, self.in_group_size = out_group_size, in_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.codebook_size = codebook_size = 2**nbits_per_codebook
        self.codebook_value_nbits = codebook_value_nbits
        self.codebook_value_num_groups = codebook_value_num_groups
        self.codebook_value_clusters = None
        self.codebook_value_scales = None

        self.scales = self.scales_clusters = self.scales_indices = None
        if straight_through_gradient is None and scale_nbits > 0:
            straight_through_gradient = scale_nbits >= 6
        self.straight_through_gradient = straight_through_gradient
        self.scale_nbits = scale_nbits

        # --- Scale Initialization ---
        # Initialize scales based on the norm of the weight groups.
        with torch.no_grad():
            weight_groupwise = reference_weight.reshape(
                self.out_features // out_group_size, out_group_size, self.in_features // in_group_size, in_group_size
            ).swapaxes(
                1, 2
            )  # [num_out_groups, num_in_groups, out_group_size, in_group_size]

            # AQLM Standard: Always use Per-Channel (Per-Output-Group) Scales
            # We calculate the norm over the entire input vector for each output group.
            scales = weight_groupwise.flatten(1, -1).norm(dim=-1).view(-1, 1, 1, 1) + self.EPS
            
            # ... quantization logic for scales ...
            self.scales_are_lossless = scale_nbits == 0 or scale_nbits >= 16 or (2**scale_nbits >= scales.shape[1])
            if self.scales_are_lossless or self.straight_through_gradient:
                # ^-- this checks if scales can be preserved losslessly
                self.scales = nn.Parameter(scales, requires_grad=True)
            else:
                scales_clusters, scales_indices, _ = fit_kmeans_1d(scales.flatten(1, -1), k=2**scale_nbits)
                self.scales_clusters = nn.Parameter(scales_clusters, requires_grad=True)
                self.scales_indices = nn.Parameter(scales_indices, requires_grad=False)

            # Normalize weights by scales before initialization to help convergence
            weight_for_init = (weight_groupwise / scales).swapaxes(1, 2).reshape_as(reference_weight)
            del weight_groupwise

        self.use_bfp = init_kwargs.get("use_bfp", False) 

        # --- AQLM Initialization (Residual K-Means) ---
        # Initialize codebooks and codes using iterative K-Means
        codes, codebooks = init_aq_kmeans(
            weight_for_init,
            num_codebooks=num_codebooks,
            out_group_size=out_group_size,
            in_group_size=in_group_size,
            codebook_size=self.codebook_size,
            **init_kwargs,
        )

        self.codebooks = nn.Parameter(
            codebooks, requires_grad=True
        )  # Tensor Shape: [num_codebooks, codebook_size, out_group_size, in_group_size]
        
        self.codes: Optional[nn.Parameter] = nn.Parameter(
            codes.to(code_dtype), requires_grad=False
        )  # Tensor Shape: [num_out_groups, num_in_groups, num_codebooks]
        
        self.codes_storage: Optional[IntCodes] = None  # storage for FSDP compatibility

    def get_codes(self):
        if self.codes is not None:
            return self.codes
        if self.codes_storage is not None:
            return self.codes_storage()
        raise RuntimeError("No codes found")

    def set_codes(self, codes: torch.Tensor):
        if self.codes is not None:
            self.codes.data = codes.to(self.codes.dtype)
        else:
            self.codes = nn.Parameter(codes, requires_grad=False)

    def get_codebooks(self):
        return self.codebooks

    def get_scales(self):
        if self.scales is not None:
            return self.scales
        if self.scales_clusters is not None and self.scales_indices is not None:
            # Reconstruct quantized scales
            # scales_indices was computed on scales of shape [num_out, num_in, 1, 1]
            scale_shape = (
                self.out_features // self.out_group_size,
                self.in_features // self.in_group_size,
                1,
                1,
            )
            return self.scales_clusters[self.scales_indices.long()].view(scale_shape)
        return None

    def forward(self, selection: Union[slice, ellipsis, torch.Tensor] = ...):
        """
        Differentiably reconstruct the weight (or parts thereof) from compressed components.
        
        Formula: W_hat = Scale * Sum(Codebook[Code])
        
        The gradient can flow through 'Codebooks' and 'Scales' to optimize them.
        'Codes' are discrete indices so gradient does NOT flow through them (they are optimized via Beam Search).
        
        :param selection: By default, reconstruct the entire weight.
        """
        # _dequantize_weight handles the gathering and summation of codebook vectors
        weight = _dequantize_weight(self.get_codes()[selection], self.get_codebooks(), self.get_scales()[selection])
        return weight

    @torch.no_grad()
    def beam_search_update_codes_(
        self,
        XTX: torch.Tensor,
        reference_weight: torch.Tensor,
        beam_size: int,
        dim_rng: Optional[random.Random] = None,
        selection: Union[slice, ellipsis] = ...,
        **kwargs,
    ):
        """
        Update codes using Beam Search to minimize local objective (MSE).
        """
        if isinstance(selection, slice):
            codes = self.get_codes()[selection]
            scales = self.get_scales()
            if scales is not None:
                scales = scales[selection]
        else:
            codes = self.get_codes()
            scales = self.get_scales()
            
        new_codes = beam_search_minimize_activation_mse(
            XTX=XTX,
            reference_weight=reference_weight,
            codebooks=self.get_codebooks(),
            prev_codes=codes,
            scales=scales,
            beam_size=beam_size,
            dim_rng=dim_rng,
            verbose=kwargs.get("verbose", False),
            sparsity_regularizer=kwargs.get("sparsity_regularizer", 0),
        )
        
        if isinstance(selection, slice):
            self.get_codes()[selection] = new_codes
        else:
            self.set_codes(new_codes)
        return new_codes


@torch.no_grad()
def init_aq_kmeans(
    reference_weight: torch.Tensor,
    *,
    num_codebooks: int,
    out_group_size: int,
    in_group_size: int,
    codebook_size: int,
    verbose: bool = False,
    use_faiss: bool = False,
    max_points_per_centroid: Optional[int] = None,
    max_iter: int = 1000,
    devices: Optional[List[torch.device]] = None,
    **kwargs,
):
    """
    Create initial codes and codebooks using Residual K-means clustering.
    
    Why Residual K-Means?
    AQLM approximates W as C1[i] + C2[j] + ...
    Standard K-Means only finds one codebook (M=1).
    To find M codebooks, we:
    1. Run K-Means on W -> get C1.
    2. Compute Residual R = W - C1[i].
    3. Run K-Means on R -> get C2.
    4. Repeat.
    """
    out_features, in_features = reference_weight.shape
    num_out_groups = out_features // out_group_size
    num_in_groups = in_features // in_group_size
    
    # Reshape weight into groups (vectors) for clustering
    weight_residue = (
        reference_weight.reshape(num_out_groups, out_group_size, num_in_groups, in_group_size)
        .clone()
        .swapaxes(-3, -2)
        .reshape(num_out_groups * num_in_groups, out_group_size * in_group_size)
    )
    codebooks = []
    codes = []

    if max_points_per_centroid is not None:
        print("Clustering:", max_points_per_centroid * codebook_size, "points from", weight_residue.shape[0])

    for _ in trange(num_codebooks, desc="initializing with kmeans") if verbose else range(num_codebooks):
        if use_faiss:
            # Use Faiss for faster K-Means on GPU
            codebook_i, codes_i, reconstructed_weight_i = fit_faiss_kmeans(
                weight_residue,
                k=codebook_size,
                max_iter=max_iter,
                gpu=(weight_residue.device.type == "cuda"),
                max_points_per_centroid=max_points_per_centroid,
            )
        else:
            # Use Torch implementation
            chosen_ids = None
            if max_points_per_centroid is not None:
                chosen_ids = torch.randperm(weight_residue.shape[0], device=weight_residue.device)[
                    : max_points_per_centroid * codebook_size
                ]
            codebook_i, _, _ = fit_kmeans(
                weight_residue if chosen_ids is None else weight_residue[chosen_ids, :],
                k=codebook_size,
                max_iter=max_iter,
                devices=devices,
                **kwargs,
            )
            # Find closest centroids (codes) for all vectors
            codes_i, reconstructed_weight_i = find_nearest_cluster(weight_residue, codebook_i, devices=devices)

        # Reshape and store
        codes_i = codes_i.reshape(num_out_groups, num_in_groups, 1)
        codebook_i = codebook_i.reshape(1, codebook_size, out_group_size, in_group_size)
        
        # Remove the approximation from the residue (W <- W - C[k])
        # This prepares for the next codebook optimization (Residual K-Means)
        weight_residue -= reconstructed_weight_i
        codes.append(codes_i)
        codebooks.append(codebook_i)
        del reconstructed_weight_i
        
    codebooks = torch.cat(codebooks, dim=0)
    codes = torch.cat(codes, dim=-1)
    return codes, codebooks
