
import torch
import torch.nn as nn
from argparse import Namespace
import sys
import os

# Add the directory containing this file to sys.path to ensure local imports work
# This allows 'import aq_engine' to work if this file is imported from elsewhere
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import necessary components from the existing AQLM implementation
from custom_aqlm.quantize.aq_engine import AQEngine
from custom_aqlm.utils.aq import QuantizedLinear

class AqlmLayer(nn.Module):
    """
    A unified AQLM Layer that encapsulates the original linear layer,
    the quantization engine, and the resulting quantized linear module.
    
    This follows the pattern of MaddnessHadamardLayer with fit() and forward() methods.
    It provides a high-level API to quantize a single layer without managing the complex loop manually.
    """
    def __init__(
        self,
        original_layer: nn.Linear,
        num_codebooks: int = 1,
        nbits_per_codebook: int = 16,
        in_group_size: int = 8,
        out_group_size: int = 1,
        scale_nbits: int = 0,
        codebook_value_nbits: int = 16,
        init_max_iter: int = 100,
        lr: float = 1e-4,
        max_epochs: int = 10,
        beam_size: int = 1,
        steps_per_epoch: int = 100,
        relative_mse_tolerance: float = None,
        use_bfp: bool = False,
        accum_dtype: str = "float64"
    ):
        """
        Initialize the AqlmLayer.

        Args:
            original_layer (nn.Linear): The original linear layer to be quantized.
            
            # --- Architecture Hyperparameters ---
            num_codebooks (int): Number of codebooks per layer (M).
            nbits_per_codebook (int): Number of bits per codebook (B). Codebook size will be 2**B.
            in_group_size (int): Input group size (block size). Weights in this block (1 x in_group_size) are quantized together.
            out_group_size (int): Output group size. Usually 1.
            
            # --- Metadata Compression Hyperparameters ---
            scale_nbits (int): Bits for scale quantization. 0 means keep scales in FP16/BF16.
            codebook_value_nbits (int): Bits for codebook values. 16 means FP16. <16 means quantized codebooks (double quantization).
            
            # --- Optimization Hyperparameters ---
            init_max_iter (int): K-means initialization iterations.
            lr (float): Learning rate for codebook optimization (continuous part).
            max_epochs (int): Max epochs for optimization.
            beam_size (int): Beam size for Beam Search (discrete part). Higher is better but slower.
            steps_per_epoch (int): Optimizer steps before every beam search round.
            relative_mse_tolerance (float): Tolerance for early stopping.
            use_bfp (bool): Whether to use Block Floating Point for codebook values (post-training compression).
            accum_dtype (str): Dtype for XTX accumulation ("float64", "float32", "float16").
        """
        super().__init__()
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Store configuration
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.scale_nbits = scale_nbits
        self.codebook_value_nbits = codebook_value_nbits
        self.init_max_iter = init_max_iter
        self.lr = lr
        self.max_epochs = max_epochs
        self.beam_size = beam_size
        self.steps_per_epoch = steps_per_epoch
        self.relative_mse_tolerance = relative_mse_tolerance
        self.use_bfp = use_bfp
        self.accum_dtype = accum_dtype
        
        # This will hold the quantized module after calling fit()
        self.quantized_linear = None
        
        # Initialize the engine which handles XTX accumulation and optimization
        
        dtype_map = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16
        }
        engine_dtype = dtype_map.get(accum_dtype, torch.float64)
        
        self.aq_engine = AQEngine(self.original_layer, accumulator_dtype=engine_dtype)

    def fit(self, X: torch.Tensor, debug: bool = False):
        """
        Fits the AQLM model to the provided input activations X.
        
        Args:
            X (torch.Tensor): Input activations of shape [N, in_features] or [N, seq_len, in_features].
            debug (bool): Whether to print debug information during training.
        """
        if debug:
            print(f"Accumulating XTX with input shape {X.shape}...")
        
        # 1. Accumulate XTX statistics
        # Iterate over chunks to prevent OOM
        chunk_size = 1024 # Adjustable chunk size
        with torch.no_grad():
             total_samples = X.shape[0]
             for i in range(0, total_samples, chunk_size):
                 chunk = X[i:i+chunk_size].to(self.aq_engine.device)
                 self.aq_engine.add_batch(chunk)
                 del chunk
                 if i % (chunk_size * 5) == 0:
                     torch.cuda.empty_cache()
            
        # 2. Prepare arguments for the quantization process
        # If use_bfp is enabled, we perform training with FP16 codebooks (nbits=16)
        # to ensure stable convergence, then switch to BFP for inference.
        train_codebook_value_nbits = 16 if self.use_bfp else self.codebook_value_nbits
        
        args = Namespace(
            devices=[self.aq_engine.device],
            num_codebooks=self.num_codebooks,
            nbits_per_codebook=self.nbits_per_codebook,
            in_group_size=self.in_group_size,
            out_group_size=self.out_group_size,
            scale_nbits=self.scale_nbits,
            codebook_value_nbits=train_codebook_value_nbits,
            codebook_value_num_groups=1,
            init_max_iter=self.init_max_iter,
            init_max_points_per_centroid=None,
            lr=self.lr,
            max_epochs=self.max_epochs,
            beam_size=self.beam_size,
            steps_per_epoch=self.steps_per_epoch,
            relative_mse_tolerance=self.relative_mse_tolerance,
            print_frequency=10 if debug else 1000000,
            use_bfp=False # Always False during training (managed manually post-training)
        )
        
        if debug:
            print("Running AQLM optimization (Codebook training + Beam Search)...")
            
        # 3. Running the optimization
        # This returns a src.aq.QuantizedWeight object
        quantized_weight = self.aq_engine.quantize(args=args, verbose=debug)
        
        # 4. Apply Post-Training BFP settings
        if self.use_bfp:
             quantized_weight.use_bfp = True
             quantized_weight.codebook_value_nbits = self.codebook_value_nbits
             # This will trigger BFP quantization on next forward/get_codebooks
        
        # 5. Construct the QuantizedLinear module
        self.quantized_linear = QuantizedLinear(quantized_weight, self.original_layer.bias)
        
        # 6. Cleanup to free memory
        self.aq_engine = None
        self.original_layer = None 
        
        if debug:
            print("AQLM Layer fitted successfully.")

    def forward(self, x: torch.Tensor):
        """
        Forward pass using the quantized weights.
        """
        if self.quantized_linear is None:
             raise RuntimeError("AqlmLayer not fitted. Call .fit(X) first.")
        return self.quantized_linear(x)
