# Import from parent directory and subdirectories
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from plain_script import *
from llama_backend.precision_policy import resolve_policy, PrecisionPolicy
from llama_backend.clone.hf_rope import HFRotaryEmbedding
from llama_backend.clone.clone_backend import clone_decoder_layer, clone_rmsnorm
import argparse
from quantize_model_script.block_quantization import block_floating_point_quantize
from utils import *
from llama_backend.clone.hf_clone import clone_forward_all

class LlamaMyModel:
    """A custom Llama model implementation for flexible backend and precision control.

    This class wraps a Hugging Face Llama model to enable experimentation with different
    execution backends ('custom', 'huggingface', 'clone') and precision policies. It serves
    as a primary interface for running inference, applying quantization, and comparing
    performance against reference implementations.

    Attributes:
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which the model is loaded.
        dtype (torch.dtype): The primary data type for the model's parameters.
        tokenizer (AutoTokenizer): The tokenizer loaded from the pretrained model name.
        model (AutoModelForCausalLM): The underlying Hugging Face model instance.
        backend (str): The execution backend to use for the forward pass.
        clone_compute_dtype (torch.dtype): The data type for computations in the 'clone' backend.
        softmax_fp32 (bool): If True, forces softmax operations to run in float32.
        precision_policy (PrecisionPolicy): The policy defining dtypes for different model parts.
        num_layers (int): The number of transformer layers in the model.
        num_heads (int): The number of attention heads.
        num_kv_heads (int): The number of key/value heads for Grouped Query Attention.
        rope_cache_dtype (torch.dtype): The data type for the RoPE sinusoidal caches.
        head_dim (int): The dimensionality of each attention head.
        rope_emb (HFRotaryEmbedding): The Rotary Positional Embedding module for the 'clone' backend.
        attention_blocks (list): A list of custom `transfomer_block_with_kv_cache` modules.
        stop_criteria (callable): A function to determine when to stop text generation.
    """
    def __init__(self, model_name="TinyLlama/TinyLlama_v1.1", device='cpu', stop_criteria=None, dtype=torch.float32,
                 apply_bfp=False, bfp_block_size=16, bfp_mantissa_bits=4,
                 precision_policy: "str|PrecisionPolicy|None"=None, backend: str = "custom",
                 rope_cache_dtype: "torch.dtype|None" = None,
                 clone_compute_dtype: "torch.dtype|None" = None,
                 softmax_fp32: bool = True,
                 dynamic_mix_ratio: float = 0.0):
        """Initializes the LlamaMyModel instance.

        Args:
            model_name (str, optional): The name of the pretrained Hugging Face model.
                Defaults to "TinyLlama/TinyLlama_v1.1".
            device (str, optional): The device to load the model onto ('cpu' or 'cuda').
                Defaults to 'cpu'.
            stop_criteria (callable, optional): A function for stopping text generation.
                Defaults to None.
            dtype (torch.dtype, optional): The primary data type for model parameters.
                Defaults to torch.float32.
            apply_bfp (bool, optional): If True, applies Block Floating-Point quantization
                to the model weights. Defaults to False.
            bfp_block_size (int, optional): The block size for BFP quantization. Defaults to 16.
            bfp_mantissa_bits (int, optional): The number of mantissa bits for BFP quantization.
                Defaults to 4.
            precision_policy (str or PrecisionPolicy, optional): The precision policy to apply.
                Can be a string identifier or a PrecisionPolicy object. Defaults to None.
            backend (str, optional): The execution backend ('custom', 'huggingface', 'clone').
                Defaults to "custom".
            rope_cache_dtype (torch.dtype, optional): The dtype for the RoPE cache in the
                'clone' backend. If None, defaults to the main `dtype`.
            clone_compute_dtype (torch.dtype, optional): The computation dtype for the 'clone'
                backend. If None, defaults to the main `dtype`.
            softmax_fp32 (bool, optional): Whether to cast softmax to float32 for stability.
                Defaults to True.
            dynamic_mix_ratio (float, optional): The ratio of channels to use high precision for.
                Defaults to 0.0.
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        self.model.eval()
        self.backend = backend  # "custom" (default) or "huggingface"
        # For clone backend: default to dtype for both compute and rope cache to match HF behavior
        # (minimizes dtype conversions that introduce rounding errors)
        self.clone_compute_dtype = clone_compute_dtype if clone_compute_dtype is not None else dtype
        self.softmax_fp32 = softmax_fp32
        # precision policy
        self.precision_policy = resolve_policy(precision_policy)
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.num_kv_heads = self.model.config.num_key_value_heads
        
        # Initialize RoPE for clone backend
        # We allow selecting a separate dtype for the precomputed sin/cos cache (rope_cache_dtype).
        # If not provided, default to the model's main dtyp        python scripts/sweep_precision_configs.py        python scripts/sweep_precision_configs.pye for consistency; for strict HF parity use torch.float32.
        self.rope_cache_dtype = rope_cache_dtype if rope_cache_dtype is not None else self.dtype
        if backend == 'clone':
            hidden_size = self.model.config.hidden_size
            self.head_dim = hidden_size // self.num_heads
            self.rope_emb = HFRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.model.config.max_position_embeddings,
                base=self.model.config.rope_theta if hasattr(self.model.config, 'rope_theta') else 10000.0,
                device=device,
                cache_dtype=self.rope_cache_dtype,
            )
        
        self.attention_blocks = []
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            
            # Capture original weights before quantization
            wq_orig = layer.self_attn.q_proj.weight.data.clone()
            wk_orig = layer.self_attn.k_proj.weight.data.clone()
            wv_orig = layer.self_attn.v_proj.weight.data.clone()
            wo_orig = layer.self_attn.o_proj.weight.data.clone()
            w_gate_orig = layer.mlp.gate_proj.weight.data.clone()
            w_up_orig = layer.mlp.up_proj.weight.data.clone()
            w_down_orig = layer.mlp.down_proj.weight.data.clone()

            if apply_bfp:
                layer.input_layernorm.weight.data = block_floating_point_quantize(layer.input_layernorm.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.self_attn.q_proj.weight.data = block_floating_point_quantize(layer.self_attn.q_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.self_attn.k_proj.weight.data = block_floating_point_quantize(layer.self_attn.k_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.self_attn.v_proj.weight.data = block_floating_point_quantize(layer.self_attn.v_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.self_attn.o_proj.weight.data = block_floating_point_quantize(layer.self_attn.o_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.post_attention_layernorm.weight.data = block_floating_point_quantize(layer.post_attention_layernorm.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.mlp.gate_proj.weight.data = block_floating_point_quantize(layer.mlp.gate_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.mlp.up_proj.weight.data = block_floating_point_quantize(layer.mlp.up_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
                layer.mlp.down_proj.weight.data = block_floating_point_quantize(layer.mlp.down_proj.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
            params = {
                'norm1_weight': layer.input_layernorm.weight.to(device).to(dtype),
                'rms_eps': float(self.model.config.rms_norm_eps) if hasattr(self.model.config, 'rms_norm_eps') else 1e-6,
                'wq': layer.self_attn.q_proj.weight.to(device).to(dtype),
                'wq_bf16': wq_orig.to(device).to(dtype),
                'wk': layer.self_attn.k_proj.weight.to(device).to(dtype),
                'wk_bf16': wk_orig.to(device).to(dtype),
                'wv': layer.self_attn.v_proj.weight.to(device).to(dtype),
                'wv_bf16': wv_orig.to(device).to(dtype),
                'wo': layer.self_attn.o_proj.weight.to(device).to(dtype),
                'wo_bf16': wo_orig.to(device).to(dtype),
                'norm2_weight': layer.post_attention_layernorm.weight.to(device).to(dtype),
                'w_gate': layer.mlp.gate_proj.weight.to(device).to(dtype),
                'w_gate_bf16': w_gate_orig.to(device).to(dtype),
                'w_up': layer.mlp.up_proj.weight.to(device).to(dtype),
                'w_up_bf16': w_up_orig.to(device).to(dtype),
                'w_down': layer.mlp.down_proj.weight.to(device).to(dtype),
                'w_down_bf16': w_down_orig.to(device).to(dtype),
                'precision_policy': self.precision_policy,
                'dynamic_mix_ratio': dynamic_mix_ratio,
            }
            self.attention_blocks.append(transfomer_block_with_kv_cache(params, self.num_heads, self.num_kv_heads, device=device))
        self.stop_criteria = stop_criteria

        if apply_bfp:
            self.model.model.norm.weight.data = block_floating_point_quantize(self.model.model.norm.weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
            self.model.get_input_embeddings().weight.data = block_floating_point_quantize(self.model.get_input_embeddings().weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
            self.model.get_output_embeddings().weight.data = block_floating_point_quantize(self.model.get_output_embeddings().weight.data, block_size=bfp_block_size, mantissa_bits=bfp_mantissa_bits)
            
    def single_step(self, inputs, return_latents=False):
        """Performs a single forward pass through the model.

        This method routes the input through the selected backend ('custom', 'huggingface',
        or 'clone') to compute the logits. It is the core inference function for processing
        a batch of token IDs. The method can optionally return intermediate hidden states
        for debugging and analysis.

        Args:
            inputs (dict): A dictionary containing input tensors. Must include 'input_ids',
                which is a tensor of token IDs with shape (batch_size, sequence_length).
            return_latents (bool, optional): If True, the method returns a tuple containing
                the logits and a list of hidden states from each transformer layer.
                Defaults to False.

        Returns:
            torch.Tensor or tuple:
            - If `return_latents` is False, returns the output logits tensor of shape
              (batch_size, sequence_length, vocab_size).
            - If `return_latents` is True, returns a tuple `(logits, latents)`, where
              `latents` is a list of hidden state tensors.
        """
        # Optional exact HF execution path for bit-exact parity
        if getattr(self, 'backend', 'custom') in ('huggingface','clone'):
            # We deliberately branch early for two special backends:
            # 1) "huggingface"  -> Call the original HF model forward for a fully trusted reference implementation.
            # 2) "clone"        -> Reproduce HF layer-by-layer with our own lightweight functional clones for debugging/parity.
            # If backend is neither, we fall through to the custom implementation below.
            with torch.no_grad():  # Inference-only: disable autograd to save memory & time.
                if self.backend == 'huggingface':
                    # HUGGINGFACE PATH ----------------------------------------------------
                    # Directly invoke HF's forward.
                    # If the caller wants latent (hidden) states, we request them via output_hidden_states=True.
                    if return_latents:
                        out = self.model(
                            input_ids=inputs['input_ids'],              # Token IDs batch [B, T]
                            attention_mask=inputs.get('attention_mask', None),  # Optional mask if provided
                            output_hidden_states=True,                  # Ask HF to return all intermediate hidden states
                        )
                        # out.hidden_states is a tuple; convert to list for consistency with our own format.
                        return out.logits, list(out.hidden_states)
                    else:
                        # Standard forward: only logits are needed.
                        out = self.model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs.get('attention_mask', None),
                        )
                        return out.logits
                else:  # CLONE PATH --------------------------------------------------------
                    # We manually perform the embedding lookup then reconstruct each layer's parameters
                    # into plain tensors so that clone_forward_all can execute a functional forward.
                    x = self.model.get_input_embeddings()(inputs['input_ids'])  # Shape: [B, T, hidden_size]
                    # Ensure embedding output matches model dtype (HF does this automatically)
                    x = x.to(self.dtype)
                    layers_params = []  # Will hold per-layer dicts of weights in the expected functional format.
                    for layer in self.model.model.layers:
                        # Extract all weights needed for attention + MLP + layer norms; cast to desired dtype & device.
                        lp = {
                            'norm1_weight': layer.input_layernorm.weight,
                            'norm2_weight': layer.post_attention_layernorm.weight,
                            'wq': layer.self_attn.q_proj.weight,
                            'wk': layer.self_attn.k_proj.weight,
                            'wv': layer.self_attn.v_proj.weight,
                            'wo': layer.self_attn.o_proj.weight,
                            'w_gate': layer.mlp.gate_proj.weight,
                            'w_up': layer.mlp.up_proj.weight,
                            'w_down': layer.mlp.down_proj.weight,
                        }
                        layers_params.append(lp)
                    # RMSNorm epsilon: use HF config if present, otherwise default to 1e-6.
                    rms_eps = float(self.model.config.rms_norm_eps) if hasattr(self.model.config, 'rms_norm_eps') else 1e-6
                    # Functional forward over all layers (attention + MLP). Returns final hidden states and list of per-layer latents.
                    # Functional forward with configurable RoPE cache precision.
                    x, latents = clone_forward_all(
                        x,
                        layers_params,
                        self.num_heads,
                        self.num_kv_heads,
                        rms_eps,
                        rope_cache_dtype=self.rope_cache_dtype,
                        compute_dtype=self.clone_compute_dtype,
                        softmax_fp32=self.softmax_fp32,
                    )
                    # Cast x back to self.dtype to match final norm and lm_head weights
                    x = x.to(self.dtype)
                    # Final normalization using the model's last RMSNorm weight.
                    x = rmsnorm(x, self.model.model.norm.weight.to(self.dtype), eps=rms_eps)
                    # Projection to vocabulary logits via lm_head (weight tied to output embeddings).
                    logits = lm_head(x, self.model.get_output_embeddings().weight.to(self.dtype))
                    if return_latents:
                        # Append the final normalized hidden state so caller sees the full chain.
                        latents.append(x)
                        return logits, latents
                    return logits
        # assert torch.device(inputs['input_ids'].device) == self.device, f"Input IDs must be on the same device as the model. Input device: {torch.device(inputs['input_ids'].device)}, Model device: {self.device}"
        
        ################################################
        # 'custom' backend
        ################################################
        if return_latents:
            x_list = []
        
        # embedding
        x = input_embedding(inputs['input_ids'], self.model.get_input_embeddings().weight.to(self.dtype))
        if return_latents:
            x_list.append(x)
        
        # transformer blocks: use KV cache for both prefill and decode
        for layer_idx in range(self.num_layers):
            params = self.attention_blocks[layer_idx].params
            # Always use cache path - it handles both prefill (seq_len > 1) and decode (seq_len == 1)
            x = self.attention_blocks[layer_idx].forward(x, params)
            if return_latents:
                x_list.append(x)

        # final rmsnorm (use HF eps)
        final_eps = float(self.model.config.rms_norm_eps) if hasattr(self.model.config, 'rms_norm_eps') else 1e-6
        x = rmsnorm(x, self.model.model.norm.weight.to(self.dtype), eps=final_eps)
        if return_latents:
            x_list.append(x)

        # lm head - ensure x is in model dtype for final projection
        logits = lm_head(x.to(self.dtype), self.model.get_output_embeddings().weight.to(self.dtype))

        if return_latents:
            return logits, x_list
        else:
            return logits
    
            
    def reset_kv_cache(self):
        """Resets the key-value cache for all attention blocks.

        This method is essential for starting a new, independent generation sequence. It iterates
        through each `transfomer_block_with_kv_cache` in the model and clears its stored
        key and value caches, ensuring that there is no leakage of context from previous
        generations.
        """
        for block in self.attention_blocks:
            block.k_cache = None
            block.v_cache = None
            block.sequence_length = 0
            
    def get_computation_stats(self):
        """Returns the computation statistics from the tracker."""
        return tracker.get_stats()
        
    def reset_computation_stats(self):
        """Resets the computation statistics in the tracker."""
        tracker.reset()
            
    def generate(self, input_text, max_new_tokens=100):
        """Generates a sequence of text given an input prompt.

        This method implements a simple auto-regressive generation loop using greedy decoding.
        It tokenizes the input text, then iteratively calls the model's forward pass to
        predict the next token. The process continues until a stop criterion is met or
        `max_new_tokens` are generated.

        Note:
            This is a basic implementation for demonstration and testing. For advanced use
            cases, more sophisticated sampling methods (e.g., top-k, nucleus sampling)
            and batching would be required.

        Args:
            input_text (str): The prompt to begin generation from.
            max_new_tokens (int, optional): The maximum number of new tokens to generate.
                Defaults to 100.

        Returns:
            torch.Tensor: A tensor of shape (1, sequence_length) containing the token IDs
                of the generated text, including the prompt.
        """
        input_ids = self.tokenizer([input_text], return_tensors="pt").input_ids.to(self.device)
        generated_ids = input_ids.tolist()[0]
        current_input_ids = input_ids
        
        while True:
        
            # embedding
            x = input_embedding(current_input_ids, self.model.get_input_embeddings().weight.to(self.dtype))
            
            # transformer blocks with KV cache
            for layer_idx in range(self.num_layers):
                x = self.attention_blocks[layer_idx].forward(x, self.attention_blocks[layer_idx].params)
                
            # final rmsnorm
            x = rmsnorm(x, self.model.model.norm.weight.to(self.dtype))
            
            # lm head
            logits = lm_head(x, self.model.get_output_embeddings().weight.to(self.dtype))

            # Get the last token's logits
            next_token_logits = logits[:, -1, :]
            
            # Sample the next token (greedy approach for simplicity)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append to generated sequence
            generated_ids.extend(next_token_id.tolist()[0])
            
            # Check for stopping criteria
            if self.stop_criteria(torch.tensor([generated_ids]), None): # scores argument is not used by StopOnTokens
                break
                
            # Set current_input_ids for the next iteration to be just the newly generated token
            current_input_ids = next_token_id
            
            # Limit generation length to avoid infinite loops
            if len(generated_ids) > 100: # Max 100 new tokens
                break

        decoded_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return torch.tensor([generated_ids])
    

def test():
    """Validates the custom Llama model implementation against the HF reference.

    This function performs a forward pass using both the `LlamaMyModel` with the 'custom'
    backend and the original Hugging Face `AutoModelForCausalLM`. It then compares the
    output logits and intermediate hidden states to ensure they are numerically close,
    serving as a regression test for the custom backend's correctness.
    """
    # Step 1: Load the tinyllama model & example input
    example_input = '\n<|user|>:hello</s>\n<|assistant|>:'
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
    model_inputs = tokenizer([example_input], return_tensors="pt").to('cpu')
    
    # run my model    
    my_model = LlamaMyModel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device='cuda',
        backend='custom',
        stop_criteria=StopOnTokens(),
        dtype=torch.float32
    )
    
    my_logits, x_list = my_model.single_step(model_inputs, return_latents=True)
    
    # ground truth generation
    with torch.no_grad():
        ground_truth_outputs = model(**model_inputs, output_hidden_states=True)
        ground_truth_logits = ground_truth_outputs.logits.to(torch.float32)
        ground_truth_latents = ground_truth_outputs.hidden_states
        
    # compare hidden states
    print("Latents from your implementation (last token, first 10):")
    # print(x[0, -1, :10])
    print(x_list[21][0, -1, :10])
    print("\nLatents from Hugging Face model (last token, first 10):")
    print("size of ground_truth_latents: ", len(ground_truth_latents))
    # print(ground_truth_latents)
    print(ground_truth_latents[21][0, -1, :10])
    print(x_list[21][0, -1, :10] / ground_truth_latents[21][0, -1, :10])


    # Step 7: Compare the logits
    print("Logits from your implementation (last token, first 10):")
    print(my_logits[0, -1, :10])
    print("\nLogits from Hugging Face model (last token, first 10):")
    print(ground_truth_logits[0, -1, :10])

    # Check if they are close
    # Using a small tolerance (1e-3 or 1e-4 is standard for fp32)
    are_close = torch.allclose(my_logits, ground_truth_logits, atol=1e-4) 
    print(f"\nAre the logits close? {are_close}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="tinyllama ground truth inference",
        description="Run inference on TinyLlama model in the full precision mode.",
    )
    parser.add_argument(
        "-i", "--input_text", type=str, help="Input text for the model.",
        default='Who is the president of US now?',
    )
    args = parser.parse_args()

    input_text = args.input_text
    
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float32)
    
    my_model = LlamaMyModel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=device,
        backend='custom',
        stop_criteria=StopOnTokens(),
        dtype=torch.float32
    )
    
    outputs = my_model.generate(
        input_formatting([], input_text),
        max_new_tokens=1024
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Input Text:\n", input_formatting([], input_text))
    print("Generated Text:\n", generated_text)