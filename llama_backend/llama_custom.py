"""A custom Llama model for flexible backend and precision control."""
import torch
import torch.nn as nn
import math
import sys
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, TextIteratorStreamer

# Add repository root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# custom imports
from llama_backend.custom.llama_backend import (
    DEFAULT_BACKEND, 
    input_embedding, 
    rmsnorm, 
    lm_head, 
    LlamaTransformerBlock
)
from llama_backend.custom.precision_policy import PrecisionPolicy, resolve_policy
from llama_backend.clone.hf_rope import HFRotaryEmbedding
from llama_backend.clone.clone_backend import clone_decoder_layer, clone_rmsnorm
from llama_backend.clone.hf_clone import clone_forward_all
from llama_backend.utils import StopOnTokens, input_formatting
from custom_model_only_quantize.block_floating_point.block_quantization import block_floating_point_quantize

class CustomLlamaModel(nn.Module):
    """A custom Llama model implementation for flexible backend and precision control."""
    def __init__(self, model_name="TinyLlama/TinyLlama_v1.1", device='cpu', stop_criteria=None, dtype=torch.float32,
                 apply_bfp=False, bfp_block_size=16, bfp_mantissa_bits=4,
                 precision_policy: "str|PrecisionPolicy|None"=None, backend: str = "custom",
                 rope_cache_dtype: "torch.dtype|None" = None,
                 clone_compute_dtype: "torch.dtype|None" = None,
                 softmax_fp32: bool = True,
                 quantization_config=None):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        
        # Apply runtime quantization if config is provided
        if quantization_config:
            from custom_model_only_quantize.runtime_quantize import apply_runtime_quantization
            print(f"Applying runtime quantization to CustomLlamaModel base: {quantization_config}")
            apply_runtime_quantization(
                self.model,
                method=quantization_config.method,
                block_height=quantization_config.block_height,
                block_width=quantization_config.block_width,
                mantissa_bits=quantization_config.mantissa_bits,
                top_k=quantization_config.top_k,
                activations=quantization_config.activations
            )
            
        self.model.eval()
        self.backend = backend
        self.clone_compute_dtype = clone_compute_dtype if clone_compute_dtype is not None else dtype
        self.softmax_fp32 = softmax_fp32
        self.precision_policy = resolve_policy(precision_policy)
        
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.num_kv_heads = self.model.config.num_key_value_heads
        
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
        
        self.attention_blocks = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            
            # Note: If runtime quantization was applied above, these weights are already quantized.
            # We just need to ensure we grab the data from the layer correctly.
            
            params = {
                'norm1_weight': layer.input_layernorm.weight.to(device).to(dtype),
                'rms_eps': float(self.model.config.rms_norm_eps) if hasattr(self.model.config, 'rms_norm_eps') else 1e-6,
                'wq': layer.self_attn.q_proj.weight.to(device).to(dtype),
                'wk': layer.self_attn.k_proj.weight.to(device).to(dtype),
                'wv': layer.self_attn.v_proj.weight.to(device).to(dtype),
                'wo': layer.self_attn.o_proj.weight.to(device).to(dtype),
                'norm2_weight': layer.post_attention_layernorm.weight.to(device).to(dtype),
                'w_gate': layer.mlp.gate_proj.weight.to(device).to(dtype),
                'w_up': layer.mlp.up_proj.weight.to(device).to(dtype),
                'w_down': layer.mlp.down_proj.weight.to(device).to(dtype),
                'precision_policy': self.precision_policy,
            }
            block = LlamaTransformerBlock(self.num_heads, self.num_kv_heads, backend=DEFAULT_BACKEND)
            block.params = params
            self.attention_blocks.append(block)
            
        self.stop_criteria = stop_criteria

        # Note: BFP for embeddings/norm also handled by runtime_quantize if applied.
            
    def single_step(self, inputs, return_latents=False):
        """Standard Llama forward pass logic."""
        if getattr(self, 'backend', 'custom') in ('huggingface','clone'):
            with torch.no_grad():
                if self.backend == 'huggingface':
                    if return_latents:
                        out = self.model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask', None), output_hidden_states=True)
                        return out.logits, list(out.hidden_states)
                    else:
                        out = self.model(input_ids=inputs['input_ids'], attention_mask=inputs.get('attention_mask', None))
                        return out.logits
                else:  # CLONE PATH
                    x = self.model.get_input_embeddings()(inputs['input_ids']).to(self.dtype)
                    layers_params = []
                    for layer in self.model.model.layers:
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
                    rms_eps = float(self.model.config.rms_norm_eps) if hasattr(self.model.config, 'rms_norm_eps') else 1e-6
                    x, latents = clone_forward_all(x, layers_params, self.num_heads, self.num_kv_heads, rms_eps, rope_cache_dtype=self.rope_cache_dtype, compute_dtype=self.clone_compute_dtype, softmax_fp32=self.softmax_fp32)
                    x = x.to(self.dtype)
                    x = rmsnorm(x, self.model.model.norm.weight.to(self.dtype), eps=rms_eps)
                    logits = lm_head(x, self.model.get_output_embeddings().weight.to(self.dtype))
                    if return_latents:
                        latents.append(x)
                        return logits, latents
                    return logits

        # CUSTOM PATH
        if return_latents:
            x_list = []
        
        x = input_embedding(inputs['input_ids'], self.model.get_input_embeddings().weight.to(self.dtype))
        if return_latents:
            x_list.append(x)
        
        for layer_idx in range(self.num_layers):
            params = self.attention_blocks[layer_idx].params
            x = self.attention_blocks[layer_idx].forward(x, params)
            if return_latents:
                x_list.append(x)

        final_eps = float(self.model.config.rms_norm_eps) if hasattr(self.model.config, 'rms_norm_eps') else 1e-6
        x = rmsnorm(x, self.model.model.norm.weight.to(self.dtype), eps=final_eps)
        if return_latents:
            x_list.append(x)

        logits = lm_head(x.to(self.dtype), self.model.get_output_embeddings().weight.to(self.dtype))
        if return_latents:
            return logits, x_list
        else:
            return logits
    
    def forward(self, inputs, return_latents=False):
        """Standard PyTorch entry point."""
        return self.single_step(inputs, return_latents)
    
    def reset_kv_cache(self):
        """Resets the key-value cache for all attention blocks."""
        for block in self.attention_blocks:
            block.attention.kv_cache = None

    def generate(self, input_text, max_new_tokens=100):
        """Greedy text generation."""
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        self.reset_kv_cache()
        generated_ids = input_ids[0].tolist()
        current_input_ids = input_ids
        
        while True:
            inputs = {'input_ids': current_input_ids}
            logits = self.forward(inputs)
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            generated_ids.append(next_token_id.item())
            if next_token_id.item() == self.tokenizer.eos_token_id or len(generated_ids) >= input_ids.shape[1] + max_new_tokens:
                break
            
            current_input_ids = next_token_id.unsqueeze(0)
            if self.stop_criteria and self.stop_criteria(torch.tensor([generated_ids]), None):
                break
                
        return torch.tensor([generated_ids])