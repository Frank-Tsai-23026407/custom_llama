import sys
from pathlib import Path

# Add project root directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import math
from typing import Optional, Protocol, Any
from llama_backend.custom.precision_policy import PrecisionPolicy, resolve_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

# --- Section 1: Math/Functional Blocks (Standalone Functions) ---
# Helper function for RoPE (Rotary Positional Embeddings)
def __rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# Helper function for RoPE (Rotary Positional Embeddings)
def apply_rope(x, seq_len, head_dim, start_pos=0, policy: Optional[PrecisionPolicy]=None):
    """HF-compatible RoPE application with HALVED dims (not interleaved)."""
    input_dtype = x.dtype
    policy = resolve_policy(policy) if policy is not None else PrecisionPolicy.default()
    rope_compute = policy.rope_compute_dtype or torch.float32
    x = x.to(rope_compute)

    rope_theta = 10000.0
    positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.float, device=x.device)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float) / head_dim))
    freqs = torch.outer(positions, inv_freq)  # (S, D/2)
    # HF uses cat instead of stack for the full embedding
    emb = torch.cat((freqs, freqs), dim=-1)  # (S, D)
    cos = torch.cos(emb).unsqueeze(0).unsqueeze(0)  # (1,1,S,D)
    sin = torch.sin(emb).unsqueeze(0).unsqueeze(0)  # (1,1,S,D)

    # Halved RoPE: rotate_half implementation matching HF
    x1 = x[..., : x.shape[-1] // 2]  # First half
    x2 = x[..., x.shape[-1] // 2 :]  # Second half
    x_rotated = torch.cat((-x2, x1), dim=-1)  # Rotate: (-x2, x1)
    
    # Apply rotation: x * cos + rotate_half(x) * sin
    x_out = (x * cos) + (x_rotated * sin)
    return x_out.to(input_dtype)


# --- Section 2: Backend Definition (The "Strategy" Pattern) ---
class LlamaBackend:
    """
    This class defines how low-level operations are performed.
    By subclassing this, you can swap out GEMM, RoPE, or RMSNorm 
    with special research implementations (e.g., custom CUDA kernels).
    """
    def gemm(self, input: torch.Tensor, weight: torch.Tensor, transpose_b: bool = True, policy: Optional[PrecisionPolicy] = None) -> torch.Tensor:
        """
        Generic matrix multiplication. 
        Args:
            transpose_b: If True, transposes the last two dimensions of weight (standard for Linear [out, in]).
        """
        if transpose_b:
            weight = weight.transpose(-2, -1)
        return torch.matmul(input, weight)

    def rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32) 
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return weight * x.to(input_dtype)

    def softmax(self, x: torch.Tensor, dim: int = -1, policy: Optional[PrecisionPolicy] = None) -> torch.Tensor:
        policy = resolve_policy(policy)
        if policy.stable_softmax:
            x = x - x.max(dim=dim, keepdim=True)[0]
        return torch.nn.functional.softmax(x.to(torch.float32), dim=dim).to(x.dtype)

    def apply_rope(self, x: torch.Tensor, seq_len: int, head_dim: int, start_pos: int = 0, policy: Optional[PrecisionPolicy] = None) -> torch.Tensor:
        return apply_rope(x, seq_len, head_dim, start_pos, policy)

    def cat(self, tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
        """Concatenate tensors. Can be overridden for specialized memory management."""
        return torch.cat(tensors, dim=dim)

# Global default backend
DEFAULT_BACKEND = LlamaBackend()



# --- Section 3: Structural Modules (Classes) ---
class LlamaAttention(nn.Module):
    """
    Multi-Query Attention (MQA) component.
    Takes backend in __init__ for easy operation swapping.
    """
    def __init__(self, num_heads: int, num_kv_heads: int, backend: LlamaBackend = DEFAULT_BACKEND):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.backend = backend
        
        # KV Cache state
        self.k_cache = torch.empty(0)
        self.v_cache = torch.empty(0)
        self.kv_seq_len = 0

    def reset_cache(self):
        self.k_cache = torch.empty(0)
        self.v_cache = torch.empty(0)
        self.kv_seq_len = 0

    def forward(self, x, params, precision_policy: Optional[PrecisionPolicy] = None, use_cache=False):
        batch_size, seq_len, model_dim = x.shape
        head_dim = model_dim // self.num_heads
        policy = resolve_policy(precision_policy)
        mm_dtype = policy.attn_matmul_dtype or x.dtype
        
        # 1. Projections
        x_mm = x.to(mm_dtype)
        q = self.backend.gemm(x_mm, params['wq'].to(mm_dtype), policy=policy)
        k = self.backend.gemm(x_mm, params['wk'].to(mm_dtype), policy=policy)
        v = self.backend.gemm(x_mm, params['wv'].to(mm_dtype), policy=policy)
        
        # 2. Reshape Heads
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)
        
        # 3. Apply RoPE
        start_pos = self.kv_seq_len if use_cache else 0
        q = self.backend.apply_rope(q, seq_len, head_dim, start_pos=start_pos, policy=policy)
        k = self.backend.apply_rope(k, seq_len, head_dim, start_pos=start_pos, policy=policy)
        
        # 4. KV Cache Processing
        if use_cache:
            if self.kv_seq_len == 0:
                self.k_cache, self.v_cache = k, v
            else:
                self.k_cache = self.backend.cat([self.k_cache.to(k.device), k], dim=-2)
                self.v_cache = self.backend.cat([self.v_cache.to(v.device), v], dim=-2)
            self.kv_seq_len += seq_len
            k_final, v_final = self.k_cache, self.v_cache
        else:
            k_final, v_final = k, v
            
        # 5. Grouped Query / Multi-Query Expansion
        k_final = k_final.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v_final = v_final.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        
        # 6. Attention core: Q @ K^T
        # q: [B, H, T, D], k_final: [B, H, T_cache, D]
        # We need q @ k_final.T -> [B, H, T, T_cache]
        attn_scores = self.backend.gemm(q.to(mm_dtype), k_final.to(mm_dtype), transpose_b=True, policy=policy) / math.sqrt(head_dim)
        
        if seq_len > 1: # Apply causal mask for prompt processing
            mask = torch.triu(torch.ones(seq_len, k_final.size(-2), device=x.device, dtype=torch.bool), diagonal=1+start_pos).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
            
        attn_weights = self.backend.softmax(attn_scores, dim=-1, policy=policy).to(mm_dtype)
        # Weights @ V: [B, H, T, T_cache] @ [B, H, T_cache, D] -> [B, H, T, D]
        # No transpose needed for V here
        attn_output = self.backend.gemm(attn_weights, v_final.to(mm_dtype), transpose_b=False, policy=policy)
        
        # 7. Output Projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        return self.backend.gemm(attn_output, params['wo'].to(mm_dtype), policy=policy)

class LlamaMLP(nn.Module):
    """
    SwiGLU Feed-Forward Network component.
    """
    def __init__(self, backend: LlamaBackend = DEFAULT_BACKEND):
        super().__init__()
        self.backend = backend

    def forward(self, x, params, precision_policy: Optional[PrecisionPolicy] = None):
        policy = resolve_policy(precision_policy)
        ffn_dtype = policy.ffn_matmul_dtype or x.dtype
        
        i_ffn = x.to(ffn_dtype)
        gate = self.backend.gemm(i_ffn, params['w_gate'].to(ffn_dtype), policy=policy)
        up = self.backend.gemm(i_ffn, params['w_up'].to(ffn_dtype), policy=policy)
        
        # SwiGLU: SiLU(gate) * up
        hidden_states = torch.nn.functional.silu(gate) * up
        
        return self.backend.gemm(hidden_states, params['w_down'].to(ffn_dtype), policy=policy)

class LlamaTransformerBlock(nn.Module):
    """
    A single Transformer Block tying together Norm, Attention, and MLP.
    """
    def __init__(self, num_heads, num_kv_heads, backend: LlamaBackend = DEFAULT_BACKEND):
        super().__init__()
        self.attention = LlamaAttention(num_heads, num_kv_heads, backend)
        self.mlp = LlamaMLP(backend)
        self.backend = backend

    def forward(self, x, params, use_cache=False):
        # 1. Attention path
        x_norm = self.backend.rmsnorm(x, params['norm1_weight'], eps=params.get('rms_eps', 1e-5))
        x = x + self.attention(x_norm, params, precision_policy=params.get('precision_policy'), use_cache=use_cache)
        
        # 2. MLP path
        x_norm = self.backend.rmsnorm(x, params['norm2_weight'], eps=params.get('rms_eps', 1e-5))
        x = x + self.mlp(x_norm, params, precision_policy=params.get('precision_policy'))
        
        return x

# function: rmsnorm
def rmsnorm(x, weight, eps=1e-5):
    # x: input tensor (batch_size, seq_len, model_dim)
    # weight: (model_dim)
    input_dtype = x.dtype
    # The Llama implementation uses float32 for variance calculation
    x = x.to(torch.float32) 
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    # The final result is returned in the original dtype
    return weight * x.to(input_dtype)

# function: input embedding
def input_embedding(input_ids, wte):
    # input_ids: (batch_size, seq_len)
    # wte: (vocab_size, model_dim)
    return torch.nn.functional.embedding(input_ids, wte)

# function: lm head
def lm_head(hidden_states, wte):
    # hidden_states: (batch_size, seq_len, model_dim)
    # wte: (vocab_size, model_dim)
    return torch.matmul(hidden_states, wte.t())


# main 
def main():
    # Step 1: Load the tinyllama model & example input
    device = 'cpu'
    # Use torch.float32 for consistency and to avoid bfloat16 issues if not using a GPU
    dtype = torch.float32 
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=dtype)

    example_input = '\n<|user|>:hello</s>\n<|assistant|>:'
    model_inputs = tokenizer([example_input], return_tensors="pt").to(device)
    x_list = [] # To store intermediate latents for comparison later


    # Step 2: embedding
    x = input_embedding(model_inputs['input_ids'], model.get_input_embeddings().weight.to(dtype))
    x_init = x.clone() # Save initial embeddings for comparison later
    x_list.append(x) # Store initial embeddings for comparison later
    
    # Step 3: transformer blocks
    num_layers = model.config.num_hidden_layers
    # Initialize our block classes
    blocks = [
        LlamaTransformerBlock(model.config.num_attention_heads, model.config.num_key_value_heads)
        for _ in range(num_layers)
    ]
    
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        params = {
            'norm1_weight': layer.input_layernorm.weight.to(dtype),
            'wq': layer.self_attn.q_proj.weight.to(dtype),
            'wk': layer.self_attn.k_proj.weight.to(dtype),
            'wv': layer.self_attn.v_proj.weight.to(dtype),
            'wo': layer.self_attn.o_proj.weight.to(dtype),
            'norm2_weight': layer.post_attention_layernorm.weight.to(dtype),
            'w_gate': layer.mlp.gate_proj.weight.to(dtype),
            'w_up': layer.mlp.up_proj.weight.to(dtype),
            'w_down': layer.mlp.down_proj.weight.to(dtype),
            'rms_eps': model.config.rms_norm_eps
        }
        x = blocks[layer_idx](x, params)
        x_list.append(x) # Store intermediate latents for comparison later
    
    # step 4: final rmsnorm
    x = rmsnorm(x, model.model.norm.weight.to(dtype))
    x_list.append(x)
    
    # step 5: lm head [important: the lm_head do not use the same weights as the input embedding]
    my_logits = lm_head(x, model.get_output_embeddings().weight.to(dtype))
    
    # step 6: Get ground truth logits from the original model
    with torch.no_grad():
        ground_truth_outputs = model(**model_inputs, output_hidden_states=True)
        ground_truth_logits = ground_truth_outputs.logits.to(dtype)
        ground_truth_latents = ground_truth_outputs.hidden_states # Input embeddings
        
    # compare hidden states
    print("Latents from your implementation (last token, first 10):")
    # print(x[0, -1, :10])
    print(x_list[23][0, -1, :10])
    print("\nLatents from Hugging Face model (last token, first 10):")
    print("size of ground_truth_latents: ", len(ground_truth_latents))
    # print(ground_truth_latents)
    print(ground_truth_latents[22][0, -1, :10])
    print(x_list[22][0, -1, :10] / ground_truth_latents[22][0, -1, :10])


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
    main()