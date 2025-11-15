import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import math
from typing import Optional
from llama_backend.precision_policy import PrecisionPolicy, resolve_policy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


# Helper function for RoPE (Rotary Positional Embeddings)
def __rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

# Helper function for RoPE (Rotary Positional Embeddings)
def apply_rope(x, seq_len, head_dim, start_pos=0, policy: Optional[PrecisionPolicy]=None):
    """HF-compatible RoPE application with HALVED dims (not interleaved).
    x: (B, H, S, D) where D=head_dim.
    
    This matches HuggingFace's rotate_half implementation:
    - First half and second half are rotated separately
    - NOT interleaved even/odd like some other implementations
    """
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

# function: multi-query attention w/ rope
def mqa_rope(i, wq, wk, wv, wo, num_heads, num_kv_heads, get_attn_scores=False, precision_policy: Optional[PrecisionPolicy]=None):
    # i: input tensor (batch_size, seq_len, model_dim)
    # wq, wk, wv: weight matrices for Q, K, V (e.g., [out_dim, in_dim])
    # wo: output weight matrix (model_dim, model_dim)
    
    batch_size, seq_len, model_dim = i.shape
    head_dim = model_dim // num_heads
    num_q_heads = num_heads
    num_kv_heads = num_kv_heads if num_kv_heads is not None else num_q_heads
    
    # FIX 1: Use the correct projection result dimensions for GQA
    # Q projection result: (B, S, model_dim)
    policy = resolve_policy(precision_policy) if precision_policy is not None else PrecisionPolicy.default()
    matmul_dtype = policy.attn_matmul_dtype or i.dtype
    i_mm = i.to(matmul_dtype)
    q_proj = torch.matmul(i_mm, wq.to(matmul_dtype).t())
    k_proj = torch.matmul(i_mm, wk.to(matmul_dtype).t())
    v_proj = torch.matmul(i_mm, wv.to(matmul_dtype).t())

    # view: Splitting into Multiple Heads
    q = q_proj.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
    k = k_proj.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v_proj.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    q = apply_rope(q, seq_len, head_dim, start_pos=0, policy=policy)
    k = apply_rope(k, seq_len, head_dim, start_pos=0, policy=policy)
    
    # Repeat k and v heads if num_q_heads > num_kv_heads (Grouped Query Attention)
    k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
    
    # Attention calculation
    # Compute attention scores in chosen dtype
    attn_scores  = torch.matmul(q.to(matmul_dtype), k.to(matmul_dtype).transpose(-2, -1)) / math.sqrt(head_dim)

    # Apply causal mask
    # The mask should be broadcastable to the batch dimension
    mask = torch.triu(torch.ones(seq_len, seq_len, device=i.device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)
    attn_scores = attn_scores.masked_fill(mask, float('-inf'))

    softmax_dtype = policy.attn_softmax_dtype or attn_scores.dtype
    scores_for_softmax = attn_scores.to(softmax_dtype)
    if policy.stable_softmax:
        scores_for_softmax = scores_for_softmax - scores_for_softmax.max(dim=-1, keepdim=True)[0]
    attn_weights = torch.nn.functional.softmax(scores_for_softmax, dim=-1).to(matmul_dtype)
    attn_output  = torch.matmul(attn_weights.to(matmul_dtype), v.to(matmul_dtype))
    
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
    o = torch.matmul(attn_output, wo.to(matmul_dtype).t()) # Final output
    if get_attn_scores:
        return o, attn_scores
    else:
        return o

# function: feed-forward network
def ffn_SwiGLU(i, w_gate, w_up, w_down, precision_policy: Optional[PrecisionPolicy]=None):
    # i: input tensor (batch_size, seq_len, model_dim)
    # w_gate, w_up, w_down: weight matrices (e.g., [out_dim, in_dim])
    
    policy = resolve_policy(precision_policy) if precision_policy is not None else PrecisionPolicy.default()
    ffn_dtype = policy.ffn_matmul_dtype or i.dtype
    
    # Step 1: Linear transformation for gate and up projections
    i_ffn = i.to(ffn_dtype)
    gate = torch.matmul(i_ffn, w_gate.to(ffn_dtype).t())
    up = torch.matmul(i_ffn, w_up.to(ffn_dtype).t())
    
    # Step 2: Swish activation on the gate (SiLU is Swish-1)
    gate = torch.nn.functional.silu(gate)
    
    # Step 3: Element-wise multiplication of gate and up projections
    hidden_states = gate * up
    
    # Step 4: Down projection
    output = torch.matmul(hidden_states, w_down.to(ffn_dtype).t())
    return output.to(i.dtype)

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

def lm_head(hidden_states, wte):
    # hidden_states: (batch_size, seq_len, model_dim)
    # wte: (vocab_size, model_dim)
    return torch.matmul(hidden_states, wte.t())

# function: forward pass of a tinyllama transformer block
def transformer_block(x, params, num_heads, num_kv_heads, get_attention_scores=False):
    
    # LayerNorm 1
    # The norm has a learnable weight parameter
    x_norm = rmsnorm(x, params['norm1_weight'], eps=params.get('rms_eps', 1e-5))
    
    # Multi-Query Attention with RoPE
    attn_output, attn_scores = mqa_rope(
        x_norm,
        params['wq'], params['wk'], params['wv'], params['wo'],
        num_heads, num_kv_heads,
        get_attn_scores=True,
        precision_policy=params.get('precision_policy')
    )
    
    # Residual connection 1
    x = x + attn_output
    
    # LayerNorm 2
    x_norm = rmsnorm(x, params['norm2_weight'], eps=params.get('rms_eps', 1e-5))
    
    # Feed-Forward Network with SwiGLU activation
    ffn_output = ffn_SwiGLU(
        x_norm,
        params['w_gate'], params['w_up'], params['w_down'],
        precision_policy=params.get('precision_policy')
    )
    
    # Residual connection 2
    x = x + ffn_output
    
    if get_attention_scores:
        return x, attn_scores
    else:
        return x
    
class transfomer_block_with_kv_cache:
    def __init__(self, params, num_heads, num_kv_heads, get_attention_scores=False, device='cpu'):
        self.device = device # Assuming all params are on the same device
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.empty(0, device=self.device) # Initialize as empty tensor on the correct device
        self.v_cache = torch.empty(0, device=self.device) # Initialize as empty tensor on the correct device
        self.sequence_length = 0
        self.params = params
        self.get_attention_scores = get_attention_scores
        
        
    def forward(self, x, params):
        # LayerNorm 1
        x_norm = rmsnorm(x, params['norm1_weight'].to(self.device), eps=params.get('rms_eps', 1e-5))
        batch_size, seq_len, model_dim = x.shape
        head_dim = model_dim // self.num_heads
        # Resolve precision policy
        policy = resolve_policy(params.get('precision_policy')) if params.get('precision_policy') is not None else PrecisionPolicy.default()
        mm_dtype = policy.attn_matmul_dtype or x_norm.dtype
        x_mm = x_norm.to(mm_dtype)
        # Projections
        q_proj = torch.matmul(x_mm, params['wq'].to(mm_dtype).t())
        k_proj = torch.matmul(x_mm, params['wk'].to(mm_dtype).t())
        v_proj = torch.matmul(x_mm, params['wv'].to(mm_dtype).t())
        # Reshape heads
        q = q_proj.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k_proj.view(batch_size, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)
        v = v_proj.view(batch_size, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)
        # RoPE
        q = apply_rope(q, seq_len, head_dim, start_pos=self.sequence_length, policy=policy)
        k = apply_rope(k, seq_len, head_dim, start_pos=self.sequence_length, policy=policy)
        # KV cache update
        if self.sequence_length == 0:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache.to(self.device), k], dim=-2)
            self.v_cache = torch.cat([self.v_cache.to(self.device), v], dim=-2)
        self.sequence_length += seq_len
        # Expand kv heads
        k_expanded = self.k_cache.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v_expanded = self.v_cache.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        # Attention scores
        attn_scores = torch.matmul(q.to(mm_dtype), k_expanded.to(mm_dtype).transpose(-2, -1)) / math.sqrt(head_dim)
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, self.sequence_length, device=self.device, dtype=torch.bool), diagonal=1).unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        softmax_dtype = policy.attn_softmax_dtype or attn_scores.dtype
        scores_for_softmax = attn_scores.to(softmax_dtype)
        if policy.stable_softmax:
            scores_for_softmax = scores_for_softmax - scores_for_softmax.max(dim=-1, keepdim=True)[0]
        attn_weights = torch.nn.functional.softmax(scores_for_softmax, dim=-1).to(mm_dtype)
        attn_output = torch.matmul(attn_weights, v_expanded.to(mm_dtype))
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        o = torch.matmul(attn_output.to(mm_dtype), params['wo'].to(mm_dtype).t()).to(self.device)
        x = x + o
        # LayerNorm 2
        x_norm = rmsnorm(x, params['norm2_weight'].to(self.device), eps=params.get('rms_eps', 1e-5))
        # Feed-forward with policy
        ffn_out = ffn_SwiGLU(x_norm, params['w_gate'], params['w_up'], params['w_down'], precision_policy=policy)
        x = x + ffn_out
        if self.get_attention_scores:
            return x, attn_scores
        return x

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
            'w_down': layer.mlp.down_proj.weight.to(dtype)
        }
        x = transformer_block(x, params, model.config.num_attention_heads, model.config.num_key_value_heads)
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