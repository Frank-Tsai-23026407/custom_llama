import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
import torch.nn.functional as F
import os
import sys

# Ensure repository root is on sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llama_backend.llama_my import LlamaMyModel

# optional import for embedding helper
try:
    from llama_backend.custom.plain_script import input_embedding, rmsnorm, apply_rope, ffn_SwiGLU
except Exception:
    input_embedding = None
    rmsnorm = None
    apply_rope = None
    ffn_SwiGLU = None

def to_cpu_f32(tensor):
    return tensor.detach().cpu().to(torch.float32)

def force_hf_eager_attn(model):
    # Try to switch HF model attention implementation to 'eager' so output_attentions works
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
        else:
            # fallback: set config flag (may not always work at runtime)
            if hasattr(model, "config"):
                setattr(model.config, "attn_implementation", "eager")
    except Exception:
        pass


def try_get_hf_attentions(hflm, input_ids, attention_mask):
    # try standard HF forward with attentions
    try:
        force_hf_eager_attn(hflm.model)
        out = hflm.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, return_dict=True)
        if hasattr(out, "attentions") and out.attentions is not None:
            return out.attentions  # tuple: (layer0_attn, layer1_attn, ...)
    except Exception:
        pass
    # fallback to internal call
    try:
        force_hf_eager_attn(hflm.model)
        out = hflm._model_call(input_ids=input_ids, attn_mask=attention_mask, output_attentions=True)
        if hasattr(out, "attentions"):
            return out.attentions
    except Exception:
        pass
    return None

def try_get_my_attentions(my, inputs):
    # try common options from llama_my interface
    # 1) return_attentions flag
    try:
        ret = my.single_step(inputs, return_attentions=True)
        # expected structure: (logits, attns) or attns directly
        if isinstance(ret, tuple) and len(ret) >= 2:
            return ret[1]
        return ret
    except Exception:
        pass
    # 2) return_latents and inspect latents for attn-like tensors
    try:
        ret = my.single_step(inputs, return_latents=True)
        if isinstance(ret, tuple) and len(ret) == 2:
            _, latents = ret
            # guess: latents may be list per-layer; try to find attention matrices by shape
            attns = []
            for item in latents:
                if isinstance(item, torch.Tensor) and item.dim() == 4:
                    # (batch, heads, seq_len, seq_len) likely attention
                    attns.append(item)
                elif isinstance(item, (list, tuple)):
                    # flatten nested
                    for t in item:
                        if isinstance(t, torch.Tensor) and t.dim() == 4:
                            attns.append(t)
            if len(attns) > 0:
                return tuple(attns)
    except Exception:
        pass
    # 3) no attention available
    return None


def get_my_first_layer_attention_direct(my: LlamaMyModel, input_ids: torch.Tensor):
    """Compute first-layer attention scores from LlamaMyModel directly, without modifying its APIs.
    Returns a tensor shaped (batch, heads, seq, seq).
    """
    # Build input embeddings similarly to single_step
    if input_embedding is not None:
        x0 = input_embedding(input_ids, my.model.get_input_embeddings().weight.to(my.dtype))
    else:
        # fallback: use F.embedding directly
        x0 = F.embedding(input_ids, my.model.get_input_embeddings().weight.to(my.dtype))

    # Access first transformer block
    if not hasattr(my, "attention_blocks") or len(my.attention_blocks) == 0:
        return None
    blk = my.attention_blocks[0]

    # Save old state and enable attention scores
    old_flag = getattr(blk, "get_attention_scores", False)
    old_seq_len = blk.sequence_length
    old_k = blk.k_cache
    old_v = blk.v_cache
    try:
        blk.get_attention_scores = True
        # Reset cache to simulate fresh forward for this sequence
        blk.sequence_length = 0
        blk.k_cache = torch.empty(0, device=x0.device)
        blk.v_cache = torch.empty(0, device=x0.device)

        out = blk.forward(x0, blk.params)
        if isinstance(out, tuple) and len(out) == 2:
            _, attn_scores = out
            return attn_scores
        return None
    finally:
        # Restore original state
        blk.get_attention_scores = old_flag
        blk.sequence_length = old_seq_len
        blk.k_cache = old_k
        blk.v_cache = old_v


def extract_hf_qkv(hflm, input_ids: torch.Tensor):
    """Extract raw Q,K,V (after layernorm, before RoPE) and after RoPE for first HF layer."""
    model = hflm.model
    try:
        layer = model.model.layers[0]
    except Exception:
        return None
    with torch.no_grad():
        emb = model.get_input_embeddings()(input_ids)
        ln = layer.input_layernorm(emb)
        q_proj = layer.self_attn.q_proj(ln)
        k_proj = layer.self_attn.k_proj(ln)
        v_proj = layer.self_attn.v_proj(ln)
        num_heads = model.config.num_attention_heads
        num_kv = model.config.num_key_value_heads
        head_dim = q_proj.shape[-1] // num_heads
        q = q_proj.view(emb.size(0), emb.size(1), num_heads, head_dim).transpose(1, 2)
        k = k_proj.view(emb.size(0), emb.size(1), num_kv, head_dim).transpose(1, 2)
        v = v_proj.view(emb.size(0), emb.size(1), num_kv, head_dim).transpose(1, 2)
        # Apply RoPE using HF internal helper if available, else skip (we compare both pre and post)
        try:
            rope_fn = layer.self_attn.rotary_emb
            # HF RoPE expects (seq_len, head_dim); call on q,k
            seq_len = q.shape[2]
            pos = rope_fn._cos_cached[:seq_len], rope_fn._sin_cached[:seq_len]
            cos, sin = pos
            from math import prod
            # The HF apply_rotary_pos_emb signature may vary; try robust calls
            try:
                q_rope, k_rope = layer.self_attn.apply_rotary_pos_emb(q, k, cos, sin)
            except Exception:
                q_rope, k_rope = q, k
        except Exception:
            q_rope, k_rope = q, k
    return {
        'emb': emb, 'ln': ln, 'q': q, 'k': k, 'v': v, 'q_rope': q_rope, 'k_rope': k_rope
    }


def extract_my_qkv(my: LlamaMyModel, input_ids: torch.Tensor):
    """Extract raw Q,K,V (after layernorm, before RoPE) and after RoPE for first custom block."""
    if not hasattr(my, 'attention_blocks') or len(my.attention_blocks) == 0:
        return None
    blk = my.attention_blocks[0]
    params = blk.params
    with torch.no_grad():
        emb = F.embedding(input_ids, my.model.get_input_embeddings().weight.to(my.dtype))
        if rmsnorm is not None and 'norm1_weight' in params:
            ln = rmsnorm(emb, params['norm1_weight'].to(my.dtype))
        else:
            ln = emb
        batch_size, seq_len, model_dim = ln.shape
        head_dim = model_dim // blk.num_heads
        q_proj = torch.matmul(ln, params['wq'].t())
        k_proj = torch.matmul(ln, params['wk'].t())
        v_proj = torch.matmul(ln, params['wv'].t())
        q = q_proj.view(batch_size, seq_len, blk.num_heads, head_dim).transpose(1, 2)
        k = k_proj.view(batch_size, seq_len, blk.num_kv_heads, head_dim).transpose(1, 2)
        v = v_proj.view(batch_size, seq_len, blk.num_kv_heads, head_dim).transpose(1, 2)
        # Apply custom RoPE
        if apply_rope is not None:
            q_rope = apply_rope(q.clone(), seq_len, head_dim, start_pos=0)
            k_rope = apply_rope(k.clone(), seq_len, head_dim, start_pos=0)
        else:
            q_rope, k_rope = q, k
    return {
        'emb': emb, 'ln': ln, 'q': q, 'k': k, 'v': v, 'q_rope': q_rope, 'k_rope': k_rope
    }

def compare_first_layer(model_path, device_str="cuda", texts=None):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    print("Loading HFLM (lm-eval) model...")
    hflm = HFLM(pretrained=model_path, device=device_str, dtype="auto")
    tokenizer = hflm.tokenizer if hasattr(hflm, "tokenizer") else AutoTokenizer.from_pretrained(model_path)

    print("Loading my model...")
    my = LlamaMyModel(model_name=model_path, device=device, dtype=torch.bfloat16)

    if texts is None:
        texts = [
            "Hello, how are you?",
            "The capital of France is",
            "In the year 2025, AI will"
        ]

    for i, text in enumerate(texts):
        print("\n" + "="*40)
        print(f"Example {i}: {text}")

        encoding = tokenizer(text, return_tensors="pt", truncation=True, add_special_tokens=False)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # HF attentions
        hf_attns = try_get_hf_attentions(hflm, input_ids=input_ids, attention_mask=attention_mask)
        if hf_attns is None:
            print("Could not obtain HF attentions (model may not expose them).")
        else:
            print(f"HF returned {len(hf_attns)} attention layers; first layer shape: {tuple(hf_attns[0].shape)}")

        # My model attentions
        tensor_inputs = {k: v.to(device) for k, v in encoding.items()}
        my_attns = try_get_my_attentions(my, tensor_inputs)
        if my_attns is None:
            # Fallback: compute first-layer attention directly from the first block
            first_attn = get_my_first_layer_attention_direct(my, input_ids)
            if first_attn is not None:
                # Wrap to look like a tuple of layers
                my_attns = (first_attn,)
        if my_attns is None:
            print("Could not obtain my_model attentions automatically. If your LlamaMyModel can return attentions, implement `single_step(..., return_attentions=True)` or `single_step(..., return_latents=True)` that returns per-layer attention tensors shaped (batch, heads, seq, seq).")
        else:
            print(f"my_model returned {len(my_attns)} attention layers; first layer shape: {tuple(my_attns[0].shape)}")

        # Compare first layer if both available
        if hf_attns is not None and my_attns is not None:
            hf_first = to_cpu_f32(hf_attns[0][0])  # (heads, seq, seq) for batch 0
            # my_attns might be tuple/list with batch dim present
            my_first = to_cpu_f32(my_attns[0][0]) if my_attns[0].dim() == 4 else to_cpu_f32(my_attns[0])

            # If my_first looks like raw logits (pre-softmax), convert to attention probs
            if torch.isinf(my_first).any() or my_first.min() < 0 or my_first.max() > 1:
                my_first = F.softmax(my_first, dim=-1)

            # ensure same shape, if needed crop to min seq length
            min_h = min(hf_first.shape[0], my_first.shape[0])
            min_s = min(hf_first.shape[1], my_first.shape[1])
            hf_slice = hf_first[:min_h, :min_s, :min_s]
            my_slice = my_first[:min_h, :min_s, :min_s]
            diff = (hf_slice - my_slice).abs()
            print(f"First-layer attention comparison (heads x seq x seq) slice shapes: {hf_slice.shape}")
            print(f"Max abs diff: {float(diff.max()):.6f}, Mean abs diff: {float(diff.mean()):.6f}")
            # print small sample
            h0 = hf_slice[0][:5,:5].numpy()
            m0 = my_slice[0][:5,:5].numpy()
            print("HF first-head [0:5,0:5]:\n", np.array_str(h0, precision=6, suppress_small=True))
            print("MY first-head [0:5,0:5]:\n", np.array_str(m0, precision=6, suppress_small=True))
        else:
            print("Skipping numeric comparison for first layer (missing data).")

        # QKV comparison (first example only or all)
        hf_qkv = extract_hf_qkv(hflm, input_ids)
        my_qkv = extract_my_qkv(my, input_ids)
        if hf_qkv and my_qkv:
            def stat(name):
                a = to_cpu_f32(hf_qkv[name])
                b = to_cpu_f32(my_qkv[name])
                # Align heads for k/v expansion if needed
                if a.shape != b.shape:
                    min_dims = [min(a.shape[d], b.shape[d]) for d in range(len(a.shape))]
                    a = a[(slice(0,min_dims[0]), slice(0,min_dims[1]), slice(0,min_dims[2]), slice(0,min_dims[3]))] if a.dim()==4 else a[:min_dims[0],:min_dims[1],:min_dims[2]]
                    b = b[(slice(0,min_dims[0]), slice(0,min_dims[1]), slice(0,min_dims[2]), slice(0,min_dims[3]))] if b.dim()==4 else b[:min_dims[0],:min_dims[1],:min_dims[2]]
                diff = (a - b).abs()
                return float(diff.max()), float(diff.mean())
            for key in ['emb','ln','q','k','v','q_rope','k_rope']:
                mx, mn = stat(key)
                print(f"QKV {key} max={mx:.6f} mean={mn:.6f}")
            # sample one head slice after rope for q
            qh_hf = to_cpu_f32(hf_qkv['q_rope'])[0,0,:5,:8].numpy()
            qh_my = to_cpu_f32(my_qkv['q_rope'])[0,0,:5,:8].numpy()
            print("HF q_rope head0 slice:\n", np.array_str(qh_hf, precision=6, suppress_small=True))
            print("MY q_rope head0 slice:\n", np.array_str(qh_my, precision=6, suppress_small=True))
        else:
            print("Skipped QKV extraction (unavailable).")

        # Derive first-layer attn_output and post-attention states to localize differences
        try:
            model = hflm.model
            layer0 = model.model.layers[0]
            num_heads = model.config.num_attention_heads
            num_kv = model.config.num_key_value_heads
            head_dim = model.config.hidden_size // num_heads

            if hf_attns is not None and hf_qkv is not None and my_attns is not None and my_qkv is not None:
                # Prepare HF and MY attention probs
                hf_probs = to_cpu_f32(hf_attns[0][0])
                my_probs = to_cpu_f32(my_attns[0][0]) if my_attns[0].dim() == 4 else to_cpu_f32(my_attns[0])
                if torch.isinf(my_probs).any() or my_probs.min() < 0 or my_probs.max() > 1:
                    my_probs = F.softmax(my_probs, dim=-1)

                # V (expand kv heads)
                hf_v = to_cpu_f32(hf_qkv['v'])  # (B, kv, S, D)
                my_v = to_cpu_f32(my_qkv['v'])
                if hf_v.shape[1] != num_heads:
                    rep = num_heads // hf_v.shape[1]
                    hf_v = hf_v.repeat_interleave(rep, dim=1)
                if my_v.shape[1] != num_heads:
                    rep = num_heads // my_v.shape[1]
                    my_v = my_v.repeat_interleave(rep, dim=1)

                def attn_out(probs, v):
                    # (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
                    return torch.matmul(probs, v)

                hf_attn_out = attn_out(hf_probs, hf_v)
                my_attn_out = attn_out(my_probs, my_v)

                # Merge heads and project with Wo
                def merge_and_o(x, wo):
                    B, H, S, D = x.shape
                    x = x.transpose(1, 2).contiguous().view(B, S, H*D)
                    return torch.matmul(x, wo.t().cpu().to(torch.float32))

                hf_o = merge_and_o(hf_attn_out, layer0.self_attn.o_proj.weight)
                my_o = merge_and_o(my_attn_out, my.attention_blocks[0].params['wo'])

                # Residual 1
                emb_hf = to_cpu_f32(hf_qkv['emb'])
                emb_my = to_cpu_f32(my_qkv['emb'])
                hf_res1 = emb_hf + hf_o
                my_res1 = emb_my + my_o

                # LN2
                # Use custom rmsnorm for both to avoid device mix
                if rmsnorm is not None:
                    hf_ln2 = to_cpu_f32(rmsnorm(hf_res1, layer0.post_attention_layernorm.weight.detach().cpu().to(torch.float32)))
                    my_ln2 = to_cpu_f32(rmsnorm(my_res1, my.attention_blocks[0].params['norm2_weight'].cpu().to(torch.float32)))
                else:
                    hf_ln2 = hf_res1
                    my_ln2 = my_res1

                # FFN
                if ffn_SwiGLU is not None:
                    hf_ffn_out = to_cpu_f32(
                        ffn_SwiGLU(
                            hf_ln2,
                            layer0.mlp.gate_proj.weight.detach().cpu().to(torch.float32),
                            layer0.mlp.up_proj.weight.detach().cpu().to(torch.float32),
                            layer0.mlp.down_proj.weight.detach().cpu().to(torch.float32),
                        )
                    )
                    my_ffn_out = to_cpu_f32(
                        ffn_SwiGLU(
                            my_ln2,
                            my.attention_blocks[0].params['w_gate'].cpu().to(torch.float32),
                            my.attention_blocks[0].params['w_up'].cpu().to(torch.float32),
                            my.attention_blocks[0].params['w_down'].cpu().to(torch.float32),
                        )
                    )
                else:
                    hf_ffn_out = hf_ln2
                    my_ffn_out = my_ln2

                # Residual 2
                hf_res2 = hf_res1 + hf_ffn_out
                my_res2 = my_res1 + my_ffn_out

                # Report stats
                def stat_pair(name, a, b):
                    d = (a - b).abs()
                    print(f"{name} max={float(d.max()):.6f} mean={float(d.mean()):.6f}")

                print("First layer breakdown diffs (HF vs MY):")
                stat_pair("attn_out", to_cpu_f32(hf_attn_out), to_cpu_f32(my_attn_out))
                stat_pair("o_proj", to_cpu_f32(hf_o), to_cpu_f32(my_o))
                stat_pair("residual1", to_cpu_f32(hf_res1), to_cpu_f32(my_res1))
                stat_pair("ln2", to_cpu_f32(hf_ln2), to_cpu_f32(my_ln2))
                stat_pair("ffn_out", to_cpu_f32(hf_ffn_out), to_cpu_f32(my_ffn_out))
                stat_pair("residual2", to_cpu_f32(hf_res2), to_cpu_f32(my_res2))
        except Exception as e:
            print("Skipped first-layer breakdown due to:", repr(e))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="TinyLlama/TinyLlama_v1.1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--examples", type=str, nargs="*", default=None)
    args = parser.parse_args()
    compare_first_layer(model_path=args.model_path, device_str=args.device, texts=args.examples)