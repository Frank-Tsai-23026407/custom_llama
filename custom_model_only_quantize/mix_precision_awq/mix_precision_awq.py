import argparse
import copy
import os
import sys
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from awq_utils import search_awq_scale, apply_awq_scale, pseudo_quantize
from testbench import task_utils as TU

def get_weight_saliency(W, X):
    """
    實作 get_weight_saliency 函式。
    根據 W 的量級與 X 的平均量級評估權重的重要性。
    
    Args:
        W (torch.Tensor): 權重矩陣 [Co, Ci]
        X (torch.Tensor): 輸入活化值 [N, Ci]
        
    Returns:
        torch.Tensor: 顯著性分數，與 W 同形狀 [Co, Ci]
    """
    X_mean = X.abs().mean(dim=0) # [Ci]
    # W * mean(|X|)
    return W.abs() * X_mean.view(1, -1)

def mixed_precision_quantize(W, saliency, n_bits, block_height, block_width, top_k_ratio=0.01):
    """
    混合精度量化器：識別每個 Block 中的 Top-K 敏感權重並保護之。
    
    Args:
        W (torch.Tensor): 縮放後的權重 [Co, Ci]
        saliency (torch.Tensor): 顯著性分數 [Co, Ci]
        n_bits (int): 量化位元數
        block_height (int): BFP 塊高度
        block_width (int): BFP 塊寬度
        top_k_ratio (float): 每個 Block 中保護的權重比例 (例如 0.01)
        
    Returns:
        torch.Tensor: 混合精度量化後的權重
    """
    Co, Ci = W.shape
    device = W.device
    
    # 1. Padding 以符合 BFP 的 Block 劃分
    pad_h = (block_height - (Co % block_height)) % block_height
    pad_w = (block_width - (Ci % block_width)) % block_width
    
    W_padded = torch.nn.functional.pad(W, (0, pad_w, 0, pad_h))
    S_padded = torch.nn.functional.pad(saliency, (0, pad_w, 0, pad_h), value=0.0)
    
    padded_h, padded_w = W_padded.shape
    nbh = padded_h // block_height
    nbw = padded_w // block_width
    
    # 2. 將 Tensor 展平成 Blocks: [num_blocks, block_size]
    # (nbh, bh, nbw, bw) -> (nbh, nbw, bh, bw) -> (total_blocks, bh*bw)
    block_size = block_height * block_width
    W_blocks = W_padded.reshape(nbh, block_height, nbw, block_width).permute(0, 2, 1, 3).reshape(-1, block_size)
    S_blocks = S_padded.reshape(nbh, block_height, nbw, block_width).permute(0, 2, 1, 3).reshape(-1, block_size)
    
    # 3. 計算每個 Block 要保留的 Top-K 數量
    k = max(1, int(block_size * top_k_ratio))
    
    # 4. 找出每個 Block 中顯著性前 K 高的權重索引
    # vals: [num_blocks, k], indices: [num_blocks, k]
    _, indices = torch.topk(S_blocks, k, dim=1)
    
    # 5. 建立 Mask
    mask_blocks = torch.zeros_like(S_blocks, dtype=torch.bool)
    mask_blocks.scatter_(1, indices, True)
    
    # 6. 執行量化：為了讓其餘權重量化更精準，先把 salient weights 設為 0 以減小 shared exponent
    W_blocks_for_quant = W_blocks.clone()
    # 儲存原始 salient weights
    salient_weights = W_blocks[mask_blocks].clone()
    W_blocks_for_quant[mask_blocks] = 0.0
    
    # 將 Blocks 還原為矩陣形式進行量化
    W_for_quant = W_blocks_for_quant.reshape(nbh, nbw, block_height, block_width).permute(0, 2, 1, 3).reshape(padded_h, padded_w)
    W_for_quant = W_for_quant[:Co, :Ci] # 去除 padding
    
    W_quantized = pseudo_quantize(W_for_quant, n_bits=n_bits, block_height=block_height, block_width=block_width)
    
    # 再次 Padding Q 好的權重以便還原
    W_quantized_padded = torch.nn.functional.pad(W_quantized, (0, pad_w, 0, pad_h))
    W_quantized_blocks = W_quantized_padded.reshape(nbh, block_height, nbw, block_width).permute(0, 2, 1, 3).reshape(-1, block_size)
    
    # 7. 混合精度：將原本的 salient weights 塞回去
    W_quantized_blocks[mask_blocks] = salient_weights
    
    # 8. 最終還原形狀
    W_final_padded = W_quantized_blocks.reshape(nbh, nbw, block_height, block_width).permute(0, 2, 1, 3).reshape(padded_h, padded_w)
    W_final = W_final_padded[:Co, :Ci]
    
    # 生成 [Co, Ci] 的 Mask 作為 metadata 記錄
    final_mask = mask_blocks.reshape(nbh, nbw, block_height, block_width).permute(0, 2, 1, 3).reshape(padded_h, padded_w)[:Co, :Ci]
    
    return W_final, final_mask

def quantize_model_mixed(model_path, dataset_name, dataset_config, num_samples, block_height, block_width, mantissa_bits, top_k_ratio=0.01, device="auto", dry_run=False, eval_ppl=False, eval_hellaswag=False, limit_tokens_ppl=None, max_samples_hs=None):
    """
    實作 Mixed-Precision AWQ。
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"Loading and preparing dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, dataset_config, split="train").select(range(num_samples))
    text = "\n\n".join(dataset["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids.to(device)

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = input[0].detach()
        return hook

    hooks = []
    for name, module in model.named_modules():
        if "lm_head" in name or "embed_tokens" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(get_activation(name)))

    print("Performing forward pass to get activations...")
    model.to(device)
    with torch.no_grad():
        model(tokens)

    for hook in hooks:
        hook.remove()

    # Identify target layers
    target_layers = []
    for name, module in model.named_modules():
        if "lm_head" in name or "embed_tokens" in name:
            continue
        if isinstance(module, torch.nn.Linear):
            if name in activations:
                target_layers.append(name)
    
    print(f"Target layers for mixed-precision quantization ({len(target_layers)}):")
    for name in target_layers:
        print(f"  - {name}")

    print("Applying Mixed-Precision AWQ quantization...")
    awq_scales = {}
    quant_metadata = {}

    for name, module in tqdm(model.named_modules(), total=len(list(model.named_modules())), desc="Quantizing"):
        if name in target_layers:
            X = activations[name]
            X = X.view(-1, X.shape[-1])
            W = module.weight.data
            
            # 第一步：AWQ Search 並縮放權重
            s = search_awq_scale(W, X, block_height=block_height, block_width=block_width)
            awq_scales[name] = s.clone().cpu()
            W_scaled = apply_awq_scale(W, s)
            
            # 第二步：計算縮放後權重的 Saliency
            saliency = get_weight_saliency(W_scaled, X)
            
            # 第三步：混合精度量化 (Top-K 敏感通道維持 FP16)
            W_mixed, mask = mixed_precision_quantize(
                W_scaled, saliency, n_bits=mantissa_bits, 
                block_height=block_height, block_width=block_width, 
                top_k_ratio=top_k_ratio
            )
            
            # 還原縮放以維持輸出維度
            W_final = W_mixed / s.view(1, -1)
            module.weight.data = W_final
            
            # 記錄被保護的通道數量
            protected_channels = mask[0].sum().item()
            quant_metadata[name] = {"protected_channels": protected_channels}
            
        elif isinstance(module, torch.nn.Linear) and ("lm_head" not in name and "embed_tokens" not in name):
            # print(f"Skipping layer {name} as no activation was captured.")
            pass

    if eval_ppl:
        print("\nEvaluating WikiText-103 PPL...")
        TU.evaluate_ppl(model, tokenizer, device, limit_tokens=limit_tokens_ppl)
    
    if eval_hellaswag:
        print("\nEvaluating HellaSwag...")
        TU.evaluate_hellaswag(model, tokenizer, device, max_samples=max_samples_hs)

    if not dry_run:
        output_dir = f"{model_path.replace('/', '-')}-awq-quantized-mix-precision-bh{block_height}-bw{block_width}-m{mantissa_bits}-r{top_k_ratio}"
        os.makedirs(output_dir, exist_ok=True)
        
        metadata = {
            "quantized_layers": target_layers,
            "layer_stats": quant_metadata,
            "config": {
                "block_height": block_height,
                "block_width": block_width,
                "mantissa_bits": mantissa_bits,
                "top_k_ratio": top_k_ratio
            }
        }
        with open(os.path.join(output_dir, "awq_config.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        torch.save(awq_scales, os.path.join(output_dir, "awq_scales.pt"))
        
        model = model.to(torch.bfloat16)
        model.save_pretrained(output_dir, torch_dtype=torch.bfloat16)
        tokenizer.save_pretrained(output_dir)
        print(f"Quantized model saved to: {output_dir}")
    else:
        print("Dry run complete; not saving model.")

def main():
    parser = argparse.ArgumentParser(description="AWQ Mixed-Precision Quantization CLI")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama_v1.1",
                        help="Model path or identifier")
    parser.add_argument("--dataset", type=str, default="Salesforce/wikitext",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1",
                        help="Dataset config name if applicable")
    parser.add_argument("--num-samples", type=int, default=128,
                        help="Number of samples for calibration")
    parser.add_argument("--block-height", type=int, default=1,
                        help="BFP block height")
    parser.add_argument("--block-width", type=int, default=64,
                        help="BFP block width")
    parser.add_argument("--mantissa-bits", type=int, nargs="+", default=[4],
                        help="Mantissa bits to try")
    parser.add_argument("--top-k-ratio", type=float, default=0.01,
                        help="Ratio of salient channels to preserve in FP16")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Computation device")
    parser.add_argument("--dry-run", action="store_true", help="Run without saving models")
    parser.add_argument("--eval-ppl", action="store_true", help="Evaluate WikiText2 PPL")
    parser.add_argument("--eval-hellaswag", action="store_true", help="Evaluate HellaSwag Accuracy")
    parser.add_argument("--max-samples-hs", type=int, default=None, help="Max samples for HellaSwag evaluation")
    parser.add_argument("--limit-tokens-ppl", type=int, default=None, help="Limit tokens for PPL evaluation for speed")

    args = parser.parse_args()

    for m in args.mantissa_bits:
        print(f"\nEvaluating Mix-Precision AWQ: Bits={m}, Ratio={args.top_k_ratio}")
        quantize_model_mixed(
            args.model, args.dataset, args.dataset_config, args.num_samples,
            args.block_height, args.block_width, m, 
            top_k_ratio=args.top_k_ratio, 
            device=args.device, 
            dry_run=args.dry_run,
            eval_ppl=args.eval_ppl,
            eval_hellaswag=args.eval_hellaswag,
            limit_tokens_ppl=args.limit_tokens_ppl,
            max_samples_hs=args.max_samples_hs
        )

if __name__ == "__main__":
    main()
