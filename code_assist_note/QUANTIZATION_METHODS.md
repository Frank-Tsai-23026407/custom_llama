# Quantization Methods

## Definitions

**BFP**: 2D blocks with shared exponent, individual mantissas (2-5 bits) - `bfp_quantize_2d()`

**Fix-Precision AWQ**: ALL weights in BFP, uses scaling (`weight × importance → quantize → descale`) - `awq_fix_precision_quantize_2d()`

**Mix-Precision AWQ**: Top-k weights in FP32/BF16, rest in BFP (k entries per flattened block) - `awq_mix_precision_quantize_2d()`

## Comparison

| Method | Format | Storage | Accuracy |
|--------|--------|---------|----------|
| BFP | All BFP | Smallest | Baseline |
| Fix-Precision | All BFP (scaled) | = BFP | Better |
| Mix-Precision | BFP + FP32 | Larger | Best |

## ⚠️ CRITICAL: Runtime Only

**DO NOT save quantized BFP models (AWQ model can still be stored)** - Apply at runtime, store only BF16 base models

```python
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
apply_runtime_quantization(model, method="fix", block_size=64, mantissa_bits=4)
```

Storage: ~109 GB (saved variants) → ~2.2 GB (runtime) = 98% reduction

Files: `runtime_quantizer.py`, `block_quantization_2d.py`
