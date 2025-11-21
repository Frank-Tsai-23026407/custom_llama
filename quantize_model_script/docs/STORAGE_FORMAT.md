# Storage Format and Compression

## Current Limitation

### ⚠️ Important: Quantized Models Are Saved in FP32/FP16 Format

**What This Means:**
- Quantization algorithms (BFP, AWQ) are correctly applied to weight values
- Weights are computed using low-precision representations (e.g., 4-bit mantissa)
- **However**, the resulting values are still stored as standard PyTorch tensors (float32/float16)
- **File size is NOT reduced** to match the quantization bit-width

**Example:**
```
Original model:      ~4.4 GB (FP32)
BFP 4-bit quantized: ~4.4 GB (FP32 storage of quantized values)
                     
With --save-dtype float16:
                     ~2.2 GB (FP16 storage of quantized values)
```

## Why This Happens

PyTorch's `save_pretrained()` method:
1. Saves model weights as standard tensor files (safetensors or pickle)
2. Uses the tensor's dtype (float32, float16, bfloat16)
3. Does **not** have built-in support for custom quantization formats

Our quantization process:
```python
# Quantization happens here (e.g., 4-bit BFP)
quantized_weight = block_floating_point_quantize_2d(weight, ...)

# But it's still a torch.float32 tensor
module.weight.data = quantized_weight  # dtype=torch.float32

# Save still uses float32
model.save_pretrained(output_dir)  # Saves as float32
```

## Current Workarounds

### Option 1: Save in FP16 (Recommended)
```bash
python quantize_tinyllama_comprehensive_2d.py --save-dtype float16
```
- Reduces file size by 50%
- Compatible with all PyTorch tools
- Still not as compressed as true 4-bit storage

### Option 2: Save in BFloat16
```bash
python quantize_tinyllama_comprehensive_2d.py --save-dtype bfloat16
```
- Also 50% reduction
- Better numerical range than FP16
- May have better compatibility with some hardware

## True Compression: What's Needed

To achieve file sizes matching the quantization precision (e.g., 4-bit = ~0.5 bytes per weight):

### Custom Storage Format
```python
# Instead of storing float32 values:
# [1.234, 0.567, -0.891, ...]  # 4 bytes each

# Store quantized representation:
{
    'exponents': [3, 2, 1, ...],           # 8-bit per block
    'mantissas': [0b1011, 0b0111, ...],    # 4-bit per weight
    'scales': [...],                        # For AWQ
    'fp32_indices': [...],                  # For mix-precision
    'fp32_values': [...],                   # For mix-precision
}
```

### Required Implementation

1. **Serialization:**
   ```python
   def save_quantized_model(model, path, quant_config):
       for layer in model.layers:
           # Extract quantization parameters
           exponents, mantissas, ... = extract_quant_params(layer.weight)
           # Save in custom format
           save_custom_format(exponents, mantissas, ...)
   ```

2. **Deserialization:**
   ```python
   def load_quantized_model(path):
       for layer in model.layers:
           # Load quantization parameters
           exponents, mantissas = load_custom_format(...)
           # Reconstruct weights
           layer.weight = reconstruct_from_quant(exponents, mantissas)
   ```

3. **Inference Kernels:**
   - Custom CUDA kernels for quantized matmul
   - Direct computation with quantized format (no dequantization)
   - Significant speedup potential

## Comparison with Production Systems

### HuggingFace bitsandbytes
```python
model = AutoModelForCausalLM.from_pretrained(
    "model",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
```
- Has custom serialization format
- Optimized CUDA kernels
- True 4-bit storage and computation

### GPTQ
- Custom quantization format
- Compressed storage
- Fast inference kernels

### AWQ (Official)
- Channel-wise quantization
- INT4 storage format
- Custom inference engine

## Future Improvements

### Phase 1: Custom Serialization (Storage)
- [ ] Implement custom save format for BFP
- [ ] Store only exponents + quantized mantissas
- [ ] Handle mix-precision FP32 values separately
- [ ] Compression: 4-bit → ~0.5 bytes per weight

### Phase 2: Efficient Inference (Speed)
- [ ] Dequantize-on-the-fly for inference
- [ ] Custom CUDA kernels for quantized ops
- [ ] Fused dequantization + matmul

### Phase 3: Full Integration
- [ ] Integrate with HuggingFace transformers
- [ ] Support for various quantization methods
- [ ] Benchmarking suite

## Current Use Cases

Despite the storage limitation, the current implementation is useful for:

✓ **Algorithm Research:**
  - Comparing quantization methods
  - Measuring accuracy impact
  - Prototyping new quantization schemes

✓ **Inference Quality Testing:**
  - Running models with quantized weights
  - Validating generation quality
  - Benchmark accuracy metrics

✓ **Educational Purposes:**
  - Understanding quantization algorithms
  - Implementing BFP, AWQ from scratch
  - Experimenting with different configurations

✗ **Not Suitable For:**
  - Production deployment (use bitsandbytes/GPTQ/AWQ libraries)
  - Actual model compression (file size not reduced)
  - High-performance inference (no custom kernels)

## Recommendations

### For Research/Experimentation
```bash
# Use default FP32 for maximum precision in analysis
python quantize_tinyllama_comprehensive_2d.py

# Check accuracy metrics
python evaluate_quantized_model.py
```

### For Approximate Compression
```bash
# Use FP16 to reduce file size by 50%
python quantize_tinyllama_comprehensive_2d.py --save-dtype float16
```

### For Production
Use established libraries:
```bash
# bitsandbytes (easiest)
pip install bitsandbytes

# GPTQ (good balance)
pip install auto-gptq

# AWQ (fastest)
pip install autoawq
```

## Summary

| Aspect | Current Implementation | Ideal Implementation |
|--------|----------------------|---------------------|
| **Quantization Algorithm** | ✅ Fully implemented | - |
| **Weight Values** | ✅ Correctly quantized | - |
| **Inference** | ✅ Works with quantized weights | ⚠️ Could be faster |
| **Storage Format** | ❌ FP32/FP16 (large files) | ✅ Custom format (small files) |
| **File Size** | ❌ Same as unquantized | ✅ Matches bit-width |
| **Custom Kernels** | ❌ Not implemented | ✅ Optimized CUDA kernels |

The current implementation focuses on **correctness** and **research flexibility** rather than **production deployment** and **storage efficiency**.
