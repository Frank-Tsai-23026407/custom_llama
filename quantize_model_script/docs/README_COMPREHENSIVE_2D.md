# TinyLlama Comprehensive 2D Quantization

Complete quantization suite for TinyLlama with multiple methods and configurations.

## Supported Methods

### 1. BFP (Block Floating Point)
- Standard 2D block-based quantization
- No activation awareness
- All weights quantized uniformly
- Fastest quantization

### 2. AWQ Fix-Precision
- Activation-aware weight quantization with **scaling factors**
- Uses activation magnitudes to compute importance scores
- Scales weights by importance **before** quantization
- **All weights (including salient ones) are represented in BFP format**
- Scaling protects important weights from quantization error
- Better accuracy than BFP with moderate overhead

### 3. AWQ Mix-Precision
- Activation-aware weight quantization with **mixed precision**
- Identifies top-k most salient weights per block
- **Salient weights preserved in full FP32 precision**
- Remaining weights quantized with BFP
- Best accuracy, but requires storing FP32 values
- Slower quantization and larger model size

## Quantization Configurations

### Block Sizes
- 128×1, 64×2, 32×4, 16×8, 8×16, 4×32, 2×64, 1×128

### Mantissa Bits
- 5 bits, 4 bits

### Total Configurations
- **Per method:** 16 (8 block sizes × 2 mantissa bits)
- **All methods:** 48 configurations (16 × 3 methods)

## Usage

### Quick Start

```bash
# Run all methods with all configurations (48 total)
python quantize_tinyllama_comprehensive_2d.py

# Or use the bash wrapper
bash run_quantize_tinyllama_comprehensive.sh
```

### Method-Specific Quantization

```bash
# BFP only (16 configurations, fastest)
python quantize_tinyllama_comprehensive_2d.py --method bfp

# AWQ fix-precision only (16 configurations)
python quantize_tinyllama_comprehensive_2d.py --method awq-fix

# AWQ mix-precision only (16 configurations)
python quantize_tinyllama_comprehensive_2d.py --method awq-mix

# All methods (48 configurations)
python quantize_tinyllama_comprehensive_2d.py --method all
```

### Advanced Options

```bash
# Custom top-k for AWQ methods
python quantize_tinyllama_comprehensive_2d.py --method awq-fix --top-k 32

# Custom calibration dataset
python quantize_tinyllama_comprehensive_2d.py \
    --dataset "wikitext" \
    --dataset-config "wikitext-2-raw-v1" \
    --num-samples 64

# Dry run (quantize but don't save)
python quantize_tinyllama_comprehensive_2d.py --dry-run

# Skip generation tests (faster)
python quantize_tinyllama_comprehensive_2d.py --skip-generation-test

# Custom output directory
python quantize_tinyllama_comprehensive_2d.py --output-dir /path/to/output
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `tinyllama` | Model preset or path |
| `--method` | str | `all` | Quantization method: `bfp`, `awq-fix`, `awq-mix`, or `all` |
| `--output-dir` | str | `<model>-2d-comprehensive` | Base output directory |
| `--top-k` | int | `16` | Top-k salient weight **entries** per block for AWQ Mix-Precision (not per row) |
| `--dataset` | str | `Salesforce/wikitext` | Calibration dataset |
| `--dataset-config` | str | `wikitext-103-raw-v1` | Dataset configuration |
| `--num-samples` | int | `128` | Number of calibration samples |
| `--dry-run` | flag | - | Quantize without saving |
| `--skip-generation-test` | flag | - | Skip text generation validation |
| `--save-dtype` | str | `float32` | Storage format: `float32`, `float16`, or `bfloat16` |

## ⚠️ Important: Storage Format Limitation

**Current Implementation:**
- Quantization is applied to weight values (e.g., 4-bit mantissa BFP)
- BUT weights are still stored as PyTorch tensors in FP32/FP16 format
- This means **file size is NOT reduced** by quantization alone

**Storage Options:**

1. **`--save-dtype float32`** (default)
   - Full precision storage
   - ~4 bytes per weight
   - No compression

2. **`--save-dtype float16`** (recommended for smaller files)
   - Half precision storage
   - ~2 bytes per weight
   - 50% file size reduction
   - **Note:** This is storage format conversion, NOT part of quantization

3. **`--save-dtype bfloat16`**
   - BFloat16 storage
   - ~2 bytes per weight
   - Better numerical range than FP16

**Example:**
```bash
# Save in FP16 to reduce file size by 50%
python quantize_tinyllama_comprehensive_2d.py --save-dtype float16
```

**For True Compressed Storage:**
To achieve actual compression matching the quantization precision (e.g., 4-bit), you would need:
- Custom serialization format (store only exponents + quantized mantissas)
- Custom deserialization and inference kernels
- This is beyond the current implementation scope

The current approach focuses on:
- ✓ Demonstrating quantization algorithms
- ✓ Measuring accuracy impact
- ✓ Enabling inference with quantized weights
- ✗ Optimized storage format (requires custom format)

## Output Structure

```
<output-base>/
├── bfp/
│   ├── block_128x1_mantissa_5/
│   ├── block_128x1_mantissa_4/
│   ├── block_64x2_mantissa_5/
│   ├── ...
│   └── block_1x128_mantissa_4/
├── awq_fix/
│   ├── block_128x1_mantissa_5/
│   ├── block_128x1_mantissa_4/
│   ├── ...
│   └── block_1x128_mantissa_4/
└── awq_mix/
    ├── block_128x1_mantissa_5/
    ├── block_128x1_mantissa_4/
    ├── ...
    └── block_1x128_mantissa_4/
```

Each model directory contains:
- `config.json` - Model configuration
- `model.safetensors` or `pytorch_model.bin` - Quantized weights
- `tokenizer_config.json`, `tokenizer.json` - Tokenizer files

## Expected Runtime

### BFP Only (16 configs)
- **Per config:** ~2-3 minutes
- **Total:** ~30-50 minutes

### AWQ Fix-Precision (16 configs)
- **Activation collection:** ~5-10 minutes (one-time)
- **Per config:** ~3-5 minutes
- **Total:** ~55-90 minutes

### AWQ Mix-Precision (16 configs)
- **Activation collection:** ~5-10 minutes (one-time)
- **Per config:** ~3-5 minutes
- **Total:** ~55-90 minutes

### All Methods (48 configs)
- **Total:** ~2-4 hours (activation collection shared across AWQ methods)

## Memory Requirements

- **RAM:** ~10-16 GB
  - Base model: ~4 GB
  - Model copy per config: ~4 GB
  - Activations (AWQ): ~2-4 GB
- **Disk Space:** 
  - Per quantized model: ~1-2 GB
  - Total (48 configs): ~50-100 GB

## Method Comparison

| Method | Speed | Accuracy | Representation | Use Case |
|--------|-------|----------|----------------|----------|
| **BFP** | ⚡⚡⚡ Fast | ⭐⭐ Good | All BFP | Quick experiments, baseline |
| **AWQ Fix** | ⚡⚡ Medium | ⭐⭐⭐ Better | All BFP (scaled) | Production, deployment-friendly |
| **AWQ Mix** | ⚡ Slow | ⭐⭐⭐⭐ Best | FP32 + BFP | Maximum accuracy needed |

## Quantization Details

### BFP (Block Floating Point)
1. Divide weight matrix into 2D blocks
2. Find shared exponent per block (max absolute value)
3. Quantize mantissas to specified bit-width
4. Reconstruct: `value = (quantized_mantissa / Q_max) * 2^exponent`

### AWQ Fix-Precision (Scaling-Based)
1. Collect activations from calibration dataset
2. Compute per-input-feature importance scores: `s = mean(|activation|)`
3. **Scale weights UP by importance:** `W_scaled = W * s`
4. Apply standard BFP quantization to scaled weights: `Q(W_scaled)`
5. **Scale quantized weights back DOWN:** `W_final = Q(W_scaled) / s`
6. **Result:** All weights in BFP, but important weights effectively get more precision
7. **Advantage:** Hardware-friendly (uniform BFP format), better than baseline BFP

### AWQ Mix-Precision (FP32 Preservation)
1. Collect activations from calibration dataset
2. Compute per-input-feature importance scores
3. For each 2D block:
   - Compute salience: `salience = |W * s|` (element-wise)
   - Identify top-k **individual entries** with highest salience within the block
     * **Note:** top-k selects k entries from the entire flattened block
     * **Not** per-row or per-column selection
     * These k entries can be scattered across different rows and columns
     * Example: 16×16 block with top_k=16 → select 16 out of 256 entries (6.25%)
   - Apply BFP quantization to all weights in the block
   - **Replace the top-k entry positions with original FP32 values**
4. **Result:** Mix of FP32 (salient entries) and BFP (non-salient entries) weights
5. **Advantage:** Best accuracy; **Disadvantage:** Non-uniform format, larger storage

**Top-K Selection Details:**
```
Example: 4×4 block, top_k=2

Original block (16 elements):
  [0.1  0.5  0.2  0.3]
  [0.8  0.1  0.4  0.2]  
  [0.3  0.7  0.1  0.6]
  [0.2  0.4  0.3  0.9]

Top-2 salient entries (by absolute value):
  Position [3,3] = 0.9 (highest)
  Position [1,0] = 0.8 (second highest)

These 2 entries are preserved in FP32.
The other 14 entries are quantized to BFP.
```

## Block Size Selection Guide

### Tall Blocks (128×1, 64×2)
- ✅ Better for output channel patterns
- ✅ Lower storage overhead
- ❌ Less precision across input features

### Wide Blocks (1×128, 2×64)
- ✅ Better for input feature patterns
- ✅ Fine-grained across outputs
- ❌ Higher storage overhead

### Balanced Blocks (32×4, 16×8, 8×16, 4×32)
- ✅ Good compromise
- ✅ Flexible precision
- ✅ Moderate overhead

## Example Workflow

### 1. Quick Test with BFP
```bash
# Fast baseline - only BFP quantization
python quantize_tinyllama_comprehensive_2d.py \
    --method bfp \
    --skip-generation-test
```

### 2. Full AWQ Fix-Precision
```bash
# Full activation-aware quantization
python quantize_tinyllama_comprehensive_2d.py \
    --method awq-fix \
    --top-k 16 \
    --num-samples 128
```

### 3. High-Accuracy Mix-Precision
```bash
# Best accuracy with more calibration samples
python quantize_tinyllama_comprehensive_2d.py \
    --method awq-mix \
    --top-k 32 \
    --num-samples 256
```

### 4. Complete Suite
```bash
# All methods, all configurations
python quantize_tinyllama_comprehensive_2d.py --method all
```

## Troubleshooting

### ImportError: transformers or datasets
```bash
pip install transformers datasets
```

### CUDA Out of Memory
- Models run on CPU by default
- Use `--skip-generation-test` to reduce memory
- Reduce `--num-samples` for calibration

### Slow Execution
- Use `--method bfp` for fastest results
- Use `--skip-generation-test`
- Reduce calibration samples with `--num-samples 64`

### Dataset Download Issues
```bash
# Pre-download the dataset
python -c "from datasets import load_dataset; load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1')"
```

## Related Scripts

- `block_quantization_2d.py` - Core 2D quantization functions
- `quantize_tinyllama_2d_sweep.py` - BFP-only sweep (simpler)
- `bfp_quantize_2d.py` - Single BFP configuration
- `awq_quantize_2d.py` - Single AWQ configuration

## Example Output

```
Configuration 1/48
Method: BFP, Block: 128×1, Mantissa: 5
================================================================================
Creating model copy...

Quantizing with bfp:
  [  1] model.layers.0.self_attn.q_proj                MSE: 1.234567e-05
  [  2] model.layers.0.self_attn.k_proj                MSE: 2.345678e-05
  ...
  [198] model.layers.21.mlp.down_proj                  MSE: 3.456789e-05

Quantized 198 layers

Testing generation...
Generated: Hello, how are you? I'm doing well, thank you for asking...

Saving to: ../model/tinyllama/TinyLlama_1.1v-2d-comprehensive/bfp/block_128x1_mantissa_5
Saved successfully!

Configuration 1 completed
```
