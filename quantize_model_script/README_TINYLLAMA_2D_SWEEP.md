# TinyLlama 2D Block Quantization Sweep

This script performs comprehensive 2D block-based Block Floating Point (BFP) quantization on TinyLlama model with multiple configurations.

## Quantization Configurations

### Block Sizes
- 128×1
- 64×2
- 32×4
- 16×8
- 8×16
- 4×32
- 2×64
- 1×128

### Mantissa Bits
- 5 bits
- 4 bits

**Total Configurations:** 16 (8 block sizes × 2 mantissa bit settings)

## Usage

### Basic Usage

```bash
# Run the quantization sweep (will save all 16 quantized models)
python quantize_tinyllama_2d_sweep.py
```

### With Options

```bash
# Dry run (quantize but don't save)
python quantize_tinyllama_2d_sweep.py --dry-run

# Skip generation tests (faster)
python quantize_tinyllama_2d_sweep.py --skip-generation-test

# Specify custom output directory
python quantize_tinyllama_2d_sweep.py --output-dir /path/to/output

# Use bash script wrapper
bash run_quantize_tinyllama_2d_sweep.sh
```

### Command-Line Arguments

- `--model`: Model preset or path (default: `tinyllama`)
  - Presets: `tinyllama`, `llama-3.2-1b`
  - Or provide full path to model directory
- `--output-dir`: Base output directory (default: `<model_path>-bfp2d-sweep`)
- `--dry-run`: Quantize without saving models (for testing)
- `--skip-generation-test`: Skip text generation validation after each quantization

## Output Structure

Quantized models will be saved in the following structure:

```
<output-base>/
├── block_128x1_mantissa_5/
├── block_128x1_mantissa_4/
├── block_64x2_mantissa_5/
├── block_64x2_mantissa_4/
├── block_32x4_mantissa_5/
├── block_32x4_mantissa_4/
├── block_16x8_mantissa_5/
├── block_16x8_mantissa_4/
├── block_8x16_mantissa_5/
├── block_8x16_mantissa_4/
├── block_4x32_mantissa_5/
├── block_4x32_mantissa_4/
├── block_2x64_mantissa_5/
├── block_2x64_mantissa_4/
├── block_1x128_mantissa_5/
└── block_1x128_mantissa_4/
```

Each directory contains:
- `config.json` - Model configuration
- `model.safetensors` or `pytorch_model.bin` - Quantized weights
- `tokenizer_config.json`, `tokenizer.json` - Tokenizer files

## Expected Runtime

- **Per configuration:** ~2-5 minutes (depending on hardware)
- **Total sweep:** ~30-80 minutes for all 16 configurations

## Memory Requirements

- **RAM:** ~8-12 GB (model is copied for each configuration)
- **Disk Space:** ~1-2 GB per quantized model (16-32 GB total)

## Quantization Details

### Block Floating Point (BFP)

BFP quantization:
1. Divides weight matrix into 2D blocks
2. Each block shares a common exponent (determined by max absolute value)
3. Mantissa values are quantized to specified bit-width
4. Reconstruction: `reconstructed = (quantized_mantissa / Q_max) * 2^exponent`

### Block Size Trade-offs

- **Tall blocks (e.g., 128×1):** Better for channel-wise patterns
- **Wide blocks (e.g., 1×128):** Better for input feature patterns  
- **Square blocks (e.g., 8×16, 16×8):** Balance between both dimensions
- **Smaller blocks:** More exponents, higher precision, larger overhead
- **Larger blocks:** Fewer exponents, lower precision, smaller overhead

## Example Output

```
Configuration 1/16
Block Size: 128×1, Mantissa Bits: 5
================================================================================
Creating model copy...

Quantizing with block=128×1, mantissa=5:
  [  1] model.layers.0.self_attn.q_proj                (2048, 2048) MSE: 1.234567e-05
  [  2] model.layers.0.self_attn.k_proj                (2048, 2048) MSE: 2.345678e-05
  ...

Quantized 198 layers

Testing generation...
Generated: Hello, how are you? I'm doing well, thank you for asking...

Saving to: ../model/tinyllama/TinyLlama_1.1v-bfp2d-sweep/block_128x1_mantissa_5
Saved successfully!

Configuration 1 completed
```

## Troubleshooting

### ImportError: No module named 'transformers'

Install transformers:
```bash
pip install transformers
```

### CUDA Out of Memory

The script uses CPU by default. If you encounter memory issues:
- Use `--skip-generation-test` to reduce memory usage
- Close other applications
- Use `--dry-run` for testing without saving

### Model Not Found

Ensure the model path is correct:
- Default preset: `../model/tinyllama/TinyLlama_1.1v`
- Or provide custom path with `--model /path/to/model`

## Related Files

- `block_quantization_2d.py` - Core 2D BFP quantization implementation
- `bfp_quantize_2d.py` - Single-configuration BFP quantization script
- `awq_quantize_2d.py` - Activation-aware quantization variant
