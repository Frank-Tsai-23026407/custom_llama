# AQLM: Additive Quantization for Language Models

This directory contains the restructured implementation of **AQLM compression**, optimized for modularity and ease of use.

## 🚀 Execution Directory
> **Important:** All commands (quantization, evaluation, inference) must be executed from the **`custom_model_quantize_inference/`** directory to ensure proper module resolution for `custom_aqlm`.
> ```bash
> cd custom_model_quantize_inference/
> ```

## Project Structure

The codebase is organized into the following modules:

- `aqlm/quantize/`: Scripts for model quantization, including the main entry point, k-means initialization, and beam search optimization.
- `aqlm/utils/`: Core components such as AQLM layer definitions, weight containers, and model manipulation utility functions.
- `aqlm/inference/`: Tools for model inference and performance evaluation (Perplexity, HellaSwag).
- `aqlm/model/`: A dedicated directory for storing and managing quantized model checkpoints.

---

## 1. Model Quantization

To quantize a model using AQLM, use the `aqlm.quantize.main` module.

### Basic Usage
```bash
python3 -m aqlm.quantize.main \
    <MODEL_PATH_OR_NAME> \
    <DATASET> \
    --nsamples 128 \
    --model_seqlen 2048 \
    --num_codebooks 1 \
    --nbits_per_codebook 16 \
    --in_group_size 8 \
    --save <OUTPUT_PATH>.pt
```

### Example Command (TinyLlama)

Quantizing TinyLlama with 1 codebook of 16 bits:

```bash
python3 -m custom_aqlm.quantize.main \
    TinyLlama/TinyLlama_v1.1 \
    wikitext2 \
    --nsamples 128 \
    --num_codebooks 2 \
    --nbits_per_codebook 8 \
    --in_group_size 8 \
    --save custom_aqlm/model/TinyLlama-v1.1-2x8g8
```

### Key Arguments
- `model_path`: Path to the local model or Hugging Face model ID (e.g., `TinyLlama/TinyLlama_v1.1` or `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
- `dataset`: Calibration dataset. Use `wikitext2`, `c4`, or `pajama`.
- `--num_codebooks`: Number of codebooks per layer (higher = better quality, more memory).
- `--nbits_per_codebook`: Number of bits for each codebook (typically 16).
- `--in_group_size`: Number of input features quantized together (typically 8).
- `--max_epochs`: Number of optimization epochs (default: 10).
- `--save`: Path to save the resulting quantized model weights.
- `--use_bfp`: (Optional) Use Block Floating Point for codebooks. Usually unnecessary as codebooks are small.

### Recommended Configurations (Balanced Settings)

| Target BPW | Config Name | `--num_codebooks` | `--nbits_per_codebook` | `--in_group_size` | Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1.0 bit** | `1x8g8` | 1 | 8 | 8 | Extreme compression, high precision loss. |
| **1.25 bit**| `1x10g8`| 1 | 10 | 8 | Balanced ultra-low bit. |
| **2.0 bit** | **`1x16g8`** | 1 | 16 | 8 | **Official standard.** Best accuracy for 2-bit. |
| **2.0 bit** | `2x8g8` | 2 | 8 | 8 | Faster inference than 1x16. |
| **3.0 bit** | `3x8g8` | 3 | 8 | 8 | High quality, good for small models. |
| **4.0 bit** | `2x8g4` | 2 | 8 | 4 | Near-lossless, very high fidelity. |

> **Note:** BPW (Bits Per Weight) = `(num_codebooks * nbits_per_codebook) / in_group_size`.

---

## 2. Evaluation

After quantization, you can evaluate the model's performance.

### Perplexity (PPL) Evaluation
The `main.py` script automatically runs evaluation if `--eval_ppl` is specified.
```bash
python3 -m custom_aqlm.quantize.main ... --eval_ppl
```

### HellaSwag Evaluation
To evaluate on the HellaSwag task:
```bash
python3 -m custom_aqlm.quantize.main ... --eval_hellaswag
```

### Evaluating a Saved Model
To evaluate a previously saved quantized model:
```bash
python3 -m custom_aqlm.inference.evaluate_quantized_model \
    --model_path <PATH_TO_SAVED_MODEL>.pt
```

### Evaluating Baseline (FP16)
To get the baseline performance of the original model (non-quantized):
```bash
python3 evaluate_baseline.py --model_id TinyLlama/TinyLlama_v1.1
```
This script evaluates the original model in FP16/BF16 precision on WikiText2 PPL and HellaSwag to provide a "Golden Reference" for your quantization experiments.

---

## 3. Inference

To run a simple generation test with a quantized model:

```bash
python3 -m custom_aqlm.inference.inference \
    --model_path <ORIGINAL_MODEL_PATH> \
    --quantized_path <QUANTIZED_WEIGHTS_PATH>.pt \
    --prompt "The future of AI quantization is"
```

