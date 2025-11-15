# Custom TinyLlama Implementation

This project is a from-scratch implementation of the Llama architecture in PyTorch (not only for TinyLlama). The primary focus of this repository is to explore, implement, and evaluate advanced weight quantization techniques, particularly Activation-Aware Weight Quantization (AWQ) and can be extend to all kind of quantization techniques and backends.

It includes a comprehensive suite of tools for quantizing the model, running evaluations on various benchmarks, and comparing the performance of different precision settings against reference implementations.

## Key Features

- **Custom Llama Backend**: A hand-coded Llama model (`llama_my.py`) built from the ground up in PyTorch.
- **Advanced Quantization**: Implementation of Activation-Aware Weight Quantization (AWQ) with support for both fixed and mixed-precision strategies.
- **Flexible Precision Control**: A `PrecisionPolicy` class to easily manage dtypes for computation, RoPE cache, and softmax operations.
- **Comprehensive Evaluation Suite**: Scripts to benchmark model performance on tasks like HellaSwag, ARC, BoolQ, and more.
- **Comparative Analysis**: Easily compare the custom implementation against Hugging Face's transformer implementation and a "clone" backend.
- **Extensive Experimentation**: Includes numerous pre-quantized AWQ models with varying block sizes and mantissa bits, along with scripts to reproduce them.

## Project Structure

```
/
├───llama_backend/      # Core model implementations (custom, clone)
├───quantize_model_script/ # Scripts for AWQ model quantization
├───model/              # Pre-trained and quantized model checkpoints
├───testbench/          # Evaluation scripts and benchmark results
├───debug/              # Debugging and comparison utilities
└───requirements.txt    # Project dependencies
```

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd custom_llama
```

### 2. Install Dependencies

Ensure you have a Python environment with PyTorch `2.1.0` or newer. Then, install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

The project is structured around two main workflows: quantizing a model and evaluating its performance.

### 1. Model Quantization

You can generate quantized models using the scripts in `quantize_model_script/`. For example, to run fixed-precision AWQ on the base TinyLlama model:

The script `fix_precision_awq_tinyllama.py` is configured to run a sweep of quantizations. You can modify the parameters within the script or run it directly.

```bash
python quantize_model_script/fix_precision_awq_tinyllama.py
```

This will:
1. Load the base `TinyLlama_1.1v` model.
2. Use the WikiText dataset for calibration.
3. Apply AWQ with a block size of 64 and mantissa bits from 2 to 5.
4. Save each quantized model to a new directory under `model/`, such as `model/TinyLlama_1.1v-awq-quantized-fix-precision-b64-m2/`.

### 2. Model Evaluation

The `testbench/` directory contains scripts to evaluate models on various benchmarks. For example, to run the HellaSwag benchmark on a quantized model:

```bash
# Define the path to your quantized model
MODEL_CHECKPOINT="model/TinyLlama_1.1v-awq-quantized-fix-precision-b64-m2"

# Run evaluation using the custom backend
python testbench/tinyllama_my_bfp_hellaswag.py \
    --backend custom \
    --model_path "$MODEL_CHECKPOINT"
```

The evaluation scripts offer several flags for detailed analysis:
- `--backend`: Choose between `custom`, `clone`, or `huggingface`.
- `--compute_dtype`: Set the computation precision (e.g., `bf16`, `fp32`).
- `--rope_cache_dtype`: Set the RoPE cache precision.
- `--softmax_fp32`: Use `fp32` for softmax calculations.

For a comprehensive evaluation across multiple precision settings and models, you can use the provided shell scripts:

```bash
# Run a sweep across different AWQ models on HellaSwag
bash testbench/run_precision_sweep_hellaswag.sh
```

Results are logged to the `testbench/log/` directory. You can easily parse the accuracy from the logs:
```bash
grep -h 'Accuracy (acc_norm):' testbench/log/precision_sweep/*.log
```

## Models and Results

The `model/` directory contains the base TinyLlama model and dozens of AWQ-quantized versions generated from the quantization scripts. The naming convention indicates the quantization strategy:

- `...-fix-precision-b{block_size}-m{mantissa_bits}`
- `...-mix-precision-b{block_size}-m{mantissa_bits}`

Summaries of benchmark results can be found in the `testbench/` directory, such as `AWQ_MODELS_HELLASWAG.md` and `PRECISION_SWEEP_SUMMARY.md`.
