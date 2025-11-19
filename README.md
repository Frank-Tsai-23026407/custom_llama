# Llama Quantization Explorer: A PyTorch-Based Framework for Model Compression Research

This repository provides a from-scratch, PyTorch-based implementation of the Llama architecture, designed specifically for research and development in language model quantization. It offers a comprehensive and modular toolkit for applying, evaluating, and analyzing advanced compression techniques like Activation-Aware Weight Quantization (AWQ). Whether you are a researcher exploring new quantization algorithms or a developer looking to optimize language models, this framework provides the tools you need for deep, fine-grained analysis.

## Core Components

The framework is built on three pillars: a customizable backend, a powerful quantization toolkit, and a rigorous evaluation testbench.

### 1. Custom Llama Backend

Located in the `llama_backend` directory, the custom backend is a pure PyTorch implementation of the Llama model. Its modular design and functional components (e.g., attention, RMSNorm) allow for easy modification and experimentation. Key features include:
- **Multiple Backends**: Seamlessly switch between a `'custom'` backend for experimentation, a `'huggingface'` backend for bit-perfect reference, and a `'clone'` backend for debugging.
- **Precision Policies**: A `PrecisionPolicy` class gives you fine-grained control over the data types used in different parts of the model, such as matrix multiplications and softmax computations.

### 2. Advanced Quantization Toolkit

The `quantize_model_script` directory houses a suite of scripts for applying state-of-the-art quantization techniques. The centerpiece is a from-scratch implementation of **Activation-Aware Weight Quantization (AWQ)**, which intelligently protects salient weights from quantization errors. The toolkit supports:
- **Fixed-Precision AWQ**: Apply uniform quantization with varying bit depths.
- **Mixed-Precision AWQ**: Explore strategies where different layers or blocks use different precision levels.
- **Block Floating-Point (BFP)**: A highly configurable BFP implementation serves as the underlying quantization format.

### 3. Rigorous Evaluation Testbench

The `testbench` provides a comprehensive suite of scripts for evaluating the performance of quantized models on standard academic benchmarks, including HellaSwag, ARC, BoolQ, and more. This allows for a thorough analysis of the trade-offs between model compression and task accuracy.

## Key Features at a Glance

- **Pure PyTorch Llama**: A custom Llama implementation (`llama_my.py`) for maximum flexibility.
- **Activation-Aware Weight Quantization (AWQ)**: A from-scratch implementation supporting both fixed and mixed-precision strategies.
- **Multiple Backends**: Compare your custom models against Hugging Face and a lightweight "clone" for parity checking.
- **Flexible Precision Control**: Use `PrecisionPolicy` to control dtypes for RoPE, softmax, and matrix multiplications.
- **Comprehensive Evaluation Suite**: Benchmark performance on tasks like HellaSwag, ARC, BoolQ, and more.
- **Extensive Model Zoo**: Comes with dozens of pre-quantized AWQ models with varying configurations.
- **Reproducibility**: All quantization and evaluation scripts are designed to be easily run and modified.

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

## Getting Started: A Quick Example

This short guide will walk you through quantizing the TinyLlama model and evaluating it on the HellaSwag benchmark.

### Step 1: Setup and Installation

First, clone the repository and set up the environment:

```bash
git clone git@github.com:Frank-Tsai-23026407/custom_llama.git
cd custom_llama

# Create a Conda environment
conda create -n llama-env python=3.11
conda activate llama-env

# Install dependencies
pip install -r requirements.txt
```

**System Requirements**:
- PyTorch 2.3.1+ with CUDA 12.1+
- A CUDA-compatible GPU is highly recommended for reasonable performance.

### Step 2: Quantize the Model

Next, run the fixed-precision AWQ script. We'll quantize the `TinyLlama` model with a block size of 64 and 4 mantissa bits.

```bash
python quantize_model_script/fix_precision_awq.py \
    --model tinyllama \
    --block-size 64 \
    --mantissa-bits 4
```

This will:
1. Load the base `TinyLlama_1.1v` model.
2. Use the WikiText dataset for calibration.
3. Apply AWQ and save the quantized model to `model/tinyllama/TinyLlama_1.1v-awq-quantized-fix-precision-b64-m4/`.

### Step 3: Evaluate the Quantized Model

Now, evaluate the model's performance on the HellaSwag benchmark using the custom backend.

```bash
# Define the path to your quantized model
export MODEL_CHECKPOINT="model/tinyllama/TinyLlama_1.1v-awq-quantized-fix-precision-b64-m4"

# Run the evaluation script
python testbench/task_script_hellaswag.py \
    --backend custom \
    --model_path "$MODEL_CHECKPOINT"
```

The script will output the model's accuracy. You can compare this to the original model's performance or experiment with different quantization settings.

## Detailed Usage

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
4. Save each quantized model to a new directory under `model/`, such as `model/tinyllama/TinyLlmam_1.1v-awq-quantized-fix-precision-b64-m2/`.

### 2. Model Evaluation

The `testbench/` directory contains scripts to evaluate models on various benchmarks. For example, to run the HellaSwag benchmark on a quantized model:

```bash
# Define the path to your quantized model
MODEL_CHECKPOINT="model/tinyllama/TinyLlmam_1.1v-awq-quantized-fix-precision-b64-m2"

# Run evaluation using the custom backend
python testbench/task_script_hellaswag.py \
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
