# Llama Quantization Explorer: A PyTorch-Based Framework for Model Compression Research

This repository provides a from-scratch, PyTorch-based implementation of the Llama architecture, designed specifically for research and development in language model quantization. It offers a comprehensive and modular toolkit for applying, evaluating, and analyzing advanced compression techniques like Activation-Aware Weight Quantization (AWQ). Whether you are a researcher exploring new quantization algorithms or a developer looking to optimize language models, this framework provides the tools you need for deep, fine-grained analysis.

## Quick Start

### Environment Setup

1. **Create and activate conda environment:**
   ```bash
   conda create -n llama-env python=3.11
   conda activate llama-env
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up local environment activation (optional):**
   ```bash
   # Copy the example activation script
   cp activate_env.sh.example activate_env.sh
   
   # Activate environment quickly in new terminals
   source activate_env.sh
   ```
   
   Note: `activate_env.sh` is git-ignored for local customization.

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
This document reflects the current repository layout and the purpose of the main folders and files.

### Root

```
custom_llama/
├── FILE_STRUCTURE.md
├── SIMPLE_EVAL.md
├── requirements.txt
├── debug/
├── docs/
├── llama_backend/
├── model/
├── model_analysis/
├── quantize_model_script/
└── testbench/
```

### `llama_backend/` — Core backend

Custom TinyLlama backend with precision policy and optional clone backend.

```
llama_backend/
├── USAGE.md
├── llama_my.py
├── precision_policy.py
├── utils.py
├── clone/
│   ├── clone_backend.py
│   ├── hf_clone.py
│   └── hf_rope.py
└── custom/               # (empty)
```

### `debug/` — Debug and tests

Small scripts used to validate attention, precision policies, and ground-truth comparisons.

```
debug/
├── compare_qkv_attention.py
├── test_precision_policy.py
└── tinyllama_gt.py
```

### `docs/` — Documentation

Notes and comparisons relevant to the backends and debugging.

```
docs/
├── backend_comparison.md
├── custom_vs_clone_code_comparison.md
├── debug_files_inventory.md
└── git_submodule_guide.md
```

### `model/` — Models and variants

Base model and multiple AWQ-quantized variants (mix/fix precision across block sizes and magnitude settings).

```
model/
├── TinyLlama_1.1v/
├── TinyLlama_1.1v-awq-quantized/
├── TinyLlama_1.1v-awq-quantized-mix-precision-b{32,64,128}-m{2,3,4,5}/
└── TinyLlama_1.1v-awq-quantized-fix-precision-b{32,64,128}-m{2,3,4,5}/
```

### `model_analysis/` — Plots and analysis

```
model_analysis/
├── plot_element_contribution/
└── plot_weight/
```

### `quantize_model_script/` — Quantization scripts

AWQ and block quantization utilities.

```
quantize_model_script/
├── activation_aware_weight_quantization.py
├── block_quantization.py
├── fix_precision_awq_tinyllama.py
└── mix_precision_awq_tinyllama.py
```

### `testbench/` — AWQ experiments and runs

Docs, run scripts, and evaluation helpers for AWQ and BFP runs.

```
testbench/
├── AWQ.md
├── AWQ_INTEGRATION_COMPLETE.txt
├── AWQ_MODELS_HELLASWAG.md
├── NEW_FILES_SUMMARY.md
├── PRECISION_SWEEP_SUMMARY.md
├── README_PRECISION_SWEEP.md
├── awq/
├── log/
├── run_all.sh
├── run_block_floating_point.sh
├── run_dynamic_awq_experiments.sh
├── run_precision_sweep_hellaswag.sh
├── run_precision_sweep_quick.sh
├── run_static_awq_evaluation.sh
├── run_static_awq_experiments.sh
├── test_awq_integration.sh
├── task_script_arc_c.py
├── task_script_arc_e.py
├── task_script_boolq.py
├── task_script_hellaswag.py
├── task_script_obqa.py
├── task_script_piqa.py
├── task_script_winogrande.py
└── validate_modifications.sh
```

## Getting Started

### Environment Setup

First, clone the repository and set up the environment.

```bash
git clone git@github.com:Frank-Tsai-23026407/custom_llama.git
cd custom_llama

# Create a Conda environment
conda create -n llama-env python=3.11
conda activate llama-env

# Install dependencies
pip install -r requirements.txt
```

**Critical dependencies**:
- PyTorch 2.3.1 with CUDA 12.1 (not 2.4.x - Triton version conflict)
- Triton 2.3.1 (required by bitsandbytes 0.43.3)
- bitsandbytes 0.43.3 (GPU-enabled version with proper CUDA runtime)

**Installation**:
```bash
# Recommended: Use PyTorch CUDA wheels
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install -r requirements.txt
```

### Model Quantization

You can generate quantized models using the scripts in `quantize_model_script/`. For example, to run fixed-precision AWQ on the base TinyLlama model:

The script `fix_precision_awq_tinyllama.py` is configured to run a sweep of quantizations. You can modify the parameters within the script or run it directly.

```bash
python quantize_model_script/fix_precision_awq_tinyllama.py
```

This will:
1. Load the base `TinyLlama_1.1v` model.
2. Use the WikiText dataset for calibration.
3. Apply AWQ with a block size of 64 and mantissa bits from 2 to 5.
4. Save each quantized model to a new directory under `model/`, such as `model/tinyllama/TinyLlama_1.1v-awq-quantized-fix-precision-b64-m2/`.

## Evaluation Guide

This guide shows how to run evaluations on TinyLlama models using two methods:
1. **lm-evaluation-harness** (standard framework)
2. **Custom evaluation scripts** (project-specific)

### Method 1: Using `lm-evaluation-harness`

#### Installation

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..
```

#### Basic Usage

To run a single task evaluation on 100 samples for a quick test:

```bash
lm_eval --model hf \
    --model_args pretrained=TinyLlama/TinyLlama_v1.1 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --limit 100
```

#### Evaluate Quantized Models

```bash
# Mix-Precision AWQ (m5, b128)
lm_eval --model hf \
    --model_args pretrained=model/tinyllama/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m5 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

### Method 2: Using Custom Evaluation Scripts

Custom scripts in the `testbench/` directory provide precision-aware evaluation with our custom backends.

#### Basic HellaSwag Evaluation

To run a quick test on 100 samples:
```bash
python testbench/task_script_hellaswag.py \
    --model_path model/tinyllama/TinyLlama_1.1v \
    --max_samples 100
```

#### Evaluate AWQ Quantized Models

```bash
# Mix-Precision AWQ (m5, 5-bit mantissa)
python testbench/task_script_hellaswag.py \
    --model_path model/tinyllama/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m5

# Fix-Precision AWQ (m5, 5-bit mantissa)
python testbench/task_script_hellaswag.py \
    --model_path model/tinyllama/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m5
```

#### Batch Evaluation Scripts

For a comprehensive evaluation across multiple precision settings and models, you can use the provided shell scripts:

```bash
# Run a quick sweep across different AWQ models (100 samples)
bash testbench/run_precision_sweep_quick.sh

# Run a full sweep on HellaSwag
bash testbench/run_precision_sweep_hellaswag.sh
```

Results are logged to the `testbench/log/` directory. You can easily parse the accuracy from the logs:
```bash
grep -h 'Accuracy (acc_norm):' testbench/log/precision_sweep/*.log
```

### Comparison: `lm-eval` vs Custom Scripts

*   **`lm-evaluation-harness`**:
    *   **Pros**: Standardized, many tasks available, easy to compare with other models.
    *   **Cons**: Less control over precision, harder to debug custom backends.
*   **Custom Scripts**:
    *   **Pros**: Full control over precision policies, supports custom backends, easier to debug.
    *   **Cons**: Need to implement each task, less standardized.

## Models and Results

The `model/` directory contains the base TinyLlama model and dozens of AWQ-quantized versions generated from the quantization scripts. The naming convention indicates the quantization strategy:

- `...-fix-precision-b{block_size}-m{mantissa_bits}`
- `...-mix-precision-b{block_size}-m{mantissa_bits}`

Summaries of benchmark results can be found in the `testbench/` directory, such as `AWQ_MODELS_HELLASWAG.md` and `PRECISION_SWEEP_SUMMARY.md`.
