# TinyLlama 2D Quantization Scripts

This directory contains a comprehensive suite of tools for applying 2D block-based quantization to the TinyLlama model. It supports various methods, including Block Floating Point (BFP), Activation-Aware Weight Quantization (AWQ), and different storage formats.

## ⚡ Recommended Workflow: Runtime Quantization

To save significant disk space and allow for rapid experimentation, we strongly recommend using **runtime quantization**. This approach avoids saving multiple large model files. Instead, it applies quantization on-the-fly to a single base model.

**Key advantages:**
- **Minimal Disk Usage**: Requires only one BF16 model (~2.2GB), instead of ~100GB for all 48 configurations.
- **Fast Iteration**: Test new quantization configurations in seconds without creating new models.
- **Seamless Evaluation**: Integrates directly with evaluation scripts.

### Quick Start

Run evaluation with a specific quantization configuration without saving the model:

```bash
# Evaluate a BFP configuration
python evaluate_with_runtime_quantization.py \
    --method bfp \
    --block-height 16 \
    --block-width 16 \
    --mantissa-bits 4

# Evaluate an AWQ configuration
python evaluate_with_runtime_quantization.py \
    --method awq-fix \
    --block-height 32 \
    --block-width 4 \
    --mantissa-bits 5
```

For more details, see the [Runtime Quantization Guide](./docs/RUNTIME_QUANTIZATION.md).

## Scripts Overview

This directory provides scripts for both comprehensive sweeps (saving models) and runtime evaluation.

| Script | Description |
|---|---|
| `evaluate_with_runtime_quantization.py` | **(Recommended)** Applies quantization at runtime for evaluation without saving the model. |
| `run_quantize_tinyllama_comprehensive.sh` | Runs a full sweep of all 48 quantization configurations (BFP, AWQ-Fix, AWQ-Mix) and saves each model. **Warning: Uses >100GB of disk space.** |
| `block_quantization.py` | Contains the core 2D quantization logic for BFP and AWQ. |
| `runtime_quantizer.py` | Implements the logic for applying quantization at runtime. |

For a complete list and description of all scripts, see the [Scripts Overview](./docs/README_SCRIPTS_OVERVIEW.md).

## Quantization Methods

This toolkit supports three primary quantization methods:

| Method | Speed | Accuracy | Representation | Use Case |
|---|---|---|---|---|
| **BFP** | ⚡⚡⚡ Fast | ⭐⭐ Good | All BFP | Quick experiments, baseline |
| **AWQ Fix** | ⚡⚡ Medium | ⭐⭐⭐ Better | All BFP (scaled) | Production, deployment-friendly |
| **AWQ Mix** | ⚡ Slow | ⭐⭐⭐⭐ Best | FP32 + BFP | Maximum accuracy needed |

For a deep dive into how these methods work, see the [Comprehensive 2D Quantization Guide](./docs/README_COMPREHENSIVE_2D.md).

## Detailed Documentation

For more in-depth information, please refer to the documents in the `docs/` directory:

- **[Comprehensive 2D Quantization](./docs/README_COMPREHENSIVE_2D.md)**: Detailed guide to the `quantize_tinyllama_2d_comprehensive.py` script, methods, and configurations.
- **[Runtime Quantization](./docs/RUNTIME_QUANTIZATION.md)**: Guide to the recommended space-saving workflow.
- **[2D Block Quantization](./docs/README_2D_QUANTIZATION.md)**: Technical details of the 2D block quantization implementation.
- **[Scripts Overview](./docs/README_SCRIPTS_OVERVIEW.md)**: A quick reference for all scripts in this directory.
- **[Storage Format](./docs/STORAGE_FORMAT.md)**: An important explanation of how quantized models are stored and why file sizes are not immediately reduced.
- **[BFP Sweep](./docs/README_TINYLLAMA_2D_SWEEP.md)**: Guide to the BFP-only sweep script.
