# Implementation: Multiplying Matrices Without Multiplying (MADDNESS) for TinyLlama

## Overview
This document describes the implementation of the MADDNESS (Multiply-ADDition-lESS) method for the TinyLlama model. The goal is to approximate matrix multiplications in Linear layers using hashing and table lookups, as described in the MADDNESS paper.

## Implementation Details

### 1. Core Logic (`maddness.py`)
The `maddness.py` file contains the core components:
- **`MaddnessLayer`**: A custom PyTorch module that replaces `nn.Linear`. It performs inference using:
    - **Hashing**: A Balanced Binary Regression Tree maps input vectors to prototype indices.
    - **Lookup**: Precomputed tables ($T = P \times W$) are accessed using the indices.
    - **Aggregation**: Partial results from subspaces are summed to produce the output.
- **`MaddnessTrainer`**: Handles the training of the hash function and prototypes:
    - **Tree Construction**: Uses a greedy approach to find split indices and thresholds that minimize variance.
    - **Prototype Optimization**: Uses Ridge Regression to find optimal prototypes that reconstruct the input activations.
    - **Optimization**: The `_find_best_split` function uses a quantile-based approach (checking 10 quantiles) instead of iterating through all unique values, significantly reducing training time.
    - **Error Evaluation**: Calculates the Mean Squared Error (MSE) between the original matrix multiplication output and the MADDNESS approximation on the calibration data.

### 2. Execution Script (`run_maddness.py`)
The `run_maddness.py` script orchestrates the entire process:
1.  **Model Loading**: Loads `TinyLlama_1.1v` and the `wikitext` dataset.
    - *Note*: `tokenizer.json` was renamed to `tokenizer.json.bak` to force usage of `tokenizer.model` (slow tokenizer) which is then loaded as a fast tokenizer by `AutoTokenizer`. This resolves compatibility issues with the evaluation script.
2.  **Evaluation (Baseline)**: Evaluates baseline Perplexity (Wikitext) and HellaSwag Accuracy.
3.  **Model Loading**: Checks if `maddness_tinyllama.pt` exists.
    - **If exists**: Loads the saved model state dict and skips training.
    - **If not**:
        - Collects calibration data.
        - Trains MADDNESS layers for all Linear layers.
        - Reports the average Quantization Error (MSE) on training activations.
        - Replaces original layers.
        - Saves the model to `maddness_tinyllama.pt`.
4.  **Evaluation (MADDNESS)**: Evaluates MADDNESS Perplexity and HellaSwag Accuracy.

## Usage
To run the implementation:
```bash
python run_maddness.py
```

## Results
We evaluated the implementation on `wikitext` (Perplexity) and `HellaSwag` (Accuracy).

| Model | Perplexity | HellaSwag Acc | HellaSwag Acc Norm | Avg MSE | Notes |
|-------|------------|---------------|--------------------|---------|-------|
| Baseline | 112.96 | 0.4610 | 0.6184 | - | |
| MADDNESS (Full) | 7569.10 | 0.2500 | 0.3000 | 0.1946 | Full replacement (154 layers), Subspaces=4, Depth=4 |

### Analysis
- **Performance**: The training time is now very fast (~6 minutes for 154 layers) due to the optimization.
- **Quality**: The current configuration (Subspaces=4, Depth=4) results in high perplexity and low accuracy when applied to all layers. The average MSE of ~0.19 suggests a significant approximation error.
- **Saving/Loading**: The model is successfully saved to `maddness_tinyllama.pt` and can be reloaded to reproduce results without retraining.

### Configuration
The current configuration uses:
- **Subspaces**: 4
- **Prototypes**: 16 (Depth 4 tree)
- **Calibration Samples**: 50
- **HellaSwag Samples**: All (10042)
