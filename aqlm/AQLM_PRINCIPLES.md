# AQLM: Principles and Code Function

This document explains the principles of Additive Quantization for Language Models (AQLM) and how they are implemented in this codebase.

## 1. Core Principle: Additive Quantization (AQ)

Standard quantization maps a weight $w$ to the nearest value in a discrete codebook. AQLM extends this by representing a weight vector $\mathbf{w}$ as the **sum** of multiple codebook vectors:

$$ \mathbf{w} \approx \sum_{m=1}^{M} \mathbf{c}_{m, k_m} $$

Where:
- $M$ is the number of **codebooks**.
- $\mathbf{c}_{m, k_m}$ is the vector from the $m$-th codebook at index $k_m$.
- The indices $\{k_1, \dots, k_M\}$ are the **codes** for this weight vector.

This "additive" approach allows for a much richer representation (exponentially larger effective codebook size) without increasing the storage cost significantly, as we only store the indices and the small codebooks.

## 2. Code Structure Overview

The implementation is divided into the following key files:

### `main.py`
**The Entry Point.**
-   **Function**: Orchestrates the entire quantization process.
-   **Flow**:
    1.  Loads the pre-trained model (e.g., TinyLlama).
    2.  Loads calibration data (e.g., C4, WikiText).
    3.  Iterates through the model layer by layer.
    4.  Captures input activations (`inps`) for the current layer.
    5.  Passes the layer to `aq_engine` for quantization.
    6.  Performs "finetuning" (optimizing the quantized weights to minimize output error).
    7.  Saves the quantized layer and moves to the next.

### `aq_engine.py`
**The Optimizer.**
-   **Class `AQEngine`**: Manages the optimization for a single linear layer.
-   **Key Methods**:
    -   `add_batch()`: Accumulates the input covariance matrix $X^T X$ (XTX). This is crucial for minimizing the activation error $\| (W - \hat{W})X \|^2$ rather than just weight error.
    -   `quantize()`: Runs the iterative optimization loop:
        1.  **Initialization**: Uses Residual K-Means to find initial codebooks.
        2.  **Beam Search**: Finds the best discrete codes for each weight vector.
        3.  **Adam Optimization**: Updates the codebook values (vectors) to minimize error.

### `src/aq.py`
**The Mathematics & Data Structures.**
-   **Class `QuantizedWeight`**: The core data structure holding:
    -   `codebooks`: The dictionary of vectors.
    -   `codes`: The indices for each weight group.
    -   `scales`: Optional channel-wise or group-wise scaling factors.
-   **Class `QuantizedLinear`**: A PyTorch module that replaces `nn.Linear`. It performs the forward pass using the quantized weights.
-   **Function `init_aq_kmeans`**: Implements the Residual K-Means initialization strategy.

### `src/utils.py`
**Utilities.**
-   **Function `_dequantize_weight`**: The critical function that reconstructs the approximate float weights from codes and codebooks.
    -   It uses `F.embedding_bag` to efficiently sum the vectors from multiple codebooks based on the indices.

## 3. The Quantization Pipeline

1.  **Calibration**: We pass real data through the model to collect input statistics ($X$). This allows us to weigh the importance of different weights based on how much they affect the activations.
2.  **Layer-wise Processing**: To save memory, we quantize one layer at a time, keeping the rest of the model on CPU or disk.
3.  **Optimization Loop**:
    -   **Step 1: Update Codes**: Given fixed codebooks, find the best combination of indices for each weight vector. This is a discrete optimization problem solved via **Beam Search**.
    -   **Step 2: Update Codebooks**: Given fixed codes, update the vector values in the codebooks. This is a continuous optimization problem solved via **Gradient Descent (Adam)**.
    -   These steps are alternated until convergence.
4.  **Finetuning**: After finding the best discrete representation, we do a few epochs of end-to-end training (on the calibration data) for that layer to further refine the codebooks and scales.

## 4. Inference

The inference script (`inference_from_scratch.py`) demonstrates how to use the artifacts:
1.  **Load Config**: Creates an empty model skeleton.
2.  **Load Artifacts**: Reads the `.pth` files containing `QuantizedLinear` modules.
3.  **Dequantize**: Reconstructs the FP16/FP32 weights using `_dequantize_weight` (Sum of codebook vectors).
4.  **Forward**: Runs the standard model forward pass with these reconstructed weights.

*Note: In a highly optimized inference engine (like a CUDA kernel), the dequantization happens on-the-fly during matrix multiplication to save memory bandwidth. Our Python implementation dequantizes to memory for simplicity and demonstration.*
