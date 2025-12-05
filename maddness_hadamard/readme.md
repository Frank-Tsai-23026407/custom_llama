# Maddness with Hadamard Transform

This project implements the "multiplying matrices without multiplying" (Maddness) algorithm with the integration of the Hadamard transform to mitigate the effect of outliers in activation vectors, inspired by QuIP#.

## 1. Motivation: The Outlier Problem

In Large Language Models (LLMs), activation matrices often exhibit **"outlier features"**—specific dimensions (channels) that have significantly larger magnitudes than others. These outliers make quantization difficult because:

1.  **High Dynamic Range**: Quantization parameters (scale/zero-point) are dominated by these outliers, causing fine-grained details in other channels to be lost (quantization error).
2.  **High Kurtosis**: The distribution of activations is highly "peaked" (high Kurtosis), meaning most values are small but a few are huge.

### What is Kurtosis?
**Kurtosis** is a statistical measure of the "tailedness" of a distribution.
*   **High Kurtosis (>3)**: Indicates heavy tails or outliers. For example, original Llama activations often have Kurtosis > 1000.
*   **Low Kurtosis (~0)**: Indicates a flatter, more uniform distribution (like Gaussian).

By reducing Kurtosis, we make the data easier to approximate with low-bit quantization.

## 2. Solution: Hadamard Transform (Incoherence Processing)

To solve this, we apply a **Randomized Hadamard Transform** (or simply a fixed Hadamard transform) to "mix" the features. This rotation spreads the energy of outliers across all dimensions, making the distribution more Gaussian-like (lower Kurtosis).

### Mathematical Formulation

Let $X \in \mathbb{R}^{N \times D_{in}}$ be the input activation matrix and $W \in \mathbb{R}^{D_{out} \times D_{in}}$ be the weight matrix. The original linear layer computes:
$$Y = X W^T$$

We introduce orthogonal Hadamard matrices $H_{in} \in \mathbb{R}^{D_{in} \times D_{in}}$ and $H_{out} \in \mathbb{R}^{D_{out} \times D_{out}}$. Since Hadamard matrices are symmetric and orthogonal (when normalized), $H = H^T$ and $H H^T = I$.

The transformation steps are:

#### 1. Transform Input ($X'$)
We rotate the input $X$ by multiplying with $H_{in}$:
$$X' = X H_{in}^T$$
*Effect: This "smears" the outliers in $X$ across all columns of $X'$.*

#### 2. Transform Weights ($W'$)
We adjust the weights so that the mathematical equivalence holds.
$$W' = H_{out} W H_{in}^T$$
*Note: The weights are rotated on both sides to match the input rotation and the output rotation.*

#### 3. Computation ($Y'$)
We compute the matrix multiplication (or Maddness approximation) in the transformed domain:
$$Y' = X' (W')^T$$

Substituting the terms to verify equivalence:
$$Y' = (X H_{in}^T) (H_{out} W H_{in}^T)^T$$
$$Y' = X H_{in}^T H_{in} W^T H_{out}^T$$
Since $H_{in}^T H_{in} = I$:
$$Y' = X W^T H_{out}^T = Y H_{out}^T$$

#### 4. Inverse Transform Output ($Y$)
Finally, we rotate the output back to the original space:
$$Y_{recovered} = Y' H_{out}$$
$$Y_{recovered} = (Y H_{out}^T) H_{out} = Y I = Y$$

### Why this works?
By processing $X'$ instead of $X$, the quantization (Maddness) sees a "smoother" matrix with no extreme outliers. This allows the product quantization (PQ) or other compression methods to work much more effectively.

## Codebase Overview

### 1. `maddness_hadamard.py`
Core implementation of the Maddness algorithm with Hadamard transform support.

*   **`get_hadamard_matrix(n, device)`**: Generates a normalized Hadamard matrix of size $n \times n$. Recursively constructs the matrix using the Sylvester construction.
*   **`class MaddnessHadamardLayer(nn.Module)`**: A custom PyTorch layer that replaces a standard `nn.Linear` layer.
    *   **`__init__`**: Initializes the layer, registers Hadamard matrices ($H_{in}, H_{out}$) as buffers, and sets up Maddness parameters (split indices, thresholds, lookup table).
    *   **`forward(x)`**: Applies the forward pass:
        1.  Transforms input $X$ using $H_{in}$: $X' = X H_{in}^T$.
        2.  Performs Maddness approximate matrix multiplication using the lookup table.
        3.  Transforms output $Y'$ using $H_{out}$: $Y = Y' H_{out}$.
*   **`class MaddnessHadamardTrainer`**: Handles the training of the Maddness layer parameters (codebooks).
    *   **`train(X_calib, W)`**:
        1.  Applies Hadamard transform to calibration data $X$ and weights $W$.
        2.  Builds the decision trees (split indices/thresholds) using variance-based splitting.
        3.  Optimizes prototypes (leaf values) using least squares.
        4.  Constructs the lookup table.

### 2. `analyze_distribution.py`
Tools for analyzing and visualizing the distribution of activations and weights.

*   **`get_hadamard_matrix(n, device)`**: Same utility as in `maddness_hadamard.py`.
*   **`analyze_activations(model, tokenizer, dataset, ...)`**: Runs the model on a dataset and collects activation statistics (Kurtosis, Min, Max) for specified layers.
*   **`plot_3d_activations(activations, tokens, title, save_path)`**: Generates a 3D line plot visualizing activations for a specific sequence.
    *   **X-axis**: Tokens in the sequence.
    *   **Y-axis**: Channel indices.
    *   **Z-axis**: Activation magnitude.
*   **`analyze_specific_sample(model, tokenizer, text)`**: Captures activations for a single text sample to be used for plotting.
*   **`main()`**:
    1.  Runs specific sample analysis and generates 3D plots (Original vs Hadamard).
    2.  Runs bulk analysis on WikiText-2 to report Kurtosis and Range statistics in `analysis_results.txt`.

### 3. `run_maddness_hadamard.py`
Main script for end-to-end training and evaluation.

*   **`get_calibration_data(...)`**: Collects input activations from the model to be used for training Maddness layers.
*   **`evaluate_perplexity(...)`**: Evaluates the model's perplexity on the WikiText-2 dataset.
*   **`evaluate_hellaswag_wrapper(...)`**: Evaluates the model's accuracy on the HellaSwag benchmark.
*   **`main()`**:
    1.  Loads the pre-trained TinyLlama model.
    2.  Checks for an existing Maddness checkpoint.
    3.  **If checkpoint exists**: Loads the converted model.
    4.  **If no checkpoint**:
        *   Collects calibration data.
        *   Iterates through linear layers, trains `MaddnessHadamardLayer` replacements, and swaps them in.
        *   Saves the converted model.
    5.  Evaluates the final model on Perplexity and HellaSwag.
    6.  Saves results to `maddness_hadamard_results.txt`.

## Maddness Configuration

### Tree Depth
The depth of the decision trees used in Maddness determines the granularity of the quantization.
*   **Parameter**: `TREE_DEPTH` (default is typically 4 or 10 depending on the script).
*   **Effect**: A tree of depth $D$ has $2^D$ leaf nodes (prototypes).
    *   Depth 4: $2^4 = 16$ prototypes per subspace.
    *   Depth 10: $2^{10} = 1024$ prototypes per subspace.
*   **Trade-off**: Deeper trees provide higher accuracy (lower quantization error) but require more memory for the lookup table and slightly more computation during the hashing phase.

### Subspaces
*   **Parameter**: `NUM_SUBSPACES` (default 4).
*   **Effect**: The input vector is split into this many chunks. Each chunk is processed by its own decision tree.
*   **Total Prototypes**: `NUM_SUBSPACES` * $2^{\text{TREE\_DEPTH}}$.

### Why Power of 2?
The Hadamard transform implemented here uses the **Sylvester construction**, which is defined recursively:
$$H_1 = [1]$$
$$H_{2n} = \begin{bmatrix} H_n & H_n \\ H_n & -H_n \end{bmatrix}$$
This construction naturally produces matrices of size $1, 2, 4, 8, \dots, 2^k$.
*   **Efficiency**: This structure allows for the Fast Walsh-Hadamard Transform (FWHT), which reduces computational complexity from $O(N^2)$ to $O(N \log N)$.
*   **Handling Non-Power-of-2**: For dimensions like 5632 (which is $11 \times 512$), we typically **pad** the vector with zeros to the next power of 2 (e.g., 8192) or use block-diagonal approximations, though simple padding is the most common approach to enable the fast transform.

### 1. Analyze Distributions
To analyze the activation and weight distributions before and after Hadamard transform:
```bash
python3 maddness_hadamard/analyze_distribution.py
```
This will generate `analysis_results.txt` showing kurtosis and max values.

### 2. Run Maddness with Hadamard
To train and evaluate the Maddness model with Hadamard transform:
```bash
python3 maddness_hadamard/run_maddness_hadamard.py
```
This script will:
1.  Load the TinyLlama model.
2.  Calibrate and train Maddness layers with Hadamard transform.
3.  Replace the original layers.
4.  Evaluate Perplexity on WikiText-2.
5.  Evaluate Accuracy on HellaSwag.
6.  Save the model to `maddness_hadamard_tinyllama_4_10.pt`.

## Metrics Explanation

### 1. Kurtosis (Fisher's Definition)
The report uses **Fisher's Kurtosis** (also known as Excess Kurtosis), which measures the "tailedness" of the probability distribution.

$$ \text{Kurtosis} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{x_i - \mu}{\sigma} \right)^4 - 3 $$

*   **$\mu$**: Mean of the activations.
*   **$\sigma$**: Standard deviation of the activations.
*   **Interpretation**:
    *   **0**: Normal (Gaussian) distribution.
    *   **> 0 (Positive)**: Leptokurtic (Heavy tails / Outliers). The value `1036.37` indicates extreme outliers.
    *   **< 0 (Negative)**: Platykurtic (Light tails / Uniform-like). The value `-0.19` indicates a flatter distribution closer to uniform/Gaussian.

### 2. Min and Max (Range)
The **Min** and **Max** values report the actual range of the tensor values, replacing the previous "Max Absolute Value" metric. This gives a clearer picture of the distribution's spread.

*   **Original**: Large range (e.g., `Min=-6.46, Max=6.46`) due to outliers.
*   **Hadamard**: Compact range (e.g., `Min=-0.004, Max=0.004`), indicating that the energy has been spread out and outliers eliminated.

