# MADDNESS with Outlier Removal

This directory contains an implementation of the MADDNESS quantization method improved by explicit outlier handling.

## Motivation
Standard quantization methods often suffer from high errors due to activation outliers (values with large magnitudes). These outliers distort the quantization parameters (like centroids or split thresholds), leading to poor reconstruction of the "inlier" values which constitute the majority of the data.

## Method

### 1. Training Phase: Outlier-Aware Learning
**Goal**: Learn quantization parameters (hash trees and prototypes) that best represent the bulk of the data (inliers), ignoring the skew caused by outliers.

-   **Implementation**: In `MaddnessOutlierTrainer`, we calculate the 95th percentile threshold for each input channel.
-   **Action**: During the training of the MADDNESS trees and LUTs, we clip or filter out the top 5% of values based on magnitude. This ensures the learned structures focus on minimizing error for the most frequent values.

### 2. Inference Phase: Hybrid Computation
**Goal**: Preserve accuracy by handling outliers with high precision while compressing the rest.

-   **Implementation**: In `MaddnessOutlierLayer`.
-   **Step A (Identify Outliers)**: For each input vector, we identify the indices of the top-$k$ values with the largest magnitudes (default $k=32$).
-   **Step B (Exact Computation)**: We perform exact Multiply-Accumulate (MAC) operations for these $k$ outlier values using the original high-precision weights.
-   **Step C (Quantized Computation)**: The remaining values (with outliers zeroed out) are processed using the standard MADDNESS approximate matrix multiplication (hashing + LUT lookup).
-   **Step D (Combine)**: The final output is the sum of the exact outlier contribution and the approximate inlier contribution.

## Code Structure

-   **`maddness_outlier.py`**: Contains the core classes:
    -   `MaddnessOutlierLayer`: The `nn.Module` replacement that performs the hybrid inference.
    -   `MaddnessOutlierTrainer`: The trainer class that handles outlier removal during the learning phase.
-   **`run_maddness_outlier.py`**: A script to run the full pipeline:
    -   Loads the TinyLlama model.
    -   Calibrates and trains the outlier-aware MADDNESS layers.
    -   Evaluates Perplexity (Wikitext-2) and Accuracy (HellaSwag).

## Usage

To run the experiment and verify the results:

```bash
python maddness_remove_outlier/run_maddness_outlier.py
```

The script will save the evaluation results to `results.txt` in this directory.
