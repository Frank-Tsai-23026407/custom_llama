# Multiplying Matrices Without Multiplying

Based on the attached paper ["Multiplying Matrices Without Multiplying" (MADDNESS)](https://arxiv.org/abs/2106.10860), here is a detailed description of the principle and a guide on applying it to language models.

### **File: MADDNESS_Principle_and_Application.md**

## 1. Principle Overview

The MADDNESS (Multiply-ADDition-lESS) method approximates the matrix multiplication $Y = AB$ where $A$ is an input data matrix (e.g., activations) and $B$ is a weight matrix. Instead of performing $N \times M$ dot products requiring floating-point multiply-adds, it reduces the operation to **hashing**, **table lookups**, and **aggregation**.

The core idea is to approximate a row vector $a$ (from $A$) as a sum of selected "prototypes" from a codebook. If $a$ is represented by a prototype $p$, then the dot product $a \cdot b$ can be approximated as $p \cdot b$. Since $B$ is static (in inference), the products $p \cdot b$ can be precomputed and stored in a lookup table.

The runtime process becomes:

1. **Encode**: Map the input vector $a$ to a set of prototype indices using a highly efficient **hash function** (instead of searching for the nearest neighbor).
2. **Lookup**: Use the indices to retrieve precomputed partial sums from a table constructed from $B$.
3. **Aggregate**: Sum the retrieved values to get the final output.

***

## 2. MADDNESS Hash Function

The hash function $g(x)$ is designed to be extremely fast and SIMD-friendly. Unlike traditional Locality Sensitive Hashing (LSH) that uses random projections, MADDNESS uses a structured **Balanced Binary Regression Tree**.

### Mechanics

* **Structure**: The hash function is a binary decision tree of depth 4 (resulting in $2^4 = 16$ leaf nodes/buckets per subspace).
* **Input**: A sub-vector $x$ (a slice of the input row $A$).
* **Process**:

1. Start at the root.
2. At level $t$, compare the value at a specific index $x_{j_t}$ against a node-specific threshold $v_{node}$.
3. If $x_{j_t} > v_{node}$, move to the right child; otherwise, move to the left.
4. Repeat for 4 levels to reach a leaf node.
5. The leaf node index (0 to 15) is the hash code.

This creates a 4-bit integer code for each subspace, allowing efficient storage and lookup.

```
Algorithm 1 MADNESSHASH
Input: vector x, split indices j^1,...,j^4, split thresholds v^1,...,v^4
1. i <- 1  // node index within level of tree
2. for t <- 1 to 4 do
3.     v <- v^t_i  // lookup split threshold for node i at level t
4.     b <- x_{j^t} >= v ? 1 : 0  // above split threshold?
5.     i <- 2i - 1 + b  // assign to left or right child
6. end for
7. return i
```

***

## 3. Prototype Learning: Getting Parameters from Training Data

To maximize accuracy, the hash function parameters (split indices and thresholds) and the prototype values must be learned from a calibration dataset (training data $\tilde{A}$).

### Step A: Learning the Hash Function (Tree Construction)

The goal is to group similar vectors into the same bucket so they can be accurately represented by a single prototype. This is done via a **Greedy Tree Construction** algorithm:

1. **Initialization**: Start with all training vectors in one bucket (root).
2. **Split Selection**: For each level of the tree:
    * Evaluate candidate feature indices (dimensions) to split on.
    * For each candidate index, find the optimal split thresholds for every existing bucket that minimize the **Sum of Squared Errors (SSE)** within the resulting child buckets.
    * Select the index that yields the lowest total SSE across all buckets.
3. **Recursion**: Apply the splits to create the buckets for the next level. Repeat until the tree reaches depth 4.

```
Algorithm 2 Adding The Next Level to the Hashing Tree
Input: buckets B^t-1_1,...,B^t-1_{2^{t-1}}, training matrix A~
1. J^ <- heuristic_select_idxs(B^t-1_1,...,B^t-1_{2^{t-1}})  // greedily choose next split index and thresholds
2. l^min, j^min, v^min <- infinity, NaN, NaN
3. for j in J^ do
4.     l <- 0  // initialize loss for this index to 0
5.     v <- []  // empty list of split thresholds
6.     for i <- 1 to 2^{t-1} do
7.         v_i, l_i <- optimal_split_threshold(j, B^t-1_i)
8.         append(v, v_i)  // append threshold for bucket i
9.         l <- l + l_i  // accumulate loss from bucket i
10.    end for
11.    if l < l^min then
12.        l^min <- l, j^min <- j, v^min <- v  // new best split
13.    end if
14. end for
15. // create new buckets using chosen split
16. B <- []
17. for i <- 1 to 2^{t-1} do
18.     B_below, B_above <- apply_split(v^min_i, B^t-1_i)
19.     append(B, B_below)
20.     append(B, B_above)
21. end for
22. return B, l^min, j^min, v^min
```

### Step B: Optimizing the Prototypes

Once the hash function $g(\cdot)$ is fixed, the training vectors $\tilde{A}$ are assigned to buckets. Instead of just averaging the vectors in a bucket (as in K-Means), MADDNESS optimizes the prototypes globally to reconstruct $\tilde{A}$.

1. Construct a matrix $G$ where rows are one-hot encodings of the bucket assignments for $\tilde{A}$.
2. Solve for the optimal prototypes $P$ using **Ridge Regression**:

$$
P = (G^T G + \lambda I)^{-1} G^T \tilde{A}
$$

This ensures the prototypes minimize the reconstruction error $\|\tilde{A} - GP\|^2$, significantly improving approximation quality compared to simple averaging [^1_1].

***

## 4. Application to Language Models (LLMs)

Language models rely heavily on Linear layers (e.g., $W_Q, W_K, W_V$ in Attention, and Feed-Forward networks). These are essentially matrix multiplications $Y = XW$.

### Implementation Steps

#### 1. Offline Preparation (Pre-computation)

* **Calibration**: Collect a set of activation vectors $X_{calib}$ from the LLM using a small calibration dataset.
* **Train MADDNESS**:
    * Treat $X_{calib}$ as $\tilde{A}$.
    * Learn the hash trees (split indices and thresholds) and prototype matrix $P$ for each linear layer.
* **Encode Weights**:
    * The weight matrix $W$ is static. Compute the lookup table $T$ by multiplying the learned prototypes $P$ with the weights $W$: $T = PW$.
    * Quantize $T$ to 8-bit integers to allow fast SIMD shuffling.


#### 2. Online Inference

Replace the `Linear(x)` operation in the LLM forward pass with the MADDNESS function:

* **Input**: Activation vector $x$ (e.g., shape `[Batch, Seq_Len, Hidden_Dim]`).
* **Hashing**:
    * Divide $x$ into subspaces (e.g., chunks of dimensions).
    * Run the learned hash tree (using SIMD instructions) on each chunk to get 4-bit indices.
* **Lookup \& Aggregate**:
    * Use the indices to fetch partial results from the precomputed table $T$.
    * Sum the partial results (using efficient integer averaging instructions) to get the final output vector $y$.
* **Dequantize**: Apply scale factors ($\alpha, \beta$) to map the integer sum back to floating point (BF16/FP16).


### Benefits for LLMs

* **Speed**: Eliminates multiply-add operations, replacing them with fast bitwise logic and table lookups. This can yield >10x speedups on CPUs.[^1_1]
* **Compression**: The weight matrix $W$ is replaced by the lookup table $T$, which can be significantly smaller if the number of prototypes is low.

<div align="center">⁂</div>

[^1_1]: ArXiv2021-Multiplying-Matrices-Without-Multiplying-1.pdf

