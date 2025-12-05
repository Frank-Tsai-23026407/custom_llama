# Phase 1 Results: Baseline vs BFP Bit5

## Overview
We have organized the results for the baseline (bf16) and Experiment A (BFP bit5 dense) on the HellaSwag benchmark.

## Results Table

| Experiment | Configuration | Accuracy (acc) | Accuracy (acc_norm) | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | **bf16 dense** | - | **0.6180** | Reference performance |
| **Exp A** | **BFP bit5 dense** (1x128) | 0.4572 | **0.6156** | Best BFP configuration |
| Exp A | BFP bit5 dense (2x64) | 0.4551 | 0.6138 | |
| Exp A | BFP bit5 dense (4x32) | 0.4552 | 0.6120 | |
| **Exp B** | **BFP bit5 + Naive 50% Sparsity** | 0.2731 | **0.2976** | Significant degradation (expected) |
| **Exp C** | **BFP bit5 + Wanda 50% Sparsity** | 0.3620 | **0.4876** | Significant recovery (+19%) vs Naive |
| **Exp D** | **Mixed Precision + Wanda (Selective)** | 0.3943 | **0.5536** | **Best Result**. +6.6% vs Exp C, only ~6% drop from dense baseline. |
| **Debug D** | **Mixed Precision (No Sparsity)** | - | **0.5600** | (100 samples) Validates Mixed Precision implementation |
| **Debug D** | **Mixed Precision + Wanda (Selective)** | - | **0.4600** | (100 samples) Validates Selective Pruning logic |

## Observation
- The **BFP bit5 dense** model (specifically with 1x128 block size) achieves **0.6156** accuracy (acc_norm), which is extremely close to the **bf16 baseline** of **0.6180**.
- **Experiment B (Naive Sparsity)** resulted in a massive accuracy drop to **0.2976**. This confirms the literature's warning: simply zeroing out 50% of weights in a low-bit quantized model destroys performance without advanced techniques.
- **Experiment C (Wanda Pruning)** improved accuracy to **0.4876**. This validates that considering activation magnitude preserves more important weights. However, there is still a gap (~13%) compared to the dense baseline.
- **Experiment D (Mixed Precision + Wanda)** achieved **0.5536** accuracy. By keeping sensitive layers (first, last, and intermediate attention/MLP layers) in 8-bit precision and skipping pruning for them, we recovered significant performance. The gap to the dense baseline is now only **~6.2%**.
  - **Effective Sparsity**: In this experiment, we applied 50% sparsity only to the "robust" layers (91 out of 155 layers). The "sensitive" layers (64 layers) were kept dense (0% sparsity). This results in an effective global sparsity of approximately **29%**.
  - This demonstrates that a strategic combination of **Mixed Precision (8-bit/5-bit)** and **Selective Pruning (0%/50%)** is highly effective for compressing TinyLlama while maintaining accuracy.

## Conclusion
The project successfully implemented and evaluated three pruning strategies on a pre-quantized BFP model:
1.  **Naive Magnitude Pruning (Exp B)**: Failed (0.2976 accuracy), proving that simple pruning is destructive for low-bit models.
2.  **Activation-Aware (Wanda) Pruning (Exp C)**: Significant improvement (0.4876 accuracy), validating the importance of activation data.
3.  **Mixed Precision + Selective Pruning (Exp D)**: Best performance (0.5536 accuracy), coming within 6% of the dense baseline.

**Recommendation**: For optimal trade-off between compression and accuracy, **Experiment D**'s approach (Mixed Precision + Selective Wanda Pruning) is the clear winner. Future work could explore:
- Pruning the high-precision layers with a lower sparsity ratio (e.g., 20%) instead of 0%.
- Fine-tuning the mixed precision policy to include `lm_head` in high precision (currently pruned).
- Retraining (fine-tuning) the pruned model to recover the remaining 6% accuracy gap.
