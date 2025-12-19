# Quantization Concepts

This document outlines the key differences between two quantization approaches: Mix-Precision AWQ and Fix-Precision AWQ.

## Mix-Precision AWQ

The core concept of Mix-Precision AWQ is to preserve the precision of the most important weights while compressing the rest of the model.

- **Important Weights**: Identified based on activation magnitudes. These weights are kept in their original full floating-point format.
- **Other Weights**: The remaining, less critical weights are quantized into a lower-precision format, such as BFP (Block Floating Point).

This creates a mixed-precision model where critical weights are not degraded by quantization, aiming to retain higher model accuracy.

## Fix-Precision AWQ

The core concept of Fix-Precision AWQ is to scale all weights based on their importance before quantizing them uniformly.

- **Scaling**: A scaling factor `s` is calculated based on activation magnitudes. All weights are then multiplied by this scaling factor. This adjusts the distribution of weight values to make them more amenable to quantization.
- **Quantization**: All scaled weights, regardless of their original importance, are quantized into a lower-precision format like BFP.
- **De-scaling**: During inference, the quantized values are de-quantized and then divided by the scaling factor `s` to be restored to their approximate original range.

In this approach, all weights are quantized, but the scaling process helps to reduce the overall quantization error by preserving the relative importance of the weights.
