# TinyLlama Custom Backend Architecture (llama_backend)

> [!TIP]
> **[Actually, if you just want to simply replace a Layer, use the Golden Method ⮕](#golden_method)**

---

## 📑 Table of Contents
1. [🌟 Core Concept](#core-concept)
2. [🏗️ System Architecture](#system-architecture)
3. [💻 Code Deep Dive](#code-deep-dive)
4. [🛠️ Advanced Customization: Strategy Pattern](#advanced-customization)
5. [⚖️ Precision Policy Management](#precision-policy)
6. [💡 Golden Method (Standard Replacement)](#golden_method)
7. [📎 Appendix: Observing Models with Hooks](#appendix_hooks)

---

<h2 id="core-concept">🌟 Core Concept</h2>

The primary goal of this project is to provide a **completely decoupled** inference framework. In traditional libraries like `transformers`, mathematical operations, tensor manipulations, and layer structures (such as Attention Blocks) are often tightly coupled. When researchers want to test a new "quantized matrix multiplication" or a "custom CUDA Kernel," they usually need to modify extensive amounts of source code.

`llama_backend` addresses this by using the **Strategy Pattern** to separate these responsibilities:
- **Structural Modules**: Manage data flow (e.g., residual connections, Layer Norm ordering).
- **Backend Class**: Handles the actual mathematical operations (e.g., GEMM, RoPE).
- **Precision Policy**: Independently manages the `dtype` for computation.

---

<h2 id="system-architecture">🏗️ System Architecture</h2>

We employ a three-layer decoupled design:

| Layer | Component Name | Responsibility |
| :--- | :--- | :--- |
| **Top-Level** | `CustomLlamaModel` | Parameter loading, Tokenization, KV Cache management, and generation logic. |
| **Mid-Level** | `LlamaTransformerBlock` | Connects Attention and MLP, defining the residual path. |
| **Low-Level** | `LlamaBackend` | Implements concrete operations like `gemm`, `rmsnorm`, `softmax`, etc. |

---

<h2 id="code-deep-dive">💻 Code Deep Dive</h2>

### 1. Backend Abstraction (The Backend Strategy)
All operations are encapsulated within the `LlamaBackend` class. This means you can "instantly" change the computational behavior of the entire model by subclassing it.

```python
# llama_backend/custom/llama_backend.py

class LlamaBackend:
    def gemm(self, input, weight, transpose_b=True, policy=None):
        """Generic matrix multiplication, can be overridden for custom quantization ops"""
        if transpose_b:
            weight = weight.transpose(-2, -1)
        return torch.matmul(input, weight)

    def rmsnorm(self, x, weight, eps=1e-5):
        """Standard RMSNorm, numerical stability optimizations can be added here"""
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return weight * (x * torch.rsqrt(variance + eps)).to(x.dtype)
```

### 2. Decoupling Structure from Computation
In `LlamaAttention`, we don't call `torch.matmul` directly; instead, we invoke `self.backend.gemm`:

```python
# llama_backend/custom/llama_backend.py (LlamaAttention.forward)

def forward(self, x, params, precision_policy=None, use_cache=False):
    # ... preprocessing ...
    # Call the backend instead of native torch functions
    q = self.backend.gemm(x, params['wq'], policy=policy)
    k = self.backend.gemm(x, params['wk'], policy=policy)
    
    # Core Attention update
    attn_scores = self.backend.gemm(q, k_final, transpose_b=True) / math.sqrt(head_dim)
    # ...
```

---

<h2 id="advanced-customization">🛠️ Advanced Customization: Strategy Pattern</h2>

If you are researching new hardware or quantization algorithms—for example, adding noise to all matrix multiplications or replacing them with a specific FP8 implementation—you simply need to:

```python
class MyResearchBackend(LlamaBackend):
    def gemm(self, input, weight, transpose_b=True, policy=None):
        print("Executing Research GEMM...")
        # Call your custom C++/CUDA Binding
        return my_custom_cuda_op(input, weight)

# Inject the backend
custom_model = CustomLlamaModel(...)
for block in custom_model.attention_blocks:
    block.backend = MyResearchBackend()
```

---

<h2 id="precision-policy">⚖️ Precision Policy Management</h2>

We provide `PrecisionPolicy` for fine-grained control over the precision of different operations. For instance, you can use `bf16` for MLP computations while keeping the Attention Softmax in `fp32` for stability.

```python
# llama_backend/custom/precision_policy.py

policy = PrecisionPolicy(
    attn_matmul_dtype=torch.float32,   # Maintain precision
    ffn_matmul_dtype=torch.bfloat16,   # Speed optimization
    stable_softmax=True
)
```

---

<h2 id="golden_method">💡 Golden Method: The Standard Way to Replace Layers</h2>

If you feel rewriting the entire backend is too complex and only want to replace specific layers in an existing Hugging Face model, this is the **"Golden Method"**.

### Core Logic: Dynamic Assignment
PyTorch's `nn.Module` is highly dynamic; you can directly overwrite attributes with new instances.

```python
import torch.nn as nn
from transformers import AutoModelForCausalLM

# 1. Define your new layer
class MySpecialLinear(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.weight = original_layer.weight
        self.bias = original_layer.bias
    def forward(self, x):
        return x @ self.weight.T + (self.bias if self.bias is not None else 0)

# 2. Load the model and replace
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1")

# Directly replace q_proj in Layer 0
target_block = model.model.layers[0].self_attn
target_block.q_proj = MySpecialLinear(target_block.q_proj)

# 3. Verification
print(model.model.layers[0].self_attn.q_proj)
```

---

<h2 id="appendix_hooks">📎 Appendix: Observing Models with Hooks</h2>

Sometimes you don't want to "replace" a layer, but rather "observe" or "modify" inputs/outputs (e.g., extracting intermediate activations or injecting noise). **Hooks** are the most powerful tool for this.

### Using `register_forward_hook`
This allows you to intervene in the computational flow without modifying the model structure.

```python
def my_hook_fn(module, input, output):
    # input[0] is the Tensor passed into the layer
    # output is the result calculated by the layer
    print(f"Layer Name: {module}")
    print(f"Input Shape: {input[0].shape}")
    # You can even subtly modify the output before it passes to subsequent layers
    return output * 1.05 

# Attach a hook to all MLP Gate Projections
for name, module in model.named_modules():
    if "mlp.gate_proj" in name:
        module.register_forward_hook(my_hook_fn)

# During inference, the hook triggers automatically
model(torch.randint(0, 1000, (1, 10)))
```

---
*Last Updated: 2025-12-22*
