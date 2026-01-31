# TinyLlama 自定義後端架構 (llama_backend)

> [!TIP]
> **[其實如果你只是要簡單替換 Layer，請使用 Golden Method ⮕](#golden_method)**

---

## 📑 目錄 (Outline)
1. [🌟 核心理念 (Concept)](#核心理念)
2. [🏗️ 系統架構 (Architecture)](#系統架構)
3. [💻 程式碼深度解析 (Code Deep Dive)](#程式碼深度解析)
4. [🛠️ 進階自定義：後端策略 (Strategy Pattern)](#進階自定義)
5. [⚖️ 精度策略管理 (Precision Policy)](#精度策略管理)
6. [💡 Golden Method (標準替換法)](#golden_method)
7. [📎 附錄：使用 Hook 觀察模型 (Appendix: Hooks)](#appendix_hooks)

---

<h2 id="核心理念">🌟 核心理念 (Concept)</h2>

本專案的核心目標是提供一個**完全解耦 (Decoupled)** 的推理框架。在傳統的 `transformers` 庫中，數學運算、張量操作與層級結構（如 Attention Block）是混在一起的。當研究者想測試一個新的「量化矩陣乘法」或「自定義 CUDA Kernel」時，往往需要修改大量的原始碼。

`llama_backend` 透過 **策略模式 (Strategy Pattern)** 將這些職責拆分：
- **層級結構 (Structure)**：只負責資料流向（如 Residual Connection, Layer Norm 順序）。
- **底層運算 (Backend)**：負責執行所有數學運算（如 GEMM, RoPE）。
- **精度控制 (Precision)**：獨立管理計算時的 `dtype`。

---

<h2 id="系統架構">🏗️ 系統架構 (Architecture)</h2>

我們採用了三層架構分離設計：

| 層級 | 組件名稱 | 職責 |
| :--- | :--- | :--- |
| **Top-Level** | `CustomLlamaModel` | 參數加載、Tokenization、KV Cache 管理、生成邏輯。 |
| **Mid-Level** | `LlamaTransformerBlock` | 串接 Attention 與 MLP，定義 Residual 路徑。 |
| **Low-Level** | `LlamaBackend` | 實作具體的 `gemm`, `rmsnorm`, `softmax` 等運算。 |

---

<h2 id="程式碼深度解析">💻 程式碼深度解析 (Code Deep Dive)</h2>

### 1. 後端抽象化 (The Backend Strategy)
所有的運算都被封裝在 `LlamaBackend` 類別中。這意味著你可以透過繼承它來「瞬間」改變整個模型的運算行為。

```python
# llama_backend/custom/llama_backend.py

class LlamaBackend:
    def gemm(self, input, weight, transpose_b=True, policy=None):
        """通用矩陣乘法，可以被覆寫為自定義的量化運算"""
        if transpose_b:
            weight = weight.transpose(-2, -1)
        return torch.matmul(input, weight)

    def rmsnorm(self, x, weight, eps=1e-5):
        """標準 RMSNorm，可在此加入數值穩定性優化"""
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return weight * (x * torch.rsqrt(variance + eps)).to(x.dtype)
```

### 2. 結構與運算的解耦
在 `LlamaAttention` 中，我們不直接調用 `torch.matmul`，而是調用 `self.backend.gemm`：

```python
# llama_backend/custom/llama_backend.py (LlamaAttention.forward)

def forward(self, x, params, precision_policy=None, use_cache=False):
    # ... 省略前處理 ...
    # 這裡調用 backend 而不是 torch 原生函數
    q = self.backend.gemm(x, params['wq'], policy=policy)
    k = self.backend.gemm(x, params['wk'], policy=policy)
    
    # 核心 Attention 更新
    attn_scores = self.backend.gemm(q, k_final, transpose_b=True) / math.sqrt(head_dim)
    # ...
```

---

<h2 id="進階自定義">🛠️ 進階自定義：後端策略 (Strategy Pattern)</h2>

如果你正在研究新的硬體或量化演算法，例如要將所有的矩陣乘法加上雜訊 (Simulation) 或替換成特殊的 FP8 實作，你只需要：

```python
class MyResearchBackend(LlamaBackend):
    def gemm(self, input, weight, transpose_b=True, policy=None):
        print("執行研究用 GEMM...")
        # 調用你的自定義 C++/CUDA Binding
        return my_custom_cuda_op(input, weight)

# 注入後端
custom_model = CustomLlamaModel(...)
for block in custom_model.attention_blocks:
    block.backend = MyResearchBackend()
```

---

<h2 id="精度策略管理">⚖️ 精度策略管理 (Precision Policy)</h2>

我們提供 `PrecisionPolicy` 讓你能細粒度控制不同操作的精度。例如，你可以讓 MLP 運算用 `bf16`，但讓 Attention 的 Softmax 為了穩定性保持 `fp32`。

```python
# llama_backend/custom/precision_policy.py

policy = PrecisionPolicy(
    attn_matmul_dtype=torch.float32,   # 保持精度
    ffn_matmul_dtype=torch.bfloat16,   # 追求速度
    stable_softmax=True
)
```

---

<h2 id="golden_method">💡 Golden Method：在 PyTorch 中替換 Layer 的標準做法</h2>

如果你覺得重寫整個後端太過複雜，只想針對既有的 Hugging Face 模型替換某一兩層，這就是所謂的 **"Golden Method"**。

### 核心邏輯：直接賦值 (Dynamic Replacement)
PyTorch 的 `nn.Module` 是高度動態的，你可以直接用新的實例覆蓋屬性。

```python
import torch.nn as nn
from transformers import AutoModelForCausalLM

# 1. 定義你的新層
class MySpecialLinear(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.weight = original_layer.weight
        self.bias = original_layer.bias
    def forward(self, x):
        return x @ self.weight.T + (self.bias if self.bias is not None else 0)

# 2. 載入模型並替換
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1")

# 直接替換第 0 層的 q_proj
target_block = model.model.layers[0].self_attn
target_block.q_proj = MySpecialLinear(target_block.q_proj)

# 3. 驗證
print(model.model.layers[0].self_attn.q_proj)
```

---

<h2 id="appendix_hooks">📎 附錄：使用 Hook 觀察模型 (Appendix: Hooks)</h2>

有時候你不想「替換」層，只是想「觀察」或「修改」輸入輸出（例如：提取 Intermediate Activations 或注入干擾）。這時候 **Hooks** 是最強大的工具。

### 使用 `register_forward_hook`
這可以讓你在不修改模型結構的情況下，介入運算流。

```python
def my_hook_fn(module, input, output):
    # input[0] 是傳入該層的 Tensor
    # output 是該層算完的結果
    print(f"層名稱: {module}")
    print(f"輸入 Shape: {input[0].shape}")
    # 你甚至可以偷偷修改 output 再傳回後續層
    return output * 1.05 

# 為所有 MLP 的 Gate Proj 掛上鉤子
for name, module in model.named_modules():
    if "mlp.gate_proj" in name:
        module.register_forward_hook(my_hook_fn)

# 執行推理時，hook 會自動觸發
model(torch.randint(0, 1000, (1, 10)))
```

---
*最後更新：2025-12-22*
