# Llama Custom Backend Design

這個目錄開發了一套靈活且可擴展的 Llama 模型實作架構，特別針對研究需求設計，允許開發者輕鬆抽換底層運算邏輯。

## 架構設計：混合模式 (Hybrid Mode)

本架構採用三層設計，在層與層之間建立清晰的邊界，確保計算邏輯、運算後端與模型結構三者的解耦：

### 1. 第一層：功能模組 (Functional Blocks) -> **頂層函數 (Section 1)**
*   **組件**：`apply_rope`, `rmsnorm`, `input_embedding`, `lm_head`
*   **職責**：執行無狀態的原子數學運算。
*   **優勢**：純函數實作易於單元測試與公式驗證，不涉及模型狀態或後端選擇。

### 2. 第二層：運算引擎 (Math Engine/Strategy) -> **後端物件 (Section 2)**
*   **組件**：`LlamaBackend` 類別及其子類別
*   **職責**：封裝底層硬體相關或優化過的運算，如 `gemm` (矩陣乘法)、`softmax`、`cat` (記憶體合併)。
*   **優勢**：**策略模式 (Strategy Pattern)** 的核心。當你想測試新的量化演算法（例如特殊的 Quantized GEMM）或自定義 CUDA Kernel 時，**只需繼承此類別並覆寫方法**。

### 3. 第三層：架構組件 (Structural Modules) -> **有狀態類別 (Section 3)**
*   **組件**：`LlamaAttention` (MQA), `LlamaMLP` (FFN), `LlamaTransformerBlock`
*   **職責**：管理模型的結構、參數量與暫存狀態（如 **KV Cache**）。
*   **優勢**：採用 **相依注入 (Dependency Injection)**，在 `__init__` 時接受 `backend`。模型組件只負責「呼叫順序」，而不關心「如何運算」。

---

## 範例：如何擴展後端 (Extend LlamaBackend)

如果你有一篇論文提到的特殊優化 GEMM 實作，可以按照以下步驟套用：

```python
from llama_backend.custom.llama_backend import LlamaBackend, LlamaTransformerBlock

# 1. 繼承並定義你的優化後端
class MyPaperBackend(LlamaBackend):
    def gemm(self, input, weight, policy=None):
        print("Using Specialized Matmul Kernel...")
        # 插入你的優化邏輯，例如 FP8 或量化運算
        return torch.matmul(input, weight)

# 2. 初始化後端實例
my_backend = MyPaperBackend()

# 3. 將後端注入模型組件
# 現在這個 Block 所有的投影與計算都會自動使用 MyPaperBackend
block = LlamaTransformerBlock(num_heads=32, num_kv_heads=4, backend=my_backend)

# 4. 正常執行 forward
output = block(x, params, use_cache=True)
```

---

## 精度策略 (Precision Policy)

透過 `precision_policy.py`，我們可以統一管理模型內部的數據類型。例如：
- `default`: 保持權重的原生精度。
- `match_hf`: 模擬 Hugging Face 的混合精度行為。
- `bf16`: 強制全路採用 bfloat16 加速。

## 檔案快速導覽
- `llama_backend.py`: 包含 Section 1~3 的核心實作。
- `precision_policy.py`: 管理不同運算階段的 `torch.dtype` 轉換策略。
- `README.md`: 本說明文件。

---
*最後更新：2025-12-19*
