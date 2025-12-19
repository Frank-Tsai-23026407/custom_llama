# TinyLlama 自定義後端架構使用指南

本專案實作了一套高度模組化且具備研究靈活性的 Llama 推理框架。透過 **混合型物件導向設計 (Hybrid OOP Architecture)**，我們將數學公式、運算後端與模型結構完全解耦，使研究人員能輕鬆地抽換 GEMM 實作或測試新型量化演算法。

## 🎯 核心設計亮點
- **後端策略模式 (Backend Strategy Pattern)**：透過 `LlamaBackend` 物件統管所有張量運算，支援一鍵切換運算引擎。
- **三層架構分離**：
    1.  **功能模組 (Standalone Math)**：純函數實作，易於單元測試。
    2.  **底層運算 (Backend Class)**：物件化運算介面，支援相依注入 (Dependency Injection)。
    3.  **架構組件 (Structural Modules)**：繼承 `nn.Module`，管理 KV Cache 等模型狀態。
- **動態精度策略 (Precision Policy)**：細粒度控制不同算子（Matmul, Softmax, RoPE）的計算精度。

---

## 📂 專案目錄結構
```bash
llama_backend/
├── llama_custom.py       # 主介面：CustomLlamaModel
├── custom/
│   ├── llama_backend.py  # Section 1 & 2: 數學函數與後端類別定義
│   ├── precision_policy.py # 精度策略 management
│   └── README.md         # 後端架構詳細技術文件
├── utils.py              # 輔助工具 (StopCriteria, Formatting)
└── clone/                # 官方對照用 (HF-parity Clones)
```

---

## ⚡ 快速開始

### 1. 基本推理路徑
現在主模型類別為 `CustomLlamaModel`，建議透過根目錄進行導入。

```python
import torch
from llama_backend.llama_custom import CustomLlamaModel
from transformers import AutoTokenizer

# 1. 載入模型 (預設使用 custom 物件導向後端)
model = CustomLlamaModel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda",
    dtype=torch.bfloat16,
    precision_policy="bf16" # 全端採用 bfloat16 加速
)

# 2. 準備輸入
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
inputs = tokenizer("你好，請介紹一下自己", return_tensors="pt").to("cuda")

# 3. 執行推理 (支援 KV Cache)
logits = model.single_step(inputs)
print(f"Logits shape: {logits.shape}")

# 4. 生成文本
output = model.generate("請寫一首詩：", max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

---

## 🛠️ 進階：自定義後端 (The Strategy Pattern)

如果你正在開發新的硬體加速器或量化矩陣乘法（例如量化後的 GEMM），你只需要擴展 `LlamaBackend`。

### 擴展步驟
1. 繼承 `LlamaBackend`。
2. 覆寫 `gemm`, `rmsnorm` 或 `cat` 等方法。
3. 在初始化模型時注入你的後端實例。

```python
from llama_backend.custom.llama_backend import LlamaBackend, DEFAULT_BACKEND
from llama_backend.llama_custom import CustomLlamaModel

# 實作一個特殊的 FP8 實驗後端
class MyResearchBackend(LlamaBackend):
    def gemm(self, input, weight, policy=None):
        print("Using Research FP8 GEMM Engine...")
        # 在此插入你的自定義 CUDA Kernel 或模擬邏輯
        return super().gemm(input.half(), weight.half(), policy)

# 注入後端
custom_backend = MyResearchBackend()
model = CustomLlamaModel(
    model_name="TinyLlama/TinyLlama_v1.1",
    backend="custom"  # 確保使用自定義路徑
)

# 替換所有 Block 的後端 (DI 注入)
for block in model.attention_blocks:
    block.backend = custom_backend
    block.attention.backend = custom_backend
    block.mlp.backend = custom_backend
```

---

## ⚖️ 精度策略 (Precision Policy)

我們提供 `PrecisionPolicy` 物件來管理運算精度，避免手動對每個 Tensor 進行 `.to()` 操作。

| 預設策略 | 適用場景 | 數值穩定性 |
| :--- | :--- | :--- |
| `default` | 保持模型權重原樣 (通常是 FP32) | 最高 |
| `match_hf` | 模擬 Hugging Face 的混合精度 (BF16 主計算) | 高 |
| `bf16` | 全端 BF16，適合效能測試 | 中 |

**自定義策略範例：**
```python
from llama_backend.custom.precision_policy import PrecisionPolicy

my_policy = PrecisionPolicy(
    attn_matmul_dtype=torch.float32,   # Attention 矩陣乘法用 FP32 保持精準
    attn_softmax_dtype=torch.float32,  # Softmax 使用 FP32 避免溢位
    ffn_matmul_dtype=torch.bfloat16,   # FFN 則用 BF16 加速
    stable_softmax=True                # 開啟穩定性補償 (LogSumExp Trick)
)

model = CustomLlamaModel(precision_policy=my_policy)
```

---

## 🧪 效能與精度比較

透過 `backend` 參數，你可以輕鬆在不同實作之間切換以進行基準測試：

- **`backend="huggingface"`**：原生官方實作。作為 **Ground Truth** 基準。
- **`backend="clone"`**：純函數式克隆。適合 **逐行 Debug** 數值細節（無 KV Cache）。
- **`backend="custom"`**：物件導向架構。適合 **推理優化與功能開發**（支援 KV Cache）。

**實測對齊結果 (Mean Absolute Error)：**
- `Custom vs HuggingFace`: **~0.038** (在 BF16 下與官方法精度極度接近)
- `Custom vs Clone`: **~0.011** (架構間邏輯高度統一)

---

## 💎 量化功能

### Block Floating Point (BFP)
透過在初始化時設定參數，自動應用區塊浮點數：
```python
model = CustomLlamaModel(
    apply_bfp=True,
    bfp_block_size=16,
    bfp_mantissa_bits=4 # 4-bit 權重量化
)
```

---

## 🚀 最佳實踐建議
1.  **開發階段**：使用 `backend="clone"` 與 `return_latents=True` 來比較中間層的輸出數值是否偏離官方模型。
2.  **研究階段**：建立自己的 `CustomBackend` 子類別，實驗不同的矩陣乘法分配與記憶體重組策略。
3.  **推理階段**：使用 `backend="custom"` 並搭配 `PrecisionPolicy("bf16")` 獲取最佳的 KV Cache 加速與處理效能。

---
最後更新：2025-12-19
