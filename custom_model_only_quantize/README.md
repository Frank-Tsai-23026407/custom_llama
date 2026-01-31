# TinyLlama & Llama 2D Quantization Toolkit

這個目錄包含了 TinyLlama 與 Llama 模型的 2D Block Quantization 核心算法與實驗工具。支援 Block Floating Point (BFP) 與 Activation-Aware Weight Quantization (AWQ) 等多種量化策略。

## 📁 目錄結構

- **根目錄**: 核心算法共享庫與綜合執行腳本。
- **`block-floating-point/`**: BFP 相關腳本。
- **`fix-precision-awq/`**: AWQ Fix-Precision (全量化) 相關腳本。
- **`mix-precision-awq/`**: AWQ Mix-Precision (混合精度) 相關腳本。
- **`utils/`**: 工具函式。

---

## 🚀 快速開始：Runtime Quantization (推薦)

**Runtime Quantization** 允許在推理時動態應用量化，無需保存巨大的量化模型文件，節省 98% 以上的磁碟空間。

### 優勢
- **極省空間**: 只需保存 1 個原始 BF16 模型 (~2.2GB)，而非 48 個量化變體 (~100GB+)。
- **快速實驗**: 幾秒鐘內即可切換並測試不同的 block size 與 bit-width。

### 使用方法

```bash
# 1. 使用 Runtime Quantization 進行評估
python evaluate_with_runtime_quantization.py \
    --model tinyllama \
    --method bfp \
    --block-height 16 \
    --block-width 16 \
    --mantissa-bits 4
```

或者在 Python 中使用：

```python
from transformers import AutoModelForCausalLM
from custom_model_only_quantize.runtime_quantize import apply_runtime_quantization

# 載入原始模型
model = AutoModelForCausalLM.from_pretrained("model/tinyllama/TinyLlama_1.1v", torch_dtype=torch.bfloat16)

# 實時應用量化
apply_runtime_quantization(
    model,
    method="bfp",
    block_height=16,
    block_width=16,
    mantissa_bits=4
)

# 正常使用
outputs = model.generate(...)
```

---

## 🛠️ 綜合量化腳本 (Generating Models)

如果你需要生成並保存量化後的模型文件（例如為了進一步分析或部署），請使用綜合量化腳本。

### `quantize_tinyllama_comprehensive_2d.py`

這是主要入口腳本，支援生成所有配置。

**基本用法**:
```bash
# 執行所有方法與配置 (48 models)
python quantize_tinyllama_comprehensive_2d.py --method all

# 僅執行 BFP (最快)
python quantize_tinyllama_comprehensive_2d.py --method bfp

# 僅執行 AWQ Fix-Precision
python quantize_tinyllama_comprehensive_2d.py --method awq-fix
```

**常用參數**:
- `--model`: 模型路徑或預設名稱 (default: `tinyllama`)
- `--method`: `bfp`, `awq-fix`, `awq-mix`, 或 `all`
- `--output-dir`: 輸出目錄
- `--save-dtype`: `float32`, `float16`, `bfloat16` (推薦使用 `bfloat16` 或 `float16` 以減少文件大小)
- `--skip-generation-test`: 跳過生成測試以加速
- `--dry-run`: 測試運行，不保存模型

### 便捷 Shell 腳本
- `run_quantize_tinyllama_comprehensive.sh`: 執行完整實驗
- `quantize_llama_3_2_1b.sh`: 針對 Llama 3.2 1B 的實驗

---

## 📊 量化方法比較

我們實現了三種主要的量化策略：

| 方法 | 速度 | 準確度 | 權重格式 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| **BFP** | ⚡ 快 | ⭐ Good | 全 BFP | 標準 Block Floating Point，無激活感知。適合快速基準測試。 |
| **AWQ (Fix)** | ⚡ 中 | ⭐⭐ Better | 全 BFP (Scaled) | **Activation-Aware**。根據激活值**縮放**權重，保護重要權重。所有權重仍為 BFP 格式，適合硬體部署。 |
| **AWQ (Mix)** | 🐌 慢 | ⭐⭐⭐ Best | FP32 + BFP | **Mixed Precision**。每個 Block 中 Top-K 最重要的權重保持 **FP32**，其餘 BFP。準確度最高，但格式不統一。 |

### 2D Block Quantization 技術細節
我們使用 **2D Blocks** (例如 32x16) 而非傳統的 Row/Column 量化。
- **Block Size**: 靈活定義 `Block Height` x `Block Width`。
- **Shared Exponent**: 每個 Block 共享一個指數。
- **Mantissa**: 只有 Mantissa 被量化 (例如 4-bit)。

---

## ⚠️ 重要說明：存儲格式與壓縮

> [!IMPORTANT]
> **目前的量化模型在磁碟上並不會變小！**

由於 PyTorch 的 `save_pretrained` 機制限制，量化後的權重仍然以標準的 `float32` / `bfloat16` Tensor 格式存儲。
- **BFP 4-bit 量化**：數值是 4-bit 精度的，但容器是 32-bit 或 16-bit 的。
- **文件大小**：與原始模型相同（除非使用 `--save-dtype float16` 減半）。

**要獲得真正的壓縮（如 4-bit 文件大小），需要實現自定義的序列化格式與 CUDA Kernel，這超出了本專案目前的範圍。**

---

## 📂 子模組說明

### Mix-Precision AWQ (`mix-precision-awq/`)
- **核心機制**: 計算顯著性 `Saliency = mean(|X|) * |W|`。
- **Block-wise Top-K**: 在每個 2D Block 內部，保留前 K 個最敏感的權重為 FP16/BF16，其餘量化。

### Fix-Precision AWQ (`fix-precision-awq/`)
- 使用縮放因子 (Scaling Factors) 來保護重要權重，保持統一的 BFP 格式。

### Block Floating Point (`block-floating-point/`)
- 基礎的 2D BFP 實作與 Sweep 腳本。

---

*最後更新: 2025-12-22*
