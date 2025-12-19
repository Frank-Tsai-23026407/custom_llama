# TinyLlama & Llama 量化工具集 (Quantization Toolkit)

這個目錄包含了模型量化（Quantization）的核心算法與實驗工具，主要支援 **1D/2D Block Floating Point (BFP)** 以及 **Activation-aware Weight Quantization (AWQ)**。

## 📁 目錄結構

- **根目錄 (Root)**: 核心算法庫與主要執行腳本。
- **`docs/`**: 詳細的技術文檔與各腳本說明。
- **`scripts/`**: 用於批次執行實驗的 Shell 腳本。
- **`legacy/`**: 舊版或實驗性質的腳本（已由綜合腳本取代）。
- **`utils/`**: 模型轉換與視覺化工具。

---

## 🚀 核心工具 (Core Tools)

### 1. 核心算法庫
- **`block_quantization.py`**: 1D Block Floating Point 量化實作。
- **`block_quantization_2d.py`**: 支援 2D 區塊的 BFP 與 AWQ (Fix/Mix Precision) 量化實作。
- **`runtime_quantizer.py`**: 執行時期量化封裝，支援在推理時動態應用量化。

### 2. 推薦執行腳本
- **`quantize_tinyllama_comprehensive_2d.py`**: **[推薦]** 綜合量化腳本，支援 BFP, AWQ-Fix, AWQ-Mix 以及所有 2D Block 形狀的掃描。
- **`evaluate_with_runtime_quantization.py`**: 使用 Runtime Quantization 進行評估的範例（不佔用磁碟空間）。
- **`quantize_tinyllama_2d_sweep.py`**: 針對 2D BFP 進行區塊形狀掃描的專用腳本。

---

## ⚡ 推薦工作流程：Runtime Quantization

為了節省硬碟空間並加速實驗，推薦使用 **Runtime Quantization**。只需保存一個原始 BF16 模型，即可在推理時實時測試各種量化配置。

```bash
# 使用 16x16 區塊與 4-bit Mantissa 進行 BFP 量化評估
python evaluate_with_runtime_quantization.py \
    --method bfp \
    --block-height 16 \
    --block-width 16 \
    --mantissa-bits 4
```

詳細說明請參閱 `docs/RUNTIME_QUANTIZATION.md`。

---

## 📊 量化方法概覽

| 方法 | 準確度 | 硬體友好 | 校準數據 (Calibration) | 說明 |
|------|--------|----------|------------------------|------|
| **BFP** | ⭐⭐ | ✅✅✅ | ❌ 不需要 | 標準區塊浮點量化 |
| **AWQ-Fix** | ⭐⭐⭐ | ✅✅ | ✅ 需要 | 使用激活資訊縮放，保持純 BFP 格式 |
| **AWQ-Mix** | ⭐⭐⭐⭐ | ✅ | ✅ 需要 | 重要權重保留 FP32，其餘變 BFP |

---

## 📖 詳細文檔 (Documentation)

更多詳細資訊請參閱 `docs/` 目錄：
- [綜合腳本使用說明](docs/README_COMPREHENSIVE_2D.md)
- [Runtime Quantization 詳細介紹](docs/RUNTIME_QUANTIZATION.md)
- [2D 量化技術細節](docs/README_2D_QUANTIZATION.md)
- [存儲格式說明](docs/STORAGE_FORMAT.md)

---

## 🛠️ 批次實驗腳本

常用實驗腳本位於 `scripts/` 目錄：
- `scripts/run_bfp_only.sh`: 執行所有 BFP 相關實驗。
- `scripts/run_awq_fix_only.sh`: 執行 AWQ Fix-Precision 實驗。
- `scripts/run_quantize_tinyllama_comprehensive.sh`: 執行完整的綜合量化試驗。
