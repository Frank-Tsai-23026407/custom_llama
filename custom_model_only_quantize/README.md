# TinyLlama & Llama 量化工具集 (Quantization Toolkit)

這個目錄包含了模型量化 (Quantization) 的核心算法與實驗工具，已根據方法分類整理到 `custom_field` 的各個子目錄中。

## 📁 目錄結構 (Directory Structure)

- **根目錄 (Root)**: 核心算法共享庫與綜合執行腳本。
- **`block-floating-point/`**: 包含 BFP (Block Floating Point) 相關的特定腳本與實驗。
- **`fix-precision-awq/`**: 包含 AWQ Fix-Precision (全量化) 相關腳本。
- **`mix-precision-awq/`**: 包含 AWQ Mix-Precision (混合精度) 相關腳本。
- **`docs/`**: 詳細的技術文檔與各腳本說明。
- **`utils/`**: 模型轉換與工具函式。
- **`legacy/`**: 舊版或實驗性質的腳本。

---

## 🚀 核心工具 (Core Tools)

### 1. 核心算法庫 (Shared Core)
- **`block_quantization.py`**: 1D Block Floating Point 量化實作。
- **`block_quantization_2d.py`**: 支援 2D 區塊的 BFP 與 AWQ (Fix/Mix Precision) 量化核心實作。
- **`runtime_quantizer.py`**: 執行時期量化封裝，支援在推理時動態應用量化。

### 2. 綜合執行腳本 (Comprehensive Scripts)
- **`quantize_tinyllama_comprehensive_2d.py`**: **[推薦]** 綜合量化腳本，支援 BFP, AWQ-Fix, AWQ-Mix 以及所有 2D Block 形狀的掃描。
- **`evaluate_with_runtime_quantization.py`**: 使用 Runtime Quantization 進行評估的範例（不佔用磁碟空間）。
- **`run_quantize_tinyllama_comprehensive.sh`**: 執行完整的 TinyLlama 綜合量化試驗。
- **`quantize_llama_3_2_1b.sh`**: 針對 Llama 3.2 1B 的完整量化實驗組合。

---

## 📊 分類實驗腳本 (Categorized Scripts)

### [BFP] Block Floating Point
位於 `block-floating-point/`：
- `bfp_quantize.py`: 1D BFP 量化執行腳本。
- `bfp_quantize_2d.py`: 2D BFP 量化執行腳本。
- `run_bfp_only.sh`: 執行所有 BFP 相關實驗。
- `run_quantize_tinyllama_2d_sweep.sh`: 針對 2D BFP 進行區塊形狀掃描的 Shell 腳本。

### [AWQ-Fix] Fix-Precision AWQ
位於 `fix-precision-awq/`：
- `fix_precision_awq.py`: AWQ 核心執行腳本。
- `run_awq_fix_only.sh`: 執行 AWQ Fix-Precision 實驗。

### [AWQ-Mix] Mix-Precision AWQ
位於 `mix-precision-awq/`：
- `mix_precision_awq.py`: AWQ 混合精度執行腳本。
- `run_awq_mix_only.sh`: 執行 AWQ Mix-Precision 實驗。

---

## 📖 詳細文檔 (Documentation)

更多詳細資訊請參閱 `docs/` 目錄：
- [綜合腳本使用說明](docs/README_COMPREHENSIVE_2D.md)
- [Runtime Quantization 詳細介紹](docs/RUNTIME_QUANTIZATION.md)
- [2D 量化技術細節](docs/README_2D_QUANTIZATION.md)
- [存儲格式說明](docs/STORAGE_FORMAT.md)

---
*整理日期：2025-12-19*
