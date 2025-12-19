# Debug 和測試檔案清單

本文檔列出專案中所有用於 **debug**、**測試**、**驗證** 和 **分析** 的 Python 檔案。

---

## 📋 分類說明

- **🔍 Debug 檔案**: 用於除錯、比較不同實作
- **✅ 測試檔案**: 單元測試、基準測試
- **📊 分析檔案**: 視覺化、統計分析、權重分析
- **🧪 實驗檔案**: 精度實驗、配置掃描
- **📝 範例檔案**: 示範如何使用模型
- **🗑️ 可刪除**: 不再需要或已被更好的工具取代

---

## 1. 根目錄的 Debug 檔案

### 🔍 dtype 相關檢查

| 檔案 | 用途 | 狀態 | 建議 |
|------|------|------|------|
| `check_dtype.py` | 檢查 HF 模型的參數和輸出 dtype | 🗑️ 舊版 | **可刪除**（功能已被 `verify_dtype.py` 取代） |
| `verify_dtype.py` | 驗證 `CustomLlamaModel` 各層的 dtype | 🗑️ 舊版 | **可刪除**（已完成驗證，不再需要） |
| `check_lm_eval_dtype.py` | 檢查 lm-eval 使用的 dtype | 🗑️ 舊版 | **可刪除**（已確認 lm-eval 使用 auto dtype） |
| `check_precision.py` | 比較 lm-eval 和自定義模型的精度 | 🗑️ 舊版 | **可刪除**（已被更完整的工具取代） |

**總結**: 這 4 個檔案都是早期 dtype 調查時使用的，現在可以安全刪除。

---

### 🔍 模型比較和驗證

| 檔案 | 用途 | 狀態 | 建議 |
|------|------|------|------|
| `compare_models.py` | 比較 HFLM 和 CustomLlamaModel 的逐層輸出 | 🗑️ 部分過時 | **可刪除**（已被 `scripts/compare_all_backends.py` 取代） |
| `compare_qkv_attention.py` | 詳細比較 Q/K/V 和 attention 計算 | 🔍 專門工具 | **保留**（用於深度 debug attention 問題） |

**說明**:
- `compare_models.py`: 早期用於驗證自定義模型正確性，現在 `scripts/compare_all_backends.py` 功能更完整
- `compare_qkv_attention.py`: 非常詳細的 attention 層面分析，如果未來需要 debug attention 可保留

---

### 📝 其他

| 檔案 | 用途 | 狀態 | 建議 |
|------|------|------|------|
| `tinyllama_gt.py` | Ground truth 測試？ | ❓ 不明 | **需檢查內容** |
| `test_precision_policy.py` | 測試 PrecisionPolicy 類別和 backend 比較 | ✅ 單元測試（已驗證） | **保留**（確保 precision_policy.py 正確性，驗證 HF/clone/custom backend 一致性） |

---

## 2. `scripts/` 目錄

### 🧪 精度實驗工具

| 檔案 | 用途 | 狀態 | 建議 |
|------|------|------|------|
| `sweep_precision_configs.py` | 掃描不同精度配置，找最佳設定 | ✅ 主要工具 | **保留**（用於 logits 層級精度掃描） |
| `compare_all_backends.py` | 比較 HF/clone/custom 三種 backend | ✅ 主要工具 | **保留**（用於驗證 backend 正確性） |
| `debug_hf_vs_clone.py` | 逐步 debug HF 和 clone 的差異 | 🔍 專門工具 | **保留**（用於深度 debug clone backend） |

**說明**: 這三個都是重要的驗證和實驗工具，應保留。

---

### 📦 資料準備工具

| 檔案 | 用途 | 狀態 | 建議 |
|------|------|------|------|
| `prepare_redpajama.py` | 準備 RedPajama 資料集 | 📦 資料工具 | **保留**（如果未來要 pretrain） |
| `prepare_slimpajama.py` | 準備 SlimPajama 資料集 | 📦 資料工具 | **保留**（如果未來要 pretrain） |
| `prepare_starcoder.py` | 準備 StarCoder 資料集 | 📦 資料工具 | **保留**（如果未來要 pretrain） |
| `convert_hf_checkpoint.py` | 轉換 HF checkpoint | 📦 轉換工具 | **保留** |
| `convert_lit_checkpoint.py` | 轉換 Lit-GPT checkpoint | 📦 轉換工具 | **保留** |

**說明**: 這些是資料準備工具，如果不打算做 pretrain 可以移到單獨的資料夾。

---

## 3. `gt_test/` 目錄（Ground Truth 測試）

| 檔案 | 用途 | 狀態 | 建議 |
|------|------|------|------|
| `BoolQ_test.py` | BoolQ 基準測試 | 🗑️ 舊版 | **可刪除**（已有 `awq/task_script_boolq.py`） |
| `HellaSwag_test.py` | HellaSwag 基準測試 | 🗑️ 舊版 | **可刪除**（已有 `awq/task_script_hellaswag.py`） |
| `mmlu_test.py` | MMLU 基準測試 | 🗑️ 舊版 | **可刪除**（已有更完整的 lm-eval 整合） |

**總結**: `gt_test/` 整個目錄可以刪除，功能已被 `awq/` 下的評估腳本取代。

---

## 4. `plain_script/` 目錄

| 檔案 | 用途 | 狀態 | 建議 |
|------|------|------|------|
| `plain_script.py` | **核心**: Custom backend 實作 | ✅ 生產代碼 | **必須保留** |
| `sample_main.py` | 示範如何使用 plain_script | 📝 範例 | **保留**（作為範例文檔） |
| `get_weight.py` | 視覺化權重和預測 | 📊 分析工具 | **可選保留**（如需分析權重分佈） |
| `latent_activation_distribution_analyze.py` | 分析隱藏層啟動分佈 | 📊 分析工具 | **可選保留**（如需分析啟動） |

**說明**:
- `plain_script.py`: 核心代碼，必須保留
- `sample_main.py`: 作為使用範例，建議保留
- 其他兩個分析工具：如果不需要分析可刪除

---

## 5. `plot_element_contribution/` 目錄

| 檔案 | 用途 | 狀態 | 建議 |
|------|------|------|------|
| `plot_element_contribution.py` | 視覺化元素貢獻度 | 📊 分析工具 | **可選保留**（如需分析貢獻度） |

**說明**: 如果不需要分析元素貢獻度，整個目錄可以刪除或歸檔。

---

## 6. `plot_weight/` 目錄

| 檔案 | 用途 | 狀態 | 建議 |
|------|------|------|------|
| `plot_weight.py` | 視覺化權重分佈 | 📊 分析工具 | **可選保留**（如需視覺化權重） |

**說明**: 如果不需要視覺化權重，整個目錄可以刪除或歸檔。

---

## 7. `awq/` 目錄

| 檔案 | 用途 | 狀態 | 建議 |
|------|------|------|------|
| `task_script_hellaswag.py` | HellaSwag 評估（支援 BFP/AWQ） | ✅ 主要工具 | **保留** |
| `task_script_piqa.py` | PIQA 評估 | ✅ 主要工具 | **保留** |
| `task_script_boolq.py` | BoolQ 評估 | ✅ 主要工具 | **保留** |
| `task_script_arc_c.py` | ARC-Challenge 評估 | ✅ 主要工具 | **保留** |
| `task_script_arc_e.py` | ARC-Easy 評估 | ✅ 主要工具 | **保留** |
| `task_script_obqa.py` | OBQA 評估 | ✅ 主要工具 | **保留** |
| `task_script_winogrande.py` | Winogrande 評估 | ✅ 主要工具 | **保留** |
| `analyze_hellaswag_results.py` | 分析 HellaSwag 結果 | ✅ 分析工具 | **保留** |

**說明**: `awq/` 目錄是正式的評估工具集，全部保留。

---

## 8. `my_script/` 目錄

| 檔案 | 用途 | 狀態 | 建議 |
|------|------|------|------|
| `generate_bft.py` | 生成 Block Floating Point 相關？ | ❓ 不明 | **需檢查內容** |

---

## 📊 刪除建議總結

### 🗑️ 可以安全刪除的檔案

#### 根目錄（6 個檔案）
```bash
rm check_dtype.py
rm verify_dtype.py
rm check_lm_eval_dtype.py
rm check_precision.py
rm compare_models.py
```

#### gt_test/ 目錄（整個目錄）
```bash
rm -rf gt_test/
```

**原因**: 這些都是早期 debug 和測試檔案，功能已被更完整的工具取代。

---

### 📂 可選刪除/歸檔的目錄

如果你不需要以下功能，可以刪除或移到 `archive/` 目錄：

#### 視覺化和分析工具
```bash
# 如果不需要視覺化分析
mkdir -p archive/analysis_tools
mv plot_element_contribution/ archive/analysis_tools/
mv plot_weight/ archive/analysis_tools/
mv plain_script/get_weight.py archive/analysis_tools/
mv plain_script/latent_activation_distribution_analyze.py archive/analysis_tools/
```

#### 資料準備工具（如果不做 pretrain）
```bash
mkdir -p archive/data_preparation
mv scripts/prepare_redpajama.py archive/data_preparation/
mv scripts/prepare_slimpajama.py archive/data_preparation/
mv scripts/prepare_starcoder.py archive/data_preparation/
```

---

### ✅ 必須保留的檔案

#### 核心代碼
- `precision_policy.py` - 精度策略管理
- `utils.py` - 通用工具函數
- `plain_script/plain_script.py` - Custom backend 實作
- `llama_backend/` - 新的 backend 架構（如果已遷移）
- `block_quantization/` - 量化相關代碼

#### 評估和實驗工具
- `scripts/sweep_precision_configs.py`
- `scripts/compare_all_backends.py`
- `scripts/debug_hf_vs_clone.py`
- `awq/` 目錄下所有檔案

#### 測試檔案
- `test_precision_policy.py`

#### 範例和文檔
- `plain_script/sample_main.py`

---

## 🔍 需要進一步檢查的檔案

這些檔案我無法從檔名判斷用途，建議你檢查內容後決定：

1. **`tinyllama_gt.py`** - 根目錄，不確定用途
2. **`my_script/generate_bft.py`** - 不確定是否還在使用

---

## 📝 建議的清理步驟

### 第 1 步：立即刪除（低風險）
```bash
cd /home/frank23026407/TinyLlama

# 刪除舊的 dtype 檢查檔案
rm check_dtype.py check_lm_eval_dtype.py verify_dtype.py check_precision.py

# 刪除舊的比較檔案
rm compare_models.py

# 刪除舊的測試目錄
rm -rf gt_test/
```

### 第 2 步：歸檔分析工具（可選）
```bash
# 創建歸檔目錄
mkdir -p archive/analysis_tools

# 移動分析工具
mv plot_element_contribution/ archive/analysis_tools/
mv plot_weight/ archive/analysis_tools/
mv plain_script/get_weight.py archive/analysis_tools/
mv plain_script/latent_activation_distribution_analyze.py archive/analysis_tools/
mv plain_script/*.png archive/analysis_tools/ 2>/dev/null || true
mv plain_script/latent_activation_plots/ archive/analysis_tools/ 2>/dev/null || true
```

### 第 3 步：歸檔資料準備工具（如果不做 pretrain）
```bash
mkdir -p archive/data_preparation

mv scripts/prepare_redpajama.py archive/data_preparation/
mv scripts/prepare_slimpajama.py archive/data_preparation/
mv scripts/prepare_starcoder.py archive/data_preparation/
mv scripts/convert_hf_checkpoint.py archive/data_preparation/
mv scripts/convert_lit_checkpoint.py archive/data_preparation/
```

### 第 4 步：檢查並決定
```bash
# 檢查這些檔案的內容，再決定是否刪除
head -50 tinyllama_gt.py
head -50 my_script/generate_bft.py
head -50 compare_qkv_attention.py
```

---

## 📈 清理後的檔案結構

清理後，你的主要 Python 檔案應該是：

```
TinyLlama/
├── precision_policy.py              # 精度策略管理
├── utils.py                         # 通用工具
├── test_precision_policy.py         # 單元測試
│
├── plain_script/
│   ├── plain_script.py              # Custom backend (核心)
│   └── sample_main.py               # 使用範例
│
├── llama_backend/                   # 新 backend 架構
│   ├── tinyllama_my.py
│   └── clone/
│       ├── hf_clone.py
│       ├── hf_rope.py
│       └── clone_backend.py
│
├── block_quantization/              # 量化工具
│   ├── __init__.py
│   ├── awq.py
│   └── block_quantization.py
│
├── scripts/                         # 實驗和驗證工具
│   ├── sweep_precision_configs.py
│   ├── compare_all_backends.py
│   └── debug_hf_vs_clone.py
│
├── awq/                            # 評估腳本
│   ├── task_script_*.py       (7 個評估腳本)
│   ├── analyze_hellaswag_results.py
│   └── run_*.sh                    (各種執行腳本)
│
├── quantize_model_script/          # 量化腳本
├── pretrain/                       # Pretrain 相關
├── sft/                            # SFT 相關
└── archive/                        # 歸檔的舊檔案
    ├── analysis_tools/
    └── data_preparation/
```

---

## ⚠️ 注意事項

1. **備份**: 在刪除任何檔案前，建議先用 Git commit 或創建備份
2. **依賴檢查**: 確保沒有其他腳本 import 這些要刪除的檔案
3. **Git 歷史**: 如果需要找回舊代碼，可以從 Git 歷史恢復

---

## 🎯 結論

- **立即刪除**: 6 個檔案 + `gt_test/` 目錄（約 9 個檔案）
- **可選歸檔**: 分析工具和資料準備工具（約 10 個檔案）
- **必須保留**: 核心代碼、評估工具、實驗腳本（約 20+ 個檔案）
- **需要檢查**: 2-3 個不確定用途的檔案

清理後可以減少約 15-25 個 debug/測試檔案，讓專案結構更清晰！
