# AWQ 目錄新增檔案說明

## 📋 總覽

本次修改為 `awq/` 目錄新增了精度配置掃描功能，允許在 HellaSwag 基準測試上比較不同精度設定的影響。

## 📁 新增的檔案

### 1. 執行腳本（Shell Scripts）

#### `run_precision_sweep_hellaswag.sh` ⭐
**用途**: 完整的精度配置掃描  
**測試配置**: 16 種（6 個 Clone 配置 + 1 個 HF 參考 + 2 個 Custom 配置 + 8 個 AWQ 量化模型）  
**資料集大小**: 完整驗證集（~10,000 樣本）  
**執行時間**: 約 2-4 小時  
**輸出位置**: `awq/log/precision_sweep/*.log`

**測試的配置**:
- Config 1: BF16 compute + BF16 RoPE + FP32 softmax ⭐（最佳）
- Config 2: Full BF16
- Config 3: FP32 compute + BF16 RoPE + FP32 softmax
- Config 4: BF16 compute + FP32 RoPE + FP32 softmax
- Config 5: Full FP32（最高精度）
- Config 6: HuggingFace reference（ground truth）
- Config 7: Custom backend (default policy)
- Config 8: Custom backend (bf16 policy)
 - Config 9-12: AWQ Mix-Precision b128 m2-m5（HF backend）
 - Config 13-16: AWQ Fix-Precision b128 m2-m5（HF backend）

```bash
# 使用方法
cd /home/frank23026407/TinyLlama
sh awq/run_precision_sweep_hellaswag.sh
```

#### `run_precision_sweep_quick.sh` ⚡
**用途**: 快速精度測試  
**測試配置**: 13 種（5 種主要配置 + 8 個 AWQ 量化模型，100 樣本）  
**資料集大小**: 100 樣本  
**執行時間**: 約 5-10 分鐘  
**輸出位置**: `awq/log/precision_sweep_quick/*.log`

**測試的配置**:
- HuggingFace reference
- Clone: BF16 + BF16 + FP32 softmax
- Clone: Full BF16
- Clone: Full FP32
- Custom: bf16 policy
 - AWQ Mix-Precision/Fix-Precision b128 m2-m5（HF backend, 100 樣本）

```bash
# 使用方法（推薦先用這個）
cd /home/frank23026407/TinyLlama
sh awq/run_precision_sweep_quick.sh
```

#### `validate_modifications.sh` ✅
**用途**: 快速驗證修改是否正常工作  
**測試配置**: 3 種（HF + Clone + Custom）  
**資料集大小**: 5 樣本  
**執行時間**: 約 1-2 分鐘  

```bash
# 使用方法
cd /home/frank23026407/TinyLlama
sh awq/validate_modifications.sh
```

#### `test_awq_integration.sh` ✅ (NEW)
**用途**: 驗證 AWQ 模型整合是否正常  
**測試配置**: 2 種 AWQ 模型（mix-precision-m2 + fix-precision-m2）  
**資料集大小**: 5 樣本  
**執行時間**: 約 2-3 分鐘  

```bash
# 使用方法
cd /home/frank23026407/TinyLlama
bash awq/test_awq_integration.sh
```

#### `awq_help.sh` 📖 (NEW)
**用途**: 顯示 AWQ 評估快速參考指南  
**內容**: 所有可用命令、模型列表、分析方法、故障排除  

```bash
# 使用方法
bash awq/awq_help.sh
```

### 2. Python 檔案

#### `analyze_hellaswag_results.py` 📊
**用途**: 分析和比較 HellaSwag 評估結果

**功能**:
- 解析所有 log 檔案
- 提取 `acc` 和 `acc_norm` 指標
- 計算與 HuggingFace 參考的差異
- 生成排序比較表格
- 提供精度影響分析

**使用方法**:
```bash
# 分析快速測試結果
python awq/analyze_hellaswag_results.py awq/log/precision_sweep_quick

# 分析完整掃描結果
python awq/analyze_hellaswag_results.py awq/log/precision_sweep
```

**輸出範例**:
```
====================================================================================================
HellaSwag Precision Configuration Comparison
====================================================================================================

Configuration                                            Total      Acc  Acc_Norm  Diff_Acc Diff_Norm
----------------------------------------------------------------------------------------------------
HuggingFace Reference (Ground Truth)                    10042   0.5234    0.5896       REF       REF
Clone: BF16+BF16+FP32_Softmax (Best)                    10042   0.5231    0.5894   -0.0003   -0.0002 ★
...
```

### 3. 文件檔案

#### `README_PRECISION_SWEEP.md` 📖
**用途**: 完整的使用指南

**內容**:
- 檔案說明
- 使用方法和範例
- 精度配置詳細說明
- 預期結果
- 故障排除指南
- 相關文件連結

#### `PRECISION_SWEEP_SUMMARY.md` 📝
**用途**: 實作總結文件

**內容**:
- 完成的修改列表
- 使用範例
- 技術細節和關鍵發現
- Log 檔案位置
- 下一步建議

#### `AWQ_MODELS_HELLASWAG.md` 📖 (NEW)
**用途**: AWQ 量化模型評估完整指南

**內容**:
- 8 個 AWQ 模型說明（Mix-Precision/Fix-Precision，m2-m5）
- 使用方法（快速/完整掃描）
- 結果分析指引
- 預期效能趨勢
- 技術細節和故障排除

## 🔧 修改的現有檔案

### `tinyllama_my_bfp_hellaswag.py`

**新增的命令列參數**:

```python
# Clone backend 精度控制
--compute_dtype {fp32,bf16,fp16}     # 主計算 dtype
--rope_cache_dtype {fp32,bf16,fp16}  # RoPE cache dtype
--softmax_fp32                        # 是否對 attention softmax 使用 FP32

# Custom backend 精度控制
--precision_policy {default,match_hf,bf16}  # Precision policy

# 通用
--max_samples N  # 限制評估樣本數（用於快速測試）
```

**新增的程式碼邏輯**:
1. 解析精度參數並轉換為 PyTorch dtype
2. 根據 backend 類型構建對應的 kwargs
3. 印出當前配置資訊
4. 支援限制資料集大小

## 🚀 快速開始

### 步驟 1: 驗證修改
```bash
cd /home/frank23026407/TinyLlama
sh awq/validate_modifications.sh
```

### 步驟 2: 快速測試
```bash
sh awq/run_precision_sweep_quick.sh
```

### 步驟 3: 分析結果
```bash
python awq/analyze_hellaswag_results.py awq/log/precision_sweep_quick
```

### 步驟 4: 完整掃描（可選）
```bash
sh awq/run_precision_sweep_hellaswag.sh
python awq/analyze_hellaswag_results.py awq/log/precision_sweep
```

## 📊 輸出檔案結構

```
awq/
├── log/
│   ├── precision_sweep/              # 完整掃描結果
│   │   ├── config1_bf16_bf16_fp32softmax.log
│   │   ├── config2_bf16_bf16_bf16softmax.log
│   │   ├── config3_fp32_bf16_fp32softmax.log
│   │   ├── config4_bf16_fp32_fp32softmax.log
│   │   ├── config5_fp32_fp32_fp32softmax.log
│   │   ├── config6_huggingface_reference.log
│   │   ├── config7_custom_default.log
│   │   └── config8_custom_bf16.log
│   │   ├── awq_TinyLlama_1.1v-awq-quantized-mix-precision-b128-m2.log
│   │   ├── awq_TinyLlama_1.1v-awq-quantized-mix-precision-b128-m3.log
│   │   ├── awq_TinyLlama_1.1v-awq-quantized-mix-precision-b128-m4.log
│   │   ├── awq_TinyLlama_1.1v-awq-quantized-mix-precision-b128-m5.log
│   │   ├── awq_TinyLlama_1.1v-awq-quantized-fix-precision-b128-m2.log
│   │   ├── awq_TinyLlama_1.1v-awq-quantized-fix-precision-b128-m3.log
│   │   ├── awq_TinyLlama_1.1v-awq-quantized-fix-precision-b128-m4.log
│   │   └── awq_TinyLlama_1.1v-awq-quantized-fix-precision-b128-m5.log
│   │
│   └── precision_sweep_quick/        # 快速測試結果
│       ├── hf_reference.log
│       ├── clone_bf16_bf16_fp32soft.log
│       ├── clone_full_bf16.log
│       ├── clone_full_fp32.log
│       └── custom_bf16.log
│       ├── awq_TinyLlama_1.1v-awq-quantized-mix-precision-b128-m2.log
│       ├── awq_TinyLlama_1.1v-awq-quantized-mix-precision-b128-m3.log
│       ├── awq_TinyLlama_1.1v-awq-quantized-mix-precision-b128-m4.log
│       ├── awq_TinyLlama_1.1v-awq-quantized-mix-precision-b128-m5.log
│       ├── awq_TinyLlama_1.1v-awq-quantized-fix-precision-b128-m2.log
│       ├── awq_TinyLlama_1.1v-awq-quantized-fix-precision-b128-m3.log
│       ├── awq_TinyLlama_1.1v-awq-quantized-fix-precision-b128-m4.log
│       └── awq_TinyLlama_1.1v-awq-quantized-fix-precision-b128-m5.log
│
├── tinyllama_my_bfp_hellaswag.py    # 修改：新增精度參數
├── run_precision_sweep_hellaswag.sh  # 新增：完整掃描
├── run_precision_sweep_quick.sh      # 新增：快速測試
├── validate_modifications.sh         # 新增：驗證腳本
├── analyze_hellaswag_results.py      # 新增：結果分析
├── README_PRECISION_SWEEP.md         # 新增：使用指南
└── PRECISION_SWEEP_SUMMARY.md        # 新增：實作總結
```

## 🔗 相關文件

- [`USAGE.md`](../USAGE.md) - LlamaMyModel 完整使用指南
- [`docs/backend_comparison.md`](../docs/backend_comparison.md) - Backend 詳細比較
- [`scripts/sweep_precision_configs.py`](../scripts/sweep_precision_configs.py) - Logits 層級精度掃描
- [`scripts/compare_all_backends.py`](../scripts/compare_all_backends.py) - Backend 比較工具

## 💡 使用建議

1. **開發/除錯階段**: 使用 `validate_modifications.sh` 或 `run_precision_sweep_quick.sh`
2. **正式實驗**: 使用 `run_precision_sweep_hellaswag.sh`
3. **單一配置測試**: 直接呼叫 `tinyllama_my_bfp_hellaswag.py` 並指定參數
4. **結果比較**: 使用 `analyze_hellaswag_results.py` 產生比較表格

## ⚠️ 注意事項

1. 完整掃描需要 2-4 小時，請確保有足夠時間
2. 記憶體不足時可減少 `--max_samples` 參數
3. 所有腳本都已設定為可執行（`chmod +x`）
4. Log 檔案會自動儲存，不會覆蓋舊檔案（除非檔名相同）

## 🎯 預期成果

完成掃描後，你應該能夠：

1. ✅ 量化不同精度配置對 HellaSwag 分數的影響
2. ✅ 找出最佳的精度/效能平衡點
3. ✅ 驗證 Clone 和 Custom backend 的正確性
4. ✅ 了解哪些精度參數最重要
5. ✅ 為生產環境選擇合適的配置
