# AWQ 量化模型 HellaSwag 評估指南

## 📋 概述

本文件說明如何在 HellaSwag 基準測試上評估 AWQ (Activation-Aware Weight Quantization) 量化後的 TinyLlama 模型。

## 🎯 可用的 AWQ 模型

### Mix-Precision AWQ (動態量化)
```
model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m2  # 2-bit mantissa
model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m3  # 3-bit mantissa
model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m4  # 4-bit mantissa
model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m5  # 5-bit mantissa
```

### Fix-Precision AWQ (靜態量化)
```
model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m2   # 2-bit mantissa
model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m3   # 3-bit mantissa
model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m4   # 4-bit mantissa
model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m5   # 5-bit mantissa
```

**參數說明**:
- **Block Size (b128)**: 128 個元素為一個區塊
- **Mantissa Bits (m2-m5)**: 尾數位元數，影響量化精度
  - m2: 最激進壓縮，最大精度損失
  - m5: 較溫和壓縮，較小精度損失

## 🚀 使用方法

### 方法 1: 完整精度掃描（推薦用於正式實驗）

評估所有 8 個 AWQ 模型（加上原始精度配置）：

```bash
cd /home/frank23026407/TinyLlama
sh awq/run_precision_sweep_hellaswag.sh
```

**執行時間**: 約 3-5 小時  
**輸出位置**: `awq/log/precision_sweep/awq_*.log`

### 方法 2: 快速測試（推薦用於開發/驗證）

僅使用 100 個樣本快速評估：

```bash
cd /home/frank23026407/TinyLlama
sh awq/run_precision_sweep_quick.sh
```

**執行時間**: 約 10-15 分鐘  
**輸出位置**: `awq/log/precision_sweep_quick/awq_*.log`

### 方法 3: 單一模型評估

評估特定 AWQ 模型：

```bash
cd /home/frank23026407/TinyLlama

# 範例：評估 mix-precision b128 m4
conda run -n tinyllama-env python awq/tinyllama_my_bfp_hellaswag.py \
  --backend huggingface \
  --model_path model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m4

# 快速測試（100 樣本）
conda run -n tinyllama-env python awq/tinyllama_my_bfp_hellaswag.py \
  --backend huggingface \
  --model_path model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m3 \
  --max_samples 100
```

### 方法 4: 整合測試（快速驗證）

快速驗證 AWQ 模型載入與評估是否正常：

```bash
cd /home/frank23026407/TinyLlama
bash awq/test_awq_integration.sh
```

**執行時間**: 約 2-3 分鐘  
**輸出**: 測試 mix-precision-m2 和 fix-precision-m2，各 5 個樣本

## 📊 結果分析

### 自動分析所有結果

```bash
# 分析完整掃描結果
python awq/analyze_hellaswag_results.py awq/log/precision_sweep

# 分析快速測試結果
python awq/analyze_hellaswag_results.py awq/log/precision_sweep_quick
```

**輸出範例**:
```
====================================================================================================
HellaSwag Precision Configuration Comparison
====================================================================================================

Configuration                                            Total      Acc Acc_Norm  Diff_Acc Diff_Norm
----------------------------------------------------------------------------------------------------
HuggingFace Reference (Ground Truth)                    10042   0.5234   0.5896       REF       REF
Clone: BF16+BF16+FP32_Softmax (Best)                    10042   0.5231   0.5894   -0.0003  -0.0002 ★
AWQ Mix-Precision b128 m5 (HF)                                10042   0.5180   0.5840   -0.0054  -0.0056
AWQ Fix-Precision b128 m5 (HF)                                 10042   0.5175   0.5835   -0.0059  -0.0061
AWQ Mix-Precision b128 m4 (HF)                                10042   0.5120   0.5780   -0.0114  -0.0116
AWQ Fix-Precision b128 m4 (HF)                                 10042   0.5110   0.5770   -0.0124  -0.0126
AWQ Mix-Precision b128 m3 (HF)                                10042   0.4950   0.5610   -0.0284  -0.0286
AWQ Fix-Precision b128 m3 (HF)                                 10042   0.4920   0.5580   -0.0314  -0.0316
AWQ Mix-Precision b128 m2 (HF)                                10042   0.4600   0.5260   -0.0634  -0.0636
AWQ Fix-Precision b128 m2 (HF)                                 10042   0.4580   0.5240   -0.0654  -0.0656
...
```

### 手動提取結果

```bash
# 快速查看所有 AWQ 模型的 acc_norm
grep "Accuracy (acc_norm):" awq/log/precision_sweep/awq_*.log

# 比較 mix-precision vs fix-precision
grep "Accuracy (acc_norm):" awq/log/precision_sweep/awq_*mix-precision*.log
grep "Accuracy (acc_norm):" awq/log/precision_sweep/awq_*fix-precision*.log
```

## 📈 預期結果趨勢

基於量化理論，預期的效能排序：

1. **Full Precision (BF16)**: 最高精度 ≈ 0.589
2. **AWQ m5**: 輕微精度損失 ≈ 0.584
3. **AWQ m4**: 中等精度損失 ≈ 0.578
4. **AWQ m3**: 明顯精度損失 ≈ 0.561
5. **AWQ m2**: 較大精度損失 ≈ 0.526

**Mix-Precision vs Fix-Precision 差異**:
- Mix-Precision 通常略優於 Fix-Precision（約 0.1-0.2% acc_norm）
- Mix-Precision 在推理時動態計算量化參數
- Fix-Precision 在量化時固定參數

## 🔍 技術細節

### AWQ 模型載入

AWQ 模型使用 HuggingFace backend 直接載入：

```python
# tinyllama_my_bfp_hellaswag.py 中的實作
my_model = LlamaMyModel(
    model_name=model_path,  # 指向 AWQ 量化模型路徑
    device=device,
    dtype=torch.bfloat16,
    backend="huggingface"   # 使用 HF 原生 forward
)
```

### Log 檔案命名規範

```
awq_TinyLlama_1.1v-awq-quantized-{type}-b128-m{bits}.log

其中:
  {type} = mix-precision 或 fix-precision
  {bits} = 2, 3, 4, 或 5
```

### 分析器支援

`analyze_hellaswag_results.py` 已新增 AWQ 模型名稱映射：

```python
mappings = {
  'awq_TinyLlama_1.1v-awq-quantized-mix-precision-b128-m2': 'AWQ Mix-Precision b128 m2 (HF)',
    'awq_TinyLlama_1.1v-awq-quantized-mix-precision-b128-m3': 'AWQ Mix-Precision b128 m3 (HF)',
    # ... 其他 AWQ 模型
}
```

## 🎯 實驗建議

### 步驟 1: 快速驗證（5 分鐘）

```bash
bash awq/test_awq_integration.sh
```

確認 AWQ 模型可以正常載入和評估。

### 步驟 2: 快速比較（15 分鐘）

```bash
sh awq/run_precision_sweep_quick.sh
python awq/analyze_hellaswag_results.py awq/log/precision_sweep_quick
```

使用 100 樣本快速了解趨勢。

### 步驟 3: 完整評估（3-5 小時）

```bash
sh awq/run_precision_sweep_hellaswag.sh
python awq/analyze_hellaswag_results.py awq/log/precision_sweep
```

獲得完整的 HellaSwag 驗證集分數。

### 步驟 4: 結果分析

比較項目：
1. **Mantissa bits 影響**: m2 vs m3 vs m4 vs m5
2. **量化方法差異**: Mix-Precision vs Fix-Precision
3. **與 Full Precision 差距**: AWQ vs BF16 reference
4. **壓縮率 vs 精度權衡**: 模型大小 vs acc_norm

## 🐛 故障排除

### 問題：找不到模型檔案

```bash
# 檢查模型是否存在
ls -lh model/TinyLlama_1.1v-awq-quantized-*

# 確保使用絕對路徑或從專案根目錄執行
cd /home/frank23026407/TinyLlama
```

### 問題：ModuleNotFoundError

```bash
# 確保使用 conda 環境
conda run -n tinyllama-env python awq/tinyllama_my_bfp_hellaswag.py ...
```

### 問題：記憶體不足

```bash
# 使用 --max_samples 限制資料集大小
python awq/tinyllama_my_bfp_hellaswag.py \
  --backend huggingface \
  --model_path model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m4 \
  --max_samples 1000  # 只評估 1000 個樣本
```

### 問題：分析器無法識別 log

確保 log 檔案命名符合格式：
```bash
# 正確的檔名格式
awq_TinyLlama_1.1v-awq-quantized-mix-precision-b128-m2.log

# 錯誤的檔名（會被忽略）
test_awq_mix-precision_m2.log
```

## 📝 相關文件

- [`NEW_FILES_SUMMARY.md`](NEW_FILES_SUMMARY.md) - 完整檔案清單
- [`README_PRECISION_SWEEP.md`](README_PRECISION_SWEEP.md) - 精度掃描使用指南
- [`PRECISION_SWEEP_SUMMARY.md`](PRECISION_SWEEP_SUMMARY.md) - 實作技術細節
- [`../USAGE.md`](../USAGE.md) - LlamaMyModel 完整使用指南
- [`../docs/backend_comparison.md`](../docs/backend_comparison.md) - Backend 比較

## 🎉 總結

現在你可以：

✅ 評估 8 個不同配置的 AWQ 量化模型  
✅ 比較 Mix-Precision vs Fix-Precision 量化方法  
✅ 分析 Mantissa bits 對精度的影響  
✅ 找出最佳的壓縮率/精度平衡點  
✅ 使用自動化分析工具生成比較表格  

祝實驗順利！🚀
