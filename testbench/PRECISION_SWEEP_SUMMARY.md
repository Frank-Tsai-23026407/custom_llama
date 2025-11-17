# Precision Sweep 實作總結

## 完成的修改

### 1. 修改的檔案

#### `awq/task_script_hellaswag.py`
新增以下參數支援：

**Clone Backend 精度控制：**
- `--compute_dtype {fp32,bf16,fp16}` - 主計算的 dtype
- `--rope_cache_dtype {fp32,bf16,fp16}` - RoPE cache 的 dtype
- `--softmax_fp32` - 是否對 attention softmax 使用 FP32

**Custom Backend 精度控制：**
- `--precision_policy {default,match_hf,bf16}` - Precision policy 選擇

**通用參數：**
- `--max_samples N` - 限制評估樣本數（用於快速測試）

### 2. 新增的腳本

#### `awq/run_precision_sweep_hellaswag.sh`
完整的精度配置掃描腳本，測試 8 種配置：

1. Clone: BF16 compute + BF16 RoPE + FP32 softmax（最佳配置）
2. Clone: BF16 compute + BF16 RoPE + BF16 softmax
3. Clone: FP32 compute + BF16 RoPE + FP32 softmax
4. Clone: BF16 compute + FP32 RoPE + FP32 softmax
5. Clone: FP32 compute + FP32 RoPE + FP32 softmax（最高精度）
6. HuggingFace: 參考實作（ground truth）
7. Custom: default policy
8. Custom: bf16 policy

#### `awq/run_precision_sweep_quick.sh`
快速測試腳本，只測試 5 種主要配置，使用 100 個樣本：

1. HuggingFace reference
2. Clone: BF16 + BF16 + FP32 softmax
3. Clone: Full BF16
4. Clone: Full FP32
5. Custom: bf16 policy

#### `awq/analyze_hellaswag_results.py`
結果分析工具：
- 解析所有 log 檔案
- 提取 acc 和 acc_norm 指標
- 計算與 HuggingFace 參考的差異
- 生成排序比較表格
- 標示最佳非參考配置

#### `awq/README_PRECISION_SWEEP.md`
完整的使用文件，包含：
- 檔案說明
- 使用方法
- 精度配置說明
- 預期結果
- 故障排除指南

## 使用範例

### 快速測試（推薦）

```bash
cd /home/frank23026407/TinyLlama

# 執行快速掃描（5 種配置，100 個樣本，約 5-10 分鐘）
sh awq/run_precision_sweep_quick.sh

# 分析結果
python awq/analyze_hellaswag_results.py awq/log/precision_sweep_quick
```

### 完整評估

```bash
cd /home/frank23026407/TinyLlama

# 執行完整掃描（8 種配置，完整驗證集，約 2-4 小時）
sh awq/run_precision_sweep_hellaswag.sh

# 分析結果
python awq/analyze_hellaswag_results.py awq/log/precision_sweep
```

### 測試單一配置

```bash
# 測試 Clone backend 的最佳配置（只用 100 個樣本）
python awq/task_script_hellaswag.py \
    --backend clone \
    --compute_dtype bf16 \
    --rope_cache_dtype bf16 \
    --softmax_fp32 \
    --max_samples 100
```

## 預期結果格式

分析腳本會產生類似以下的表格：

```
====================================================================================================
HellaSwag Precision Configuration Comparison
====================================================================================================

Configuration                                            Total      Acc  Acc_Norm  Diff_Acc Diff_Norm
----------------------------------------------------------------------------------------------------
HuggingFace Reference (Ground Truth)                    10042   0.5234    0.5896       REF       REF
Clone: BF16+BF16+FP32_Softmax (Best)                    10042   0.5231    0.5894   -0.0003   -0.0002 ★
Clone: Full FP32 (Max Precision)                        10042   0.5230    0.5893   -0.0004   -0.0003
Custom: bf16 policy                                     10042   0.5228    0.5891   -0.0006   -0.0005
Clone: Full BF16                                        10042   0.5225    0.5888   -0.0009   -0.0008
...
----------------------------------------------------------------------------------------------------

Legend:
  Acc:       Unnormalized accuracy (argmax of sum of log-likelihoods)
  Acc_Norm:  Byte-normalized accuracy (argmax of byte-normalized log-likelihoods)
  Diff_Acc:  Difference from HuggingFace reference (Acc)
  Diff_Norm: Difference from HuggingFace reference (Acc_Norm)
  ★:         Best non-reference configuration

Precision Impact Analysis:
----------------------------------------------------------------------------------------------------
Clone: BF16+BF16+FP32_Softmax (Best)                    ✓ Nearly identical
Clone: Full FP32 (Max Precision)                        ✓ Nearly identical
Custom: bf16 policy                                     ✓ Very close
...
```

## 技術細節

### 為什麼需要這個？

1. **驗證精度對任務表現的影響**
   - 之前我們用 `scripts/sweep_precision_configs.py` 比較 logits 層級的差異
   - 現在我們測試這些差異是否影響實際的下游任務表現

2. **找出最佳的精度/效能平衡點**
   - BF16 更快但可能犧牲精度
   - FP32 更精確但更慢且用更多記憶體
   - 透過 HellaSwag 分數可以量化這個取捨

3. **驗證 Clone 和 Custom backend 的實作正確性**
   - 如果配置正確，兩者應該得到相似的分數
   - 大幅偏離表示可能有實作錯誤

### 關鍵發現（基於之前的實驗）

1. **Logits 層級的精度**（來自 `sweep_precision_configs.py`）：
   - 最佳配置：BF16 compute + BF16 RoPE + FP32 softmax
   - 與 HF 的 mean absolute difference: 0.026

2. **Backend 對齊**（來自 `compare_all_backends.py`）：
   - Clone backend: mean_abs = 0.028
   - Custom backend: mean_abs = 0.039（RoPE 修正後）

3. **預期的 HellaSwag 影響**：
   - 這些小的 logits 差異（< 0.04）應該不會顯著影響 HellaSwag 分數
   - 預期所有配置的 acc_norm 差異 < 0.01（約 100 題差異）

## Log 檔案位置

所有結果會儲存在：

```
awq/log/
├── precision_sweep/          # 完整掃描結果
│   ├── config1_*.log
│   ├── config2_*.log
│   └── ...
├── precision_sweep_quick/    # 快速測試結果
│   ├── hf_reference.log
│   ├── clone_*.log
│   └── custom_*.log
└── [其他 BFP/AWQ 實驗的 log]
```

## 下一步

完成 precision sweep 後，可以：

1. 比較結果與 logits 層級的差異
2. 決定生產環境要用哪個配置
3. 將相同的方法應用到其他任務（BoolQ, PIQA, etc.）
4. 測試量化模型（BFP/AWQ）在不同精度下的表現

## 相關文件

- `USAGE.md` - LlamaMyModel 完整使用指南
- `docs/backend_comparison.md` - Backend 比較文件
- `scripts/sweep_precision_configs.py` - Logits 層級的精度掃描
- `scripts/compare_all_backends.py` - Backend 比較工具
