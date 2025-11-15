# HellaSwag Precision Sweep

這個目錄包含用於在 HellaSwag 基準測試上比較不同精度配置的腳本。

## 檔案說明

### 執行腳本

1. **`run_precision_sweep_hellaswag.sh`** - 完整精度掃描
   - 測試 8 種不同的精度配置
   - 使用完整的 HellaSwag 驗證集（約 10,000 個樣本）
   - 執行時間：約 2-4 小時（取決於 GPU）

2. **`run_precision_sweep_quick.sh`** - 快速精度掃描
   - 測試 5 種主要精度配置
   - 只使用前 100 個樣本進行快速測試
   - 執行時間：約 5-10 分鐘

3. **`run_all.sh`** - 原始的 BFP/AWQ 實驗腳本
   - 測試不同的量化配置（Block Floating Point 和 AWQ）

### Python 檔案

1. **`tinyllama_my_bfp_hellaswag.py`** - 主要評估腳本
   - 支援多種 backend（huggingface, clone, custom）
   - 支援精度控制參數
   - 支援 BFP 量化選項

2. **`analyze_hellaswag_results.py`** - 結果分析腳本
   - 解析 log 檔案並生成比較表格
   - 計算與 HuggingFace 參考的差異
   - 標示最佳配置

## 使用方法

### 快速測試（推薦先用這個）

```bash
cd /home/frank23026407/TinyLlama
sh awq/run_precision_sweep_quick.sh
```

這會在 `awq/log/precision_sweep_quick/` 產生結果。

### 完整掃描

```bash
cd /home/frank23026407/TinyLlama
sh awq/run_precision_sweep_hellaswag.sh
```

結果會儲存在 `awq/log/precision_sweep/`。

### 分析結果

```bash
# 分析快速測試結果
python awq/analyze_hellaswag_results.py awq/log/precision_sweep_quick

# 分析完整掃描結果
python awq/analyze_hellaswag_results.py awq/log/precision_sweep
```

### 手動測試單一配置

```bash
# HuggingFace reference
python awq/tinyllama_my_bfp_hellaswag.py --backend huggingface

# Clone with best precision config
python awq/tinyllama_my_bfp_hellaswag.py \
    --backend clone \
    --compute_dtype bf16 \
    --rope_cache_dtype bf16 \
    --softmax_fp32

# Custom with bf16 policy
python awq/tinyllama_my_bfp_hellaswag.py \
    --backend custom \
    --precision_policy bf16

# 快速測試（只用 100 個樣本）
python awq/tinyllama_my_bfp_hellaswag.py \
    --backend clone \
    --compute_dtype bf16 \
    --rope_cache_dtype bf16 \
    --softmax_fp32 \
    --max_samples 100
```

## 精度配置說明

### Clone Backend 配置

Clone backend 支援細粒度的精度控制：

1. **Config 1: BF16 + BF16 + FP32 Softmax（推薦）**
   ```bash
   --backend clone --compute_dtype bf16 --rope_cache_dtype bf16 --softmax_fp32
   ```
   - 計算：BF16
   - RoPE cache：BF16
   - Attention softmax：FP32
   - 這是實驗中發現的最佳配置（與 HF 最接近）

2. **Config 2: Full BF16**
   ```bash
   --backend clone --compute_dtype bf16 --rope_cache_dtype bf16
   ```
   - 所有操作都用 BF16
   - 速度最快，但可能犧牲一點精度

3. **Config 5: Full FP32**
   ```bash
   --backend clone --compute_dtype fp32 --rope_cache_dtype fp32 --softmax_fp32
   ```
   - 所有操作都用 FP32
   - 最高精度，但速度較慢、記憶體用量大

### Custom Backend 配置

Custom backend 透過 precision policy 控制：

1. **bf16 policy（推薦）**
   ```bash
   --backend custom --precision_policy bf16
   ```
   - 所有操作用 BF16
   - 與 Clone 的最佳配置相近
   - 有 KV cache 加速生成

2. **default policy**
   ```bash
   --backend custom --precision_policy default
   ```
   - 使用預設的混合精度

## 預期結果

基於之前的實驗，預期的 acc_norm 精度：

- **HuggingFace Reference**: ~0.XXXX（ground truth）
- **Clone (BF16+BF16+FP32_Softmax)**: 與 HF 差異 < 0.001（幾乎相同）
- **Clone (Full FP32)**: 與 HF 差異 < 0.001
- **Custom (bf16)**: 與 HF 差異 < 0.002（非常接近）

## Log 檔案位置

- 完整掃描：`awq/log/precision_sweep/*.log`
- 快速測試：`awq/log/precision_sweep_quick/*.log`
- BFP/AWQ 實驗：`awq/log/*.log`

## 故障排除

### 記憶體不足

如果遇到 CUDA out of memory：

```bash
# 使用更小的 batch size 或減少樣本數
python awq/tinyllama_my_bfp_hellaswag.py \
    --backend clone \
    --compute_dtype bf16 \
    --rope_cache_dtype bf16 \
    --softmax_fp32 \
    --max_samples 50  # 更少的樣本
```

### 找不到模型

確保模型已下載：
```bash
# TinyLlama 應該會自動從 HuggingFace 下載
# 或檢查 model/ 目錄是否有本地副本
```

### 結果差異很大

如果某個配置的結果與 HuggingFace 差異很大（> 0.01）：
1. 檢查是否有 RoPE 實作錯誤
2. 檢查 dtype 轉換是否正確
3. 參考 `scripts/compare_all_backends.py` 進行 logits 層級的比較

## 相關文件

- `USAGE.md` - LlamaMyModel 的完整使用指南
- `docs/backend_comparison.md` - Clone vs Custom backend 的詳細比較
- `scripts/sweep_precision_configs.py` - Logits 層級的精度掃描
- `scripts/compare_all_backends.py` - Backend 比較工具
