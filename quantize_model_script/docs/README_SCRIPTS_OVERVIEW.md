# TinyLlama 2D Quantization Scripts - Quick Reference

這個目錄包含了完整的 TinyLlama 2D block quantization 工具集。

## ⚡ 推薦方法：Runtime Quantization（節省磁碟空間）

**新功能**：無需保存量化模型，在推理時實時量化！

```bash
# 只需保存 1 個 BF16 原始模型（~2.2GB）
# 在需要時實時應用量化（~10秒）

python evaluate_with_runtime_quantization.py \
    --method bfp \
    --block-height 16 \
    --block-width 16 \
    --mantissa-bits 4
```

**優點**：
- ✅ 磁碟空間：~2.2GB（vs ~107GB 保存所有配置）
- ✅ 快速測試不同配置（幾秒鐘）
- ✅ 與評估工具無縫整合

詳見：**`RUNTIME_QUANTIZATION.md`**

---

## 主要腳本

### 1. 綜合量化腳本（推薦）⭐

**`quantize_tinyllama_comprehensive_2d.py`**
- 支援所有量化方法：BFP、AWQ Fix-Precision、AWQ Mix-Precision
- 支援所有 block size 組合
- 一次執行生成所有配置

```bash
# 執行所有方法（48 configurations）
python quantize_tinyllama_comprehensive_2d.py

# 僅 BFP（16 configurations，最快）
python quantize_tinyllama_comprehensive_2d.py --method bfp

# 僅 AWQ Fix-Precision（16 configurations）
python quantize_tinyllama_comprehensive_2d.py --method awq-fix

# 僅 AWQ Mix-Precision（16 configurations）
python quantize_tinyllama_comprehensive_2d.py --method awq-mix
```

**便捷腳本：**
```bash
bash run_quantize_tinyllama_comprehensive.sh  # 所有方法
bash run_bfp_only.sh                          # 僅 BFP
bash run_awq_fix_only.sh                      # 僅 AWQ Fix
bash run_awq_mix_only.sh                      # 僅 AWQ Mix
```

### 2. BFP 掃描腳本

**`quantize_tinyllama_2d_sweep.py`**
- 僅支援標準 BFP quantization
- 較簡單的實現，適合學習

```bash
python quantize_tinyllama_2d_sweep.py
bash run_quantize_tinyllama_2d_sweep.sh
```

## 量化配置

### Block Sizes（所有腳本通用）
- 128×1, 64×2, 32×4, 16×8
- 8×16, 4×32, 2×64, 1×128

### Mantissa Bits
- 5 bits, 4 bits

### 總配置數
- **每個方法：** 16 configurations (8 block sizes × 2 mantissa bits)
- **所有方法：** 48 configurations (16 × 3 methods)

## 量化方法比較

| 方法 | 速度 | 準確度 | 需要校準數據 | 權重表示 | 適用場景 |
|------|------|--------|--------------|----------|----------|
| **BFP** | ⚡⚡⚡ 快 | ⭐⭐ 好 | ❌ 否 | 全 BFP | 快速實驗、基準測試 |
| **AWQ Fix** | ⚡⚡ 中等 | ⭐⭐⭐ 較好 | ✅ 是 | 全 BFP (縮放) | 生產部署、硬體友好 |
| **AWQ Mix** | ⚡ 慢 | ⭐⭐⭐⭐ 最好 | ✅ 是 | FP32 + BFP | 需要最高準確度 |

### 方法詳解

**BFP (Block Floating Point)**
- 標準的 2D block quantization
- 所有權重均勻量化為 BFP 格式
- 沒有使用激活信息

**AWQ Fix-Precision (縮放型)**
- 使用激活計算重要性分數 (scaling factors)
- 量化前先乘以分數，量化後再除回來
- **所有權重（包括重要權重）仍為 BFP 格式**
- 透過縮放讓重要權重獲得更好的量化精度
- 硬體友好，易於部署

**AWQ Mix-Precision (混合精度)**
- 識別每個 block 中最重要的 top-k 權重
- **重要權重保留為完整 FP32 格式（不量化）**
- 其餘權重使用 BFP 量化
- 準確度最高，但模型格式不統一，儲存稍大

## 快速開始

### 💡 新推薦：Runtime Quantization（無需保存模型）
```bash
# 評估單一配置
python evaluate_with_runtime_quantization.py \
    --method bfp \
    --block-height 16 \
    --block-width 16 \
    --mantissa-bits 4

# 測試多個配置（不保存模型）
for m in 4 5; do
  for b in "16 16" "32 4"; do
    python evaluate_with_runtime_quantization.py \
      --method bfp \
      --block-height ${b%% *} \
      --block-width ${b##* } \
      --mantissa-bits $m
  done
done
```
**磁碟使用**：~2.2GB（僅原始 BF16 模型）

---

### 選項 A：保存量化模型（如需永久保存）
```bash
bash run_bfp_only.sh --skip-generation-test
```
預計時間：~30-40 分鐘，16 個模型

### 選項 B：平衡方案（AWQ Fix）
```bash
bash run_awq_fix_only.sh
```
預計時間：~60-90 分鐘，16 個模型

### 選項 C：最高準確度（AWQ Mix）
```bash
bash run_awq_mix_only.sh --top-k 32
```
預計時間：~60-90 分鐘，16 個模型

### 選項 D：完整掃描（所有方法）
```bash
bash run_quantize_tinyllama_comprehensive.sh
```
預計時間：~2-4 小時，48 個模型

## 命令行參數

### 常用參數
```bash
--model tinyllama              # 模型路徑或預設
--method [bfp|awq-fix|awq-mix|all]  # 量化方法
--top-k 16                     # AWQ 保留的重要權重數量
--dry-run                      # 測試運行，不保存模型
--skip-generation-test         # 跳過生成測試（加速）
--output-dir /path/to/output   # 自定義輸出目錄
```

### AWQ 特定參數
```bash
--dataset Salesforce/wikitext          # 校準數據集
--dataset-config wikitext-103-raw-v1   # 數據集配置
--num-samples 128                       # 校準樣本數量
```

## 輸出結構

```
<output-base>/
├── bfp/                    # BFP 量化結果
│   ├── block_128x1_mantissa_5/
│   ├── block_128x1_mantissa_4/
│   ├── ...
│   └── block_1x128_mantissa_4/
├── awq_fix/                # AWQ Fix-Precision 結果
│   ├── block_128x1_mantissa_5/
│   ├── ...
│   └── block_1x128_mantissa_4/
└── awq_mix/                # AWQ Mix-Precision 結果
    ├── block_128x1_mantissa_5/
    ├── ...
    └── block_1x128_mantissa_4/
```

## 系統需求

### 記憶體
- **RAM：** 10-16 GB
- **磁碟空間：**
  - 單一方法（16 configs）：~16-32 GB
  - 所有方法（48 configs）：~50-100 GB

### 軟體依賴
```bash
pip install torch transformers datasets
```

## 核心量化函數

所有腳本都使用 `block_quantization_2d.py` 中的核心函數：

1. **`block_floating_point_quantize_2d`** - 標準 BFP 2D 量化
2. **`awq_quantize_2d`** - AWQ 2D 量化（支援 top-k 參數）

## 使用建議

### 研究/實驗
1. 先跑 BFP 建立基準
2. 選擇 1-2 個有代表性的 block size
3. 比較不同方法的效果

```bash
# 快速基準測試
python quantize_tinyllama_comprehensive_2d.py --method bfp --dry-run
```

### 生產部署
1. 收集代表性校準數據
2. 使用 AWQ Fix-Precision
3. 根據準確度需求調整 top-k

```bash
# 生產級量化
python quantize_tinyllama_comprehensive_2d.py \
    --method awq-fix \
    --top-k 32 \
    --num-samples 256
```

### 論文實驗
1. 執行所有配置進行完整比較
2. 保存所有結果用於分析

```bash
# 完整實驗
python quantize_tinyllama_comprehensive_2d.py --method all
```

## 詳細文檔

- **`README_COMPREHENSIVE_2D.md`** - 綜合腳本詳細說明
- **`README_TINYLLAMA_2D_SWEEP.md`** - BFP sweep 腳本說明
- **`../docs/2D_BLOCK_QUANTIZATION.md`** - 2D block quantization 技術文檔

## 示例工作流程

### 完整實驗流程
```bash
# 1. BFP baseline（快速）
bash run_bfp_only.sh --skip-generation-test

# 2. AWQ Fix（中等準確度）
bash run_awq_fix_only.sh --top-k 16

# 3. AWQ Mix（最高準確度）
bash run_awq_mix_only.sh --top-k 32

# 或一次執行所有
bash run_quantize_tinyllama_comprehensive.sh
```

### 快速測試單一配置
```bash
# 使用既有的單一配置腳本
python bfp_quantize_2d.py --model tinyllama --block-height 32 --block-width 4 --mantissa-bits 5
```

## 故障排除

### 問題：找不到模型
```bash
# 檢查模型路徑
ls -la ../model/tinyllama/TinyLlama_1.1v

# 或指定完整路徑
python quantize_tinyllama_comprehensive_2d.py --model /full/path/to/model
```

### 問題：記憶體不足
```bash
# 使用 dry-run 測試
python quantize_tinyllama_comprehensive_2d.py --dry-run --skip-generation-test

# 減少校準樣本
python quantize_tinyllama_comprehensive_2d.py --num-samples 64
```

### 問題：執行太慢
```bash
# 僅跑 BFP
bash run_bfp_only.sh --skip-generation-test

# 或選擇特定 block size（需修改腳本）
```

## 相關文件

### Python 腳本
- `quantize_tinyllama_comprehensive_2d.py` - 綜合量化腳本（主要）
- `quantize_tinyllama_2d_sweep.py` - BFP sweep 腳本
- `block_quantization_2d.py` - 核心量化函數
- `bfp_quantize_2d.py` - 單一 BFP 配置
- `awq_quantize_2d.py` - 單一 AWQ 配置

### Shell 腳本
- `run_quantize_tinyllama_comprehensive.sh` - 執行所有方法
- `run_bfp_only.sh` - 僅 BFP
- `run_awq_fix_only.sh` - 僅 AWQ Fix
- `run_awq_mix_only.sh` - 僅 AWQ Mix
- `run_quantize_tinyllama_2d_sweep.sh` - BFP sweep

### 文檔
- `README_COMPREHENSIVE_2D.md` - 詳細使用說明
- `README_TINYLLAMA_2D_SWEEP.md` - Sweep 腳本說明
- `README_2D_QUANTIZATION.md` - 2D 量化技術文檔
