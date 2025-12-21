# Mixed-Precision AWQ (BFP-based)

此目錄實作了結合 **AWQ (Activation-aware Weight Quantization)** 與 **Mixed-Precision Protection** 的混合精度量化流程。

## 核心機制

### 1. 顯著性計算 (Saliency Calculation)
權重的顯著性（Saliency）決定了哪些權重對模型輸出影響較大。本實作根據以下公式計算顯著性分數：
- **公式**：`Saliency = mean(|X|) * |W|`
- **邏輯**：將輸入活化值 $X$ 的平均絕對值（按輸入通道維度）與權重矩陣 $W$ 的絕對值做逐元素相乘。這能捕捉到那些接收「大活化值」且本身「量級大」的關鍵權重元件。

### 2. 區塊化 Top-K 保護 (Block-wise Top-K Protection)
為了在進行 BFP (Block Floating Point) 量化時最小化誤差，我們引入了 **Block-wise** 的保護機制：
- **範疇**：保護是在每個 **2D BFP Block**（如 8x8, 16x4 等）內部獨立進行的，而非全局或全通道。
- **動作**：
    1. 在每個區塊中，根據顯著性分數找出前 $K$ 個最敏感的權重（$K = \text{block\_size} \times \text{top\_k\_ratio}$）。
    2. 這些敏感權重將**保持原始高精度 (FP16/BF16)**，不參與量化。
    3. 在計算區塊的共用指數（Shared Exponent）前，先將敏感權重設為 0，確保剩餘的權重能分配到更精確的位元解析度。

## 檔案說明
- `mix_precision_awq.py`: 混合精度量化與評估的主程式。
- `run_eval_awq_tinyllama.sh`: TinyLlama 的自動化評估腳本。

## 使用方式
```bash
python mix_precision_awq.py \
    --model "TinyLlama/TinyLlama_v1.1" \
    --block-height 8 \
    --block-width 8 \
    --mantissa-bits 4 \
    --top-k-ratio 0.01 \
    --eval-ppl
```
