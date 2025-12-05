# 稀疏化與量化優化設計計畫 (Sparsity & Quantization Optimization Plan)

本計畫旨在探討並實作結合 Block Floating Point (BFP) 量化與稀疏化 (Sparsity) 的優化策略，特別關注於解決低位元量化與高稀疏度同時存在時的挑戰。

## 1. 背景與挑戰

根據文獻與實務經驗，同時採用「低 bit + 高 sparsity」存在以下挑戰：

*   **空間收益門檻**：非結構化稀疏 (Unstructured Sparsity) 需要極高的稀疏度 (>50% 甚至 >90%) 才能在考慮 metadata 開銷後真正節省空間。
*   **精度-效率折衷**：在 4-6 bit 量化下，若無 retraining 或 activation-aware pruning，直接強制 50% 稀疏度會導致 perplexity 顯著惡化。
*   **硬體加速**：結構化 (Structured) 或半結構化 (Semi-structured, 如 2:4) 稀疏更容易獲得硬體加速。

## 2. 設計目標

本工作將基於現有的 fix-precision AWQ 與 BFP 實作，進一步整合：

1.  **Layer-wise Mixed Precision**：不同層採用不同精度 (如 bit5, bit8)。
2.  **Activation-Aware Sparsity**：利用 Wanda/BESA 等指標選擇重要權重，而非單純 magnitude pruning。
3.  **Structured/Semi-structured Patterns**：探索對硬體友善的稀疏模式。

## 3. 實驗規劃 (Roadmap)

我們將建立一個對照實驗表來驗證設計的有效性：

| 代號 | 實驗配置 | 說明 |
| :--- | :--- | :--- |
| **Baseline** | **bf16 dense** | 原始模型基準 (TinyLlama/LLaMA)。 |
| **A** | **BFP bit5 dense** | 現有的 5-bit BFP 實作 (已完成)。 |
| **B** | **BFP bit5 + Naive 50% Magnitude Pruning** | 作為對照組，展示單純 magnitude pruning 在低 bit 下的性能損失。 |
| **C** | **BFP bit5 + Activation-Aware 50% Pruning** | 引入類似 Wanda/BESA 的機制，利用 Activation 統計資訊來決定 Pruning Mask。 |
| **D** | **Mixed Precision + Selective Wanda Pruning** | 結合 Layer-wise 混合精度與選擇性稀疏化 (Selective Pruning)。具體策略為：敏感層 (如首尾層及中間的 `o_proj`, `gate_proj`) 採用 8-bit 精度且不進行稀疏化 (0% Sparsity)；其餘 Robust 層採用 5-bit 精度並應用 50% Wanda 稀疏化。此策略在保持約 29% 全局稀疏度的同時，顯著提升了模型精度。 |
| **Debug D** | **Validation Runs** | 為了驗證 Exp D 的複雜邏輯，進行了兩個小規模 (100 samples) 驗證：<br>1. **Mixed Precision (No Sparsity)**: 驗證混合精度邏輯。<br>2. **Mixed Precision + Selective Wanda**: 驗證選擇性稀疏化邏輯。 |

## 4. 實作步驟

### 第一階段：環境建置與基準重現
- [x] 建立工作目錄 `sparse-quantized-optimization`。
- [x] 遷移現有 `llama_my.py` (包含 BFP 實作) 與 `utils.py`。
- [x] 確認 Baseline (bf16) 與 Experiment A (BFP bit5 dense) 的 HellaSwag 數據。

### 第二階段：實作 Naive Magnitude Pruning (Exp B)
- [x] 修改 `LlamaMyModel` 或量化邏輯，加入 `force_sparsity` 參數。
- [x] 實作基於 weight magnitude 的 mask 生成邏輯 (bottom 50% set to 0)。
- [x] 評估 HellaSwag 性能。

### 第三階段：實作 Activation-Aware Pruning (Exp C)
- [x] 整合 AWQ 的 activation 統計資訊 (Input X 的 magnitude)。
- [x] 實作類似 Wanda 的 metric：`metric = |W| * ||X||`。
- [x] 根據 metric 生成 mask。
- [x] 評估 HellaSwag 性能。

### 第四階段：整合 Mixed Precision (Exp D)
- [x] 定義 Layer-wise 的精度配置 (`mixed_precision_policy.py`)。
- [x] 修改 `LlamaMyModel` 支援動態調整 BFP mantissa bits。
- [x] 實作選擇性稀疏化 (Selective Pruning)：跳過高精度層的剪枝。
- [x] 執行 Debug D 驗證邏輯正確性。
- [x] 評估 HellaSwag 性能。

## 5. 預期貢獻

- 將 Layer-wise mixed precision 具體實作於 BFP 架構。
- 結合 Activation-aware sparsity，在硬體友善的 block 格式中同時實現 bit reduction 與 sparsity allocation。
- 展示在 TinyLlama 上的 HellaSwag/Perplexity-Accuracy tradeoff 改善。
