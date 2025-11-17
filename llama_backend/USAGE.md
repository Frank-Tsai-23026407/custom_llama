# TinyLlama 自定義模型使用指南

本文檔說明如何使用 `LlamaMyModel` 進行推理，包括精度控制、量化方法等高級功能。

## 目錄
- [環境要求與安裝](#環境要求與安裝)
- [基本使用](#基本使用)
- [Backend 選擇](#backend-選擇)
- [精度控制 (Precision Control)](#精度控制-precision-control)
- [Block Floating Point (BFP) 量化](#block-floating-point-bfp-量化)
- [Activation-Aware Weight Quantization (AWQ)](#activation-aware-weight-quantization-awq)
- [常見問題排除](#常見問題排除)
- [最佳實踐建議](#最佳實踐建議)

---

## 環境要求與安裝

### 必要套件版本

本專案需要以下核心套件：

- **PyTorch**: 2.3.1 (含 CUDA 12.1)
- **Triton**: 2.3.1
- **bitsandbytes**: 0.43.3
- **transformers**: 4.44.2
- **accelerate**: 0.34.2

### 安裝步驟

#### 方法 1: 使用 pip 安裝 CUDA wheels（推薦）

```bash
# 移除可能衝突的套件
pip uninstall -y torch torchvision torchaudio triton bitsandbytes

# 安裝 PyTorch CUDA 12.1 版本
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1

# 安裝其他相依套件
pip install -r requirements.txt
```

#### 方法 2: 使用 conda（替代方案）

```bash
# 透過 conda 安裝 PyTorch（包含 CUDA runtime）
conda install -y pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 移除可能衝突的套件
pip uninstall -y triton bitsandbytes

# 安裝其他相依套件
pip install -r requirements.txt
```

### 驗證安裝

```bash
python -c "import torch, bitsandbytes as bnb, triton; \
print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}'); \
print(f'bitsandbytes: {bnb.__version__}'); \
print(f'Triton: {triton.__version__}')"
```

預期輸出：
```
Torch: 2.3.1, CUDA: 12.1, Available: True
bitsandbytes: 0.43.3
Triton: 2.3.1
```

---

## 基本使用

### 最簡單的使用方式

```python
from llama_my import LlamaMyModel
import torch

# 創建模型實例
model = LlamaMyModel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda",  # 或 "cpu"
    dtype=torch.float32
)

# 準備輸入
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")

# 單步推理
logits = model.single_step(inputs)
print(f"Logits shape: {logits.shape}")

# 生成文本
from .llama_backend.utils import StopOnTokens
model.stop_criteria = StopOnTokens()
output = model.generate("Who is the president of US?", max_new_tokens=100)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

---

## Backend 選擇

`LlamaMyModel` 支援三種 backend，各有不同用途：

### 1. `backend="huggingface"` (HF 參考實作)

最準確、最穩定，直接使用 Hugging Face 的原生實作。適合作為 ground truth。

```python
model = LlamaMyModel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    backend="huggingface",
    device="cuda",
    dtype=torch.bfloat16
)
```

**使用場景：**
- 驗證其他 backend 的正確性
- 需要最高準確度的生產環境
- 不需要自定義修改

### 2. `backend="clone"` (功能性克隆)

逐層手動實作，與 HF 高度對齊（mean_abs < 0.03）。可完全控制精度、量化等細節。

```python
model = LlamaMyModel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    backend="clone",
    device="cuda",
    dtype=torch.bfloat16,
    clone_compute_dtype=torch.bfloat16,  # 計算精度
    rope_cache_dtype=torch.bfloat16,      # RoPE cache 精度
    softmax_fp32=True                     # softmax 是否用 fp32
)
```

**使用場景：**
- 實驗不同精度配置
- 研究數值行為
- 需要逐層控制（例如插入 profiling）

### 3. `backend="custom"` (自定義實作，預設)

使用 `plain_script` 中的自定義 transformer block，支援 KV cache、precision policy 等。

```python
model = LlamaMyModel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    backend="custom",
    device="cuda",
    dtype=torch.bfloat16,
    precision_policy="default"  # 或 "match_hf", "bf16"
)
```

**使用場景：**
- 使用 KV cache 進行增量生成
- 套用自定義 precision policy
- 與 BFP/AWQ 量化結合

---

## 精度控制 (Precision Control)

### 方法 1: 使用 Precision Policy (適用於 `custom` backend)

內建三種預設策略：

```python
# 策略 1: default - 保持張量原生 dtype
model = LlamaMyModel(
    backend="custom",
    dtype=torch.bfloat16,
    precision_policy="default"
)
# 配置：
# - attn_matmul: None (使用張量原生 dtype)
# - attn_softmax: None (使用張量原生 dtype)
# - ffn_matmul: None (使用張量原生 dtype)
# - rope_compute: bf16

# 策略 2: match_hf - 匹配 Hugging Face 行為（全 bf16）
model = LlamaMyModel(
    backend="custom",
    dtype=torch.bfloat16,
    precision_policy="match_hf"
)
# 配置：
# - attn_matmul: bf16 (顯式設定)
# - attn_softmax: bf16
# - ffn_matmul: bf16
# - rope_compute: bf16
# ⚠️ 注意：雖然配置是 bf16，但實際上與 HF 的混合精度策略
#    (bf16 計算 + fp32 softmax) 可能有細微差異

# 策略 3: bf16 - 全 bfloat16 端到端（與 match_hf 幾乎相同）
model = LlamaMyModel(
    backend="custom",
    dtype=torch.bfloat16,
    precision_policy="bf16"
)
# 配置：與 match_hf 完全相同
# - 所有運算都在 bf16
# - 最高效能
```

**三種策略的實際差異：**

| 策略 | attn_matmul | attn_softmax | ffn_matmul | rope | 說明 |
|------|-------------|--------------|------------|------|------|
| `default` | None* | None* | None* | bf16 | 保持張量原生 dtype，最靈活 |
| `match_hf` | bf16 | bf16 | bf16 | bf16 | 顯式設定所有運算為 bf16 |
| `bf16` | bf16 | bf16 | bf16 | bf16 | 與 match_hf 相同 |

\* `None` 表示使用張量的固有 dtype，如果權重是 bf16，計算也會是 bf16

**為什麼 `match_hf` 不完全匹配 HF？**

實際上 Hugging Face 的行為是：
- 大部分運算在 bf16
- **但 attention softmax 會臨時升到 fp32** 以提升數值穩定性
- 某些 LayerNorm 內部累加也用 fp32

我們的 `custom` backend 中，這些細節可能沒有完全複製。如果你想要**真正匹配 HF**，應該使用：

```python
# 🎯 真正匹配 HF 的方式
model = LlamaMyModel(
    backend="clone",  # 不是 custom！
    dtype=torch.bfloat16,
    clone_compute_dtype=torch.bfloat16,
    rope_cache_dtype=torch.bfloat16,
    softmax_fp32=True  # 這個才是關鍵！
)
# mean_abs=0.026 - 實測與 HF 高度對齊
```

### 方法 2: Clone Backend 精細控制

```python
from llama_my import LlamaMyModel
import torch

# 🏆 最佳配置（與 HF 完美對齊，mean_abs=0.026）
model = LlamaMyModel(
    backend="clone",
    dtype=torch.bfloat16,
    clone_compute_dtype=torch.bfloat16,  # 主計算精度
    rope_cache_dtype=torch.bfloat16,      # RoPE sin/cos cache 精度
    softmax_fp32=True                     # attention softmax 用 fp32
)

# 替代配置：更穩定但稍慢
model = LlamaMyModel(
    backend="clone",
    dtype=torch.bfloat16,
    clone_compute_dtype=torch.bfloat16,
    rope_cache_dtype=torch.float32,       # 更準確的三角函數
    softmax_fp32=True
)
# mean_abs=0.035 - 仍然很好
```

### 精度配置掃描

我們提供了腳本來找出最接近 HF 的配置：

```bash
python scripts/sweep_precision_configs.py
```

輸出示例：
```
Best config (closest to HF):
{'compute': 'bf16', 'rope': 'bf16', 'softmax_fp32': True}
mean_abs=0.025862, max_abs=0.250000
```

---

## Block Floating Point (BFP) 量化

BFP 量化將權重量化為共享指數的低精度格式，減少記憶體與計算成本。

### 基本使用

```python
model = LlamaMyModel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda",
    dtype=torch.bfloat16,
    apply_bfp=True,              # 啟用 BFP 量化
    bfp_block_size=16,           # Block 大小（通常 16, 32, 64）
    bfp_mantissa_bits=4          # 尾數位元數（2-7）
)
```

### BFP 參數說明

- **`bfp_block_size`**: 共享指數的元素數量
  - 較小（8-16）：更細粒度，更準確但壓縮率較低
  - 較大（32-64）：更高壓縮率但可能損失準確度
  
- **`bfp_mantissa_bits`**: 每個元素的尾數位元數
  - 2 bits: 極致壓縮（~4x），顯著精度損失
  - 4 bits: 平衡點（~2x），中等精度損失
  - 5-7 bits: 較小精度損失，壓縮率較低

### BFP 量化範圍

當 `apply_bfp=True` 時，以下層會被量化：
- Input/Output embeddings
- 所有層的 LayerNorm 權重
- 所有 attention 投影矩陣（Q/K/V/O）
- 所有 MLP 權重（gate/up/down）

### BFP 實驗腳本

```bash
# 運行完整 BFP 實驗（不同 block size 和 mantissa bits）
sh awq/run_block_floating_point.sh

# 單獨測試特定配置
python awq/task_script_hellaswag.py --block_size 16 --mantissa_bits 4
```

### BFP 效能/準確度權衡

| Config | 壓縮率 | HellaSwag Acc | 備註 |
|--------|--------|---------------|------|
| fp32 baseline | 1x | ~0.59 | 原始精度 |
| bf16 | ~2x | ~0.59 | 幾乎無損失 |
| BFP b=16 m=5 | ~2.5x | ~0.57 | 輕微損失 |
| BFP b=16 m=4 | ~3x | ~0.54 | 中等損失 |
| BFP b=16 m=3 | ~4x | ~0.48 | 明顯損失 |
| BFP b=16 m=2 | ~6x | ~0.35 | 顯著損失 |

---

## Activation-Aware Weight Quantization (AWQ)

AWQ 根據 activation 重要性選擇性量化權重，在 4-bit 量化下保持高準確度。

### 靜態 AWQ (預計算量化)

使用預先量化好的模型：

```bash
# 下載或生成 AWQ 量化模型
python quantize_model_script/quantize_awq.py \
    --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output_dir model/tinyllama/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m4 \
    --block_size 128 \
    --mantissa_bits 4

# 使用量化模型
python awq/task_script_hellaswag.py \
    --model_path model/tinyllama/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m4
```

### 動態 AWQ (運行時量化)

在推理時動態計算量化參數：

```bash
# 運行動態 AWQ 實驗
sh awq/run_mix-precision_awq_experiments.sh
```

### AWQ 與 BFP 的差異

| 特性 | BFP | AWQ |
|------|-----|-----|
| 量化目標 | 所有權重平等對待 | 基於 activation 重要性 |
| 精度 | 中等（4-5 bits 時） | 高（4 bits 時仍準確） |
| 校準數據 | 不需要 | 需要（用於計算重要性） |
| 計算開銷 | 低 | 中等（需額外 scaling） |
| 適用場景 | 快速壓縮 | 需保持準確度的極致壓縮 |

### AWQ 完整評估

```bash
# 運行所有 AWQ 配置的完整評估
sh awq/run_all.sh

# 查看結果
cat awq/log/fix-precision_awq_eval_m4_b128_*.log
```

---

## 最佳實踐建議

### 1. 選擇合適的 Backend

```python
# 📋 決策樹
if need_ground_truth or production:
    backend = "huggingface"
elif need_precision_control or research:
    backend = "clone"
elif need_kv_cache or custom_logic:
    backend = "custom"
```

### 2. 精度配置建議

```python
# 🎯 高準確度（與 HF 對齊，mean_abs < 0.04）
# 方式 1: Clone backend
model = LlamaMyModel(
    backend="clone",
    dtype=torch.bfloat16,
    clone_compute_dtype=torch.bfloat16,
    rope_cache_dtype=torch.bfloat16,
    softmax_fp32=True
)
# mean_abs=0.028

# 方式 2: Custom backend（也可以達到高精度！）
model = LlamaMyModel(
    backend="custom",
    dtype=torch.bfloat16,
    precision_policy="bf16"  # 或 "match_hf"（效果相同）
)
# mean_abs=0.039 - 與 clone 相當！

# ⚡ 高性能 + KV cache（生成時使用）
model = LlamaMyModel(
    backend="custom",  # custom 有 KV cache
    dtype=torch.bfloat16,
    precision_policy="bf16"
)
# 生成速度比 clone 快 10 倍以上

# 🔬 實驗/調試（完全 fp32）
model = LlamaMyModel(
    backend="clone",  # 或 "custom"
    dtype=torch.float32,
    clone_compute_dtype=torch.float32,
    rope_cache_dtype=torch.float32,
    softmax_fp32=True
)
```

### 3. 量化策略選擇

```python
# 📊 決策矩陣

# 情境 1: 需要最佳壓縮/準確度權衡
# → 使用 AWQ (4-bit, 靜態量化)
model_path = "model/tinyllama/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m4"

# 情境 2: 快速實驗，不想預處理
# → 使用 BFP (b=16, m=4 或 m=5)
model = LlamaMyModel(
    apply_bfp=True,
    bfp_block_size=16,
    bfp_mantissa_bits=5  # 或 4
)

# 情境 3: 記憶體極度受限
# → 使用 AWQ (4-bit) 或 BFP (m=3)
# AWQ 準確度更好

# 情境 4: 需要無損或近無損
# → 使用 bfloat16，不量化
model = LlamaMyModel(
    dtype=torch.bfloat16
)
```

### 4. 評估量化效果

```python
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

# 創建可評估的模型包裝
class TinyLlamaEvalWrapper(HFLM):
    def __init__(self, tinyllama_model):
        self.model = tinyllama_model.model
        self.tokenizer = tinyllama_model.tokenizer
        # ... (詳見 evaluation 腳本)

# 運行評估
results = evaluator.simple_evaluate(
    model=wrapper,
    tasks=["hellaswag", "arc_easy", "arc_challenge"],
    num_fewshot=0,
)
print(results)
```

### 5. 生成設置

```python
from .llama_backend.utils import StopOnTokens, input_formatting

model.stop_criteria = StopOnTokens()

# 生成文本
prompt = "Explain quantum computing in simple terms"
formatted_input = input_formatting([], prompt)
output_ids = model.generate(formatted_input, max_new_tokens=256)
text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)
```

---

## 常見問題

### Q: Clone backend 和 HF backend 結果為什麼有微小差異？

A: Clone backend 經過優化後與 HF 高度對齊（mean_abs < 0.03），差異主要來自：
- bfloat16 的量化誤差（固有）
- 不同的計算順序（影響極小）

使用 `backend="clone"` 配合 bf16 全套設置可達到最佳對齊。

### Q: BFP 和 AWQ 可以同時使用嗎？

A: 技術上可以，但不推薦。兩者都是權重量化方法，疊加使用會引入過多誤差。選擇其一即可。

### Q: 在 CPU 上使用 bfloat16 會怎樣？

A: 需要硬體支援（AVX512-BF16）。如果不支援，PyTorch 可能自動回退到 float32，效能無提升。建議 CPU 上使用 float32。

### Q: 如何確定最佳的 BFP 配置？

A: 運行掃描腳本：
```bash
sh awq/run_block_floating_point.sh
```
查看 `awq/log/bft_*.log` 比較不同配置的準確度。

---

## 常見問題排除

### 1. ModuleNotFoundError: No module named 'triton.ops'

**問題描述**：執行時出現 `ModuleNotFoundError: No module named 'triton.ops'`

**原因**：
- Triton 版本不相容（PyTorch 2.4.1 需要 Triton 3.0.0，但 bitsandbytes 0.43.3 需要 Triton 2.3.x）
- 舊版 Triton 未完整解除安裝

**解決方法**：
```bash
# 完整移除衝突套件
pip uninstall -y torch torchvision torchaudio triton bitsandbytes

# 重新安裝相容版本
pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install -r requirements.txt
```

### 2. bitsandbytes: libcudart.so not found

**問題描述**：
```
WARNING: No libcudart.so found! Install CUDA or the cudatoolkit package
The installed version of bitsandbytes was compiled without GPU support
```

**原因**：缺少 CUDA runtime library

**解決方法 A - 使用 conda（推薦）**：
```bash
conda install -y pytorch==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip uninstall -y triton bitsandbytes
pip install -r requirements.txt
```

**解決方法 B - 確認 CUDA 環境變數**：
```bash
# 檢查系統 CUDA
which nvcc
nvidia-smi

# 設定 LD_LIBRARY_PATH（若 CUDA 已安裝但找不到）
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### 3. CUDA version mismatch

**問題描述**：PyTorch 偵測到的 CUDA 版本與系統不符

**解決方法**：
```bash
# 檢查系統 CUDA 版本
nvcc --version
nvidia-smi  # 查看驅動支援的最高 CUDA 版本

# 重新安裝對應版本的 PyTorch wheels
# 例如：CUDA 11.8
pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

### 4. 精度測試失敗

**問題描述**：執行 `debug/test_precision_policy.py` 時輸出差異過大

**檢查步驟**：
```bash
# 1. 驗證環境
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# 2. 執行精度測試
python debug/test_precision_policy.py

# 3. 檢查預期行為
# - HF backend parity diff 應該 < 1e-6 (幾乎完全相同)
# - Clone backend diff 應該 < 0.03 (高度對齊)
# - match_hf policy 應該減少與 HF 的差異
```

### 5. OOM (Out of Memory)

**問題描述**：GPU 記憶體不足

**解決方法**：
```python
# 使用較低精度
model = LlamaMyModel(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype=torch.bfloat16,  # 或 torch.float16
    precision_policy="bf16"  # 所有計算都用 bf16
)

# 或使用量化模型
model = LlamaMyModel(
    model_name="path/to/quantized-model",
    quantized_weight_method="awq",  # 或 "bfp"
    device="cuda"
)
```

---

## 相關文檔

- [評估指南](EVAL.md) - 如何在標準 benchmark 上評估模型
- [預訓練指南](PRETRAIN.md) - 如何從頭訓練模型
- [量化概念](quantization_concepts.md) - 量化技術的詳細說明

---

## 引用

如果使用本程式碼，請引用：

```bibtex
@software{tinyllama_custom,
  title={TinyLlama Custom Implementation},
  author={Your Name},
  year={2025},
  url={https://github.com/Frank-Tsai-23026407/tiny_llama}
}
```

---

最後更新：2025-11-12
