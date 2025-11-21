# Runtime Quantization Guide

## Overview

**問題**: 保存 48 個量化模型配置會佔用大量磁碟空間（每個 ~2GB with BF16 = ~96GB total）

**解決方案**: Runtime Quantization
- 只保存 1 個原始模型（BF16 格式，~2.2GB）
- 在推理時實時應用量化（幾秒鐘內完成）
- 無需保存量化後的模型

## 新的工作流程

### 舊方法（佔用大量磁碟）
```bash
# 量化並保存 48 個模型變體
python quantize_tinyllama_comprehensive_2d.py
# 結果：~96GB 磁碟空間（48 個模型 × 2GB）
```

### 新方法（節省磁碟空間）✅
```bash
# 1. 確保只有 1 個 BF16 原始模型
#    （通過設定 --save-dtype bfloat16，現在是默認值）

# 2. 使用 runtime quantization 進行推理/評估
python evaluate_with_runtime_quantization.py \
    --model tinyllama \
    --method bfp \
    --block-height 16 \
    --block-width 16 \
    --mantissa-bits 4

# 結果：只需要 ~2.2GB（1 個 BF16 模型）
```

## 使用方法

### 基本用法

```python
from transformers import AutoModelForCausalLM
from runtime_quantizer import apply_runtime_quantization

# 1. 載入原始模型（BF16）
model = AutoModelForCausalLM.from_pretrained(
    "model/tinyllama/TinyLlama_1.1v",
    torch_dtype=torch.bfloat16
)

# 2. 實時應用量化（幾秒鐘）
apply_runtime_quantization(
    model,
    method="bfp",
    block_height=16,
    block_width=16,
    mantissa_bits=4
)

# 3. 正常使用模型
outputs = model.generate(...)
```

### 使用評估腳本

```bash
# BFP quantization
python evaluate_with_runtime_quantization.py \
    --method bfp \
    --block-height 32 \
    --block-width 4 \
    --mantissa-bits 5

# AWQ Fix-Precision
python evaluate_with_runtime_quantization.py \
    --method awq-fix \
    --block-height 16 \
    --block-width 16 \
    --mantissa-bits 4

# AWQ Mix-Precision
python evaluate_with_runtime_quantization.py \
    --method awq-mix \
    --block-height 16 \
    --block-width 16 \
    --mantissa-bits 4 \
    --top-k 16
```

### 批量測試多個配置

```bash
#!/bin/bash
# test_all_configs.sh

BLOCK_SIZES=("128 1" "64 2" "32 4" "16 8" "8 16" "4 32" "2 64" "1 128")
MANTISSA_BITS=(5 4)

for block_size in "${BLOCK_SIZES[@]}"; do
    IFS=' ' read -r height width <<< "$block_size"
    for mantissa in "${MANTISSA_BITS[@]}"; do
        echo "Testing: ${height}x${width}, mantissa=${mantissa}" >&2
        
        python evaluate_with_runtime_quantization.py \
            --method bfp \
            --block-height $height \
            --block-width $width \
            --mantissa-bits $mantissa \
            > "results_${height}x${width}_m${mantissa}.log"
    done
done
```

## 磁碟空間對比

### 舊方法（保存所有量化模型）
```
model/
├── tinyllama/                          # 原始模型
│   └── TinyLlama_1.1v/                (~4.4GB FP32)
└── tinyllama-2d-comprehensive/         # 量化模型
    ├── bfp/
    │   ├── block_128x1_mantissa_5/    (~2.2GB BF16)
    │   ├── block_128x1_mantissa_4/    (~2.2GB BF16)
    │   ├── ... (14 more configs)
    ├── awq_fix/
    │   └── ... (16 configs)
    └── awq_mix/
        └── ... (16 configs)

總計：~4.4GB + ~105GB = ~109GB
```

### 新方法（Runtime Quantization）
```
model/
└── tinyllama/
    └── TinyLlama_1.1v/                (~2.2GB BF16)

總計：~2.2GB（節省 ~107GB！）
```

## 性能考慮

### 量化時間
- BFP: ~5-10 秒（無需 activation collection）
- AWQ Fix: ~15-20 秒（需要 activation collection）
- AWQ Mix: ~15-20 秒（需要 activation collection）

### 內存使用
- 原始模型（BF16）：~2.2GB
- 量化過程中：~2.2GB（原地修改）
- 總計：~2.2GB（vs FP32 的 ~4.4GB）

### 推理速度
- 量化不會加速推理（權重仍在 BF16 tensor 中）
- 若需加速，需實現 custom CUDA kernels（見 STORAGE_FORMAT.md）

## 完整工作流程範例

### 步驟 1：準備原始模型（一次性）
```bash
# 如果你的模型是 FP32，轉換為 BF16
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained('model/tinyllama/TinyLlama_1.1v')
tokenizer = AutoTokenizer.from_pretrained('model/tinyllama/TinyLlama_1.1v')

model = model.to(torch.bfloat16)
model.save_pretrained('model/tinyllama/TinyLlama_1.1v_bf16')
tokenizer.save_pretrained('model/tinyllama/TinyLlama_1.1v_bf16')
print('Converted to BF16!')
"
```

### 步驟 2：使用 Runtime Quantization 進行實驗
```python
# experiment.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime_quantizer import apply_runtime_quantization
import torch

# 載入 BF16 模型
model = AutoModelForCausalLM.from_pretrained(
    "model/tinyllama/TinyLlama_1.1v_bf16",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained("model/tinyllama/TinyLlama_1.1v_bf16")

# 測試不同配置
configs = [
    {"method": "bfp", "block_height": 32, "block_width": 4, "mantissa_bits": 5},
    {"method": "bfp", "block_height": 16, "block_width": 8, "mantissa_bits": 4},
    {"method": "bfp", "block_height": 8, "block_width": 16, "mantissa_bits": 4},
]

for config in configs:
    # 重新載入模型（確保乾淨狀態）
    model = AutoModelForCausalLM.from_pretrained(
        "model/tinyllama/TinyLlama_1.1v_bf16",
        torch_dtype=torch.bfloat16
    )
    
    # 應用量化
    apply_runtime_quantization(model, **config)
    
    # 運行評估
    # ... your evaluation code ...
    
    print(f"Completed: {config}")
```

### 步驟 3：整合到 lm-evaluation-harness
```python
# custom_task.py for lm-evaluation-harness
from lm_eval.api.model import LM
from transformers import AutoModelForCausalLM
from runtime_quantizer import apply_runtime_quantization

class QuantizedTinyLlama(LM):
    def __init__(self, model_path, quant_config):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16
        )
        
        # Apply runtime quantization
        apply_runtime_quantization(self.model, **quant_config)
        
        # ... rest of initialization
```

## 優點與限制

### 優點 ✅
- **節省磁碟空間**：只需 1 個原始模型（~2.2GB vs ~107GB）
- **快速實驗**：幾秒鐘即可測試新配置
- **靈活性**：輕鬆切換不同量化設置
- **簡單整合**：與現有評估流程兼容

### 限制 ⚠️
- **每次都需重新量化**：載入模型後需要應用量化（~10秒）
- **無推理加速**：權重仍在 BF16 tensor（需 custom kernels 才能加速）
- **內存中量化**：量化後的模型仍佔用相同內存
- **AWQ 需 activations**：AWQ 方法需要預先收集 activations

## 建議設置

### 研究/實驗
```bash
# 使用 runtime quantization
python evaluate_with_runtime_quantization.py --method bfp
```

### 生產部署
```bash
# 如果需要永久保存某個最佳配置
python quantize_tinyllama_comprehensive_2d.py \
    --method bfp \
    --save-dtype bfloat16
# 然後只保存該配置
```

### 論文實驗
```bash
# 測試所有配置但不保存
for config in configs;
    python evaluate_with_runtime_quantization.py $config
done
# 只保存結果 logs，不保存模型
```

## 相關文件
- `runtime_quantizer.py` - Runtime quantization 核心模組
- `evaluate_with_runtime_quantization.py` - 評估範例腳本
- `STORAGE_FORMAT.md` - 存儲格式詳細說明
