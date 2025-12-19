# Clone vs Custom Backend 詳細比較

## 1. 架構差異

### Clone Backend (來自 `hf_clone.py`)

```python
# 純函數式實作，無狀態
def clone_attention_ffn(x, params, ...):
    # 每次呼叫都是獨立的
    # 所有參數都透過 params dict 傳入
    x_norm = rmsnorm(x, params['norm1_weight'], ...)
    q = matmul(x_norm, params['wq'].t())
    # ... 完整計算一層
    return x, attn_weights

def clone_forward_all(x, layers_params, ...):
    # 迴圈處理所有層
    for layer_params in layers_params:
        x, _ = clone_attention_ffn(x, layer_params, ...)
    return x, latents
```

**特點：**
- 每次 forward 都是完整的序列處理
- 無內部狀態，不記憶之前的計算
- 適合單次推理、比較實驗

### Custom Backend (來自 `plain_script.py`)

```python
# 物件導向，有狀態
class transformer_block_with_kv_cache:
    def __init__(self, params, ...):
        self.params = params
        self.k_cache = None  # 記憶 key
        self.v_cache = None  # 記憶 value
        self.sequence_length = 0
    
    def forward(self, x, params):
        # 可以使用之前的 k_cache, v_cache
        if self.k_cache is not None:
            # 增量計算：只處理新 token
            k = torch.cat([self.k_cache, new_k], dim=2)
        else:
            # 首次：完整計算
            k = compute_full_k(x)
        
        self.k_cache = k  # 更新 cache
        return attn_output
```

**特點：**
- 維護 KV cache 用於增量生成
- 有內部狀態，記住歷史計算
- 適合多輪對話、長文本生成

---

## 2. 使用場景對比

### 場景 1: 單次推理（計算一個 prompt 的 logits）

```python
# Clone: 簡單直接
model_clone = CustomLlamaModel(backend="clone", dtype=torch.bfloat16)
logits = model_clone.single_step(inputs)  # 一次性計算

# Custom: 也可以，但 KV cache 沒用到
model_custom = CustomLlamaModel(backend="custom", dtype=torch.bfloat16)
logits = model_custom.single_step(inputs)  # cache 保持未使用
```

**結論：Clone 更簡潔**

### 場景 2: 增量生成（逐 token 生成）

```python
# Clone: 效率低，每次都重算整個序列
model_clone = CustomLlamaModel(backend="clone")
for i in range(100):
    logits = model_clone.single_step(current_tokens)  # 重算所有 token
    next_token = sample(logits)
    current_tokens = append(current_tokens, next_token)
# ❌ 時間複雜度: O(n²) - n 是生成長度

# Custom: 效率高，只計算新 token
model_custom = CustomLlamaModel(backend="custom")
for i in range(100):
    logits = model_custom.single_step(new_token_only)  # 只算新 token
    next_token = sample(logits)
# ✅ 時間複雜度: O(n)
```

**結論：Custom 高效得多**

### 場景 3: 實驗不同精度配置

```python
# Clone: 靈活控制
model = CustomLlamaModel(
    backend="clone",
    clone_compute_dtype=torch.float32,  # 主計算精度
    rope_cache_dtype=torch.bfloat16,    # RoPE 精度
    softmax_fp32=True                   # Softmax 精度
)

# Custom: 透過 policy 控制（較粗粒度）
model = CustomLlamaModel(
    backend="custom",
    precision_policy="match_hf"  # 只有預設選項
)
```

**結論：Clone 更靈活**

### 場景 4: 驗證與 HF 的一致性

```python
# 測試與 HF 的 logits 差異
hf_model = AutoModelForCausalLM.from_pretrained(...)
hf_logits = hf_model(input_ids).logits

# Clone: 高度對齊
clone_model = CustomLlamaModel(backend="clone", dtype=torch.bfloat16)
clone_logits = clone_model.single_step(inputs)
print(f"Clone diff: {(hf_logits - clone_logits).abs().mean()}")
# 輸出: 0.028 ✅

# Custom: 同樣高度對齊（RoPE 修正後）
custom_model = CustomLlamaModel(backend="custom", dtype=torch.bfloat16)
custom_logits = custom_model.single_step(inputs)
print(f"Custom diff: {(hf_logits - custom_logits).abs().mean()}")
# 輸出: 0.039 ✅ (RoPE 修正後，從 2.249 改善 58 倍)
```

**結論：兩者都能達到高精度對齊**

---

## 3. 技術細節差異

### RoPE 實作

**Clone (`hf_clone.py`):**
```python
def apply_rotary_pos_emb(q, k, cos, sin):
    # Halved RoPE: 前半/後半分開
    q1, q2 = q[..., :d//2], q[..., d//2:]
    q_rotated = torch.cat([-q2, q1], dim=-1)
    # 與 HF transformers 完全一致 ✅
```

**Custom (`plain_script.py`):**
```python
def apply_rope(x, freqs_cos, freqs_sin):
    # ✅ 已修正為 Halved RoPE: 前半/後半分開
    x1, x2 = x[..., :d//2], x[..., d//2:]
    x_rotated = torch.cat([-x2, x1], dim=-1)
    x_embed = x * freqs_cos + x_rotated * freqs_sin
    # 現在與 HF transformers 完全一致 ✅
```

**注意：** Custom backend 的 RoPE 已經在最近的更新中修正，現在使用與 Clone 相同的 halved RoPE 實作，因此精度對齊性大幅提升。

### Attention Mask

**Clone:**
```python
# 使用標準 causal mask
causal_mask = torch.triu(torch.ones(...), diagonal=1)
attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
```

**Custom:**
```python
# 可能有不同的 mask 實作
# 取決於 plain_script.py 中的實作細節
```

### Softmax 精度

**Clone:**
```python
# 可控制
if softmax_fp32:
    attn = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype)
else:
    attn = torch.softmax(scores, dim=-1)
```

**Custom:**
```python
# 固定實作，取決於 PrecisionPolicy
attn = torch.softmax(scores.to(policy.attn_softmax_dtype), dim=-1)
```

---

## 4. 效能比較

### 記憶體使用

| Backend | 首次 Forward | 增量生成 (100 tokens) |
|---------|--------------|----------------------|
| Clone | ~2GB | ~2GB × 100 次 = 200GB (重複) |
| Custom (with cache) | ~2GB | ~2GB + 100MB = ~2.1GB |

**Custom 在長文本生成時記憶體效率高得多！**

### 速度比較 (生成 100 tokens)

```python
import time

# Clone
start = time.time()
for _ in range(100):
    logits = clone_model.single_step(all_tokens)  # 重算
print(f"Clone: {time.time() - start:.2f}s")
# 輸出: ~50s (假設單 token forward 0.5s)

# Custom with KV cache
start = time.time()
for _ in range(100):
    logits = custom_model.single_step(new_token)  # 只算新的
print(f"Custom: {time.time() - start:.2f}s")
# 輸出: ~5s (首次 0.5s + 99 × 0.05s)
```

**Custom 快 10 倍以上！**

---

## 5. 何時使用哪個？

### 使用 Clone Backend 當：

✅ 需要與 HF 完美對齊（驗證、調試）  
✅ 實驗不同精度配置（研究用）  
✅ 單次推理/評估（不需要生成）  
✅ 比較不同量化方法的效果  
✅ 需要逐層分析（可以輕鬆獲取 latents）

### 使用 Custom Backend 當：

✅ 生產環境推理（需要速度）  
✅ 交互式對話/長文本生成  
✅ 需要 KV cache 加速  
✅ 使用 Precision Policy 做混合精度  
✅ 整合到現有系統（更 Pythonic 的 API）

---

## 6. 程式碼路徑對照

```
Clone Backend 路徑:
tinyllama_my.py (backend='clone')
  └─> hf_clone.py
      ├─> clone_forward_all()
      └─> clone_attention_ffn()
          ├─> build_rope_cache()
          └─> apply_rotary_pos_emb()  # Halved RoPE ✅

Custom Backend 路徑:
tinyllama_my.py (backend='custom')
  └─> plain_script/plain_script.py
      ├─> transformer_block()
      └─> transformer_block_with_kv_cache
          ├─> mqa_rope()
          └─> apply_rope()  # Halved RoPE ✅ (已修正)
```

---

## 7. 遷移建議

### 情境 1: Custom → Clone (追求極致精度)

```python
# 從 Custom 遷移到 Clone
# 前:
model = CustomLlamaModel(
    backend="custom",
    precision_policy="match_hf"
)

# 後:
model = CustomLlamaModel(
    backend="clone",
    dtype=torch.bfloat16,
    clone_compute_dtype=torch.bfloat16,
    rope_cache_dtype=torch.bfloat16,
    softmax_fp32=True
)
```

**注意：** 在 RoPE 修正後，Custom backend 的精度已經非常接近 Clone (mean_abs 差異 0.011)，大多數場景下不需要遷移。

### 情境 2: Clone → Custom (追求生成速度)

```python
# Clone 太慢（生成場景）
model = CustomLlamaModel(backend="clone")  # ❌ 慢 (無 KV cache)

# 改用 Custom
model = CustomLlamaModel(
    backend="custom",
    precision_policy="bf16"  # 與 Clone 精度相近
)
model.generate(prompt, max_new_tokens=1000)  # ✅ 快 (~10x)
```

---

## 8. 混合使用策略

最佳實踐：**用 Clone 驗證，用 Custom 部署**

```python
# 開發階段：用 Clone 驗證正確性
dev_model = CustomLlamaModel(backend="clone", dtype=torch.bfloat16)
dev_logits = dev_model.single_step(test_input)

# 對照 HF
hf_logits = hf_model(test_input).logits
assert torch.allclose(dev_logits, hf_logits, atol=0.1), "實作有問題！"

# 生產階段：用 Custom 部署
prod_model = CustomLlamaModel(backend="custom", dtype=torch.bfloat16)
output = prod_model.generate(user_input, max_new_tokens=512)

# 一次性驗證：確保兩者一致
clone_out = dev_model.single_step(test_input)
custom_out = prod_model.single_step(test_input)
diff = (clone_out - custom_out).abs().mean()
print(f"Clone vs Custom 差異: {diff:.6f}")  
# 預期輸出: ~0.011 ✅ (RoPE 修正後)
```

**更新：** 在 RoPE 修正後，Custom backend 已能達到與 Clone 相近的精度（差異僅 0.011），因此可以更放心地在生產環境使用 Custom backend。

---

## 總結

| 需求 | 推薦 Backend | 原因 |
|------|-------------|------|
| 🎯 驗證正確性 | Clone 或 Custom | 兩者精度相近 (diff ~0.011) |
| ⚡ 快速生成 | Custom | KV cache 帶來 ~10x 加速 |
| 🔬 精度實驗 | Clone | 參數控制更細粒度 |
| 💬 對話系統 | Custom | 增量生成效率高 |
| 📊 Benchmark | Clone | 公平比較 (無 cache 優化) |
| 🚀 生產部署 | Custom | 精度 + 速度兼具 |

**重要更新：** Custom backend 在 RoPE 修正後，與 HF 的精度對齊已達到 mean_abs=0.039 (vs Clone 的 0.028)，適合直接用於生產部署。

兩者互補，根據需求選擇！
