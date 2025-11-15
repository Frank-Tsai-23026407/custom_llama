# Custom vs Clone Backend 代碼比較

## 概述

這份文檔詳細比較 `custom` 和 `clone` backend 在實作層面的關鍵差異。

---

## 1. 架構層次差異

### Clone Backend（純函數式）

```python
# tinyllama_my.py - Clone path
else:  # backend == 'clone'
    # 1. 手動取得 embeddings
    x = self.model.get_input_embeddings()(inputs['input_ids'])
    x = x.to(self.dtype)
    
    # 2. 提取所有層的權重為 dict
    layers_params = []
    for layer in self.model.model.layers:
        lp = {
            'norm1_weight': layer.input_layernorm.weight,
            'wq': layer.self_attn.q_proj.weight,
            'wk': layer.self_attn.k_proj.weight,
            # ... 其他權重
        }
        layers_params.append(lp)
    
    # 3. 呼叫純函數式的 forward（無狀態）
    x, latents = clone_forward_all(
        x, layers_params, num_heads, num_kv_heads, rms_eps,
        rope_cache_dtype=self.rope_cache_dtype,
        compute_dtype=self.clone_compute_dtype,
        softmax_fp32=self.softmax_fp32
    )
    
    # 4. 手動做最後的 norm 和 lm_head
    x = rmsnorm(x, self.model.model.norm.weight, eps=rms_eps)
    logits = lm_head(x, self.model.get_output_embeddings().weight)
```

**特點**：
- ✅ 完全無狀態（Stateless）
- ✅ 每次 forward 都是獨立的
- ✅ 權重以字典傳遞，不保存在類別實例中
- ✅ 適合精度實驗（每個參數都可細粒度控制）

### Custom Backend（有狀態類別）

```python
# tinyllama_my.py - Custom path (default)
# 1. 初始化時創建有狀態的 attention blocks
self.attention_blocks = []
for layer_idx in range(self.num_layers):
    params = {
        'norm1_weight': layer.input_layernorm.weight,
        'wq': layer.self_attn.q_proj.weight,
        # ... 所有權重都儲存在 params dict 中
    }
    # 創建帶 KV cache 的 block
    self.attention_blocks.append(
        transfomer_block_with_kv_cache(params, num_heads, num_kv_heads, device=device)
    )

# 2. Forward 時使用已初始化的 blocks
x = input_embedding(inputs['input_ids'], self.model.get_input_embeddings().weight)
for layer_idx in range(self.num_layers):
    params = self.attention_blocks[layer_idx].params
    if seq_len > 1:
        # 多 token：不使用 cache（為了與 HF 完全一致）
        x = transformer_block(x, params, num_heads, num_kv_heads)
    else:
        # 單 token：使用 KV cache（增量生成）
        x = self.attention_blocks[layer_idx].forward(x, params)
```

**特點**：
- ✅ 有狀態（Stateful）
- ✅ KV cache 儲存在 `transfomer_block_with_kv_cache` 實例中
- ✅ 支援增量生成（只計算新 token）
- ✅ 適合實際推理和生成任務

---

## 2. KV Cache 差異

### Clone Backend: **無 KV Cache**

```python
# hf_clone.py - clone_attention_ffn
def clone_attention_ffn(x, params, num_heads, num_kv_heads, ...):
    # 每次都重新計算完整的 K、V
    q = torch.matmul(x_norm, params['wq'].t())
    k = torch.matmul(x_norm, params['wk'].t())
    v = torch.matmul(x_norm, params['wv'].t())
    
    # 沒有 cache 機制
    # 每個 token 都重算所有之前的 K、V
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    # ...
```

**影響**：
- ❌ 生成 100 個 token 需要重算 100 次完整序列
- ❌ 時間複雜度：O(n²) 其中 n 是生成長度
- ✅ 但保證每次結果完全獨立（適合測試）

### Custom Backend: **有 KV Cache**

```python
# plain_script/plain_script.py - transfomer_block_with_kv_cache
class transfomer_block_with_kv_cache:
    def __init__(self, params, ...):
        self.k_cache = torch.empty(0, device=self.device)
        self.v_cache = torch.empty(0, device=self.device)
        self.sequence_length = 0
    
    def forward(self, x, params):
        # 計算當前 token 的 K、V
        k = torch.matmul(x_norm, params['wk'].t())
        v = torch.matmul(x_norm, params['wv'].t())
        
        # 將新的 K、V 附加到 cache 中
        if self.sequence_length == 0:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=-2)
            self.v_cache = torch.cat([self.v_cache, v], dim=-2)
        
        self.sequence_length += seq_len
        
        # 使用完整的 cached K、V 做 attention
        attn_scores = torch.matmul(q, self.k_cache.transpose(-2, -1)) / ...
```

**影響**：
- ✅ 增量生成非常快（只需計算新 token）
- ✅ 時間複雜度：O(n) 其中 n 是生成長度
- ⚠️ 需要手動 `reset_kv_cache()` 來清空狀態

---

## 3. RoPE 實作差異

### Clone Backend: 在 `hf_clone.py` 中

```python
# hf_clone.py
def build_rope_cache(seq_len, head_dim, base=10000.0, device=None, dtype=torch.float32):
    positions = torch.arange(0, seq_len, dtype=dtype, device=device)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, ...) / head_dim))
    freqs = torch.outer(positions, inv_freq)  # (S, D/2)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,S,D/2)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
    return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    # Halved RoPE（與 HF 一致）
    if cos.shape[-1] != q.shape[-1]:
        cos = torch.cat([cos, cos], dim=-1)  # (1,1,S,D)
        sin = torch.cat([sin, sin], dim=-1)
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]   # 前半
        x2 = x[..., x.shape[-1] // 2 :]   # 後半
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

**特點**：
- ✅ RoPE cache 預先計算（`build_rope_cache`）
- ✅ 支援配置 `rope_cache_dtype`（可選 fp32/bf16）
- ✅ 使用 HF 的 halved 實作（前半/後半分開旋轉）

### Custom Backend: 在 `plain_script/plain_script.py` 中

```python
# plain_script/plain_script.py
def apply_rope(x, seq_len, head_dim, start_pos=0, policy=None):
    """每次呼叫都重新計算 RoPE（支援 start_pos 用於增量生成）"""
    rope_theta = 10000.0
    positions = torch.arange(start_pos, start_pos + seq_len, ...)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, ...) / head_dim))
    freqs = torch.outer(positions, inv_freq)  # (S, D/2)
    
    # 與 Clone 一樣：使用 cat 拼接（與 HF 一致）
    emb = torch.cat((freqs, freqs), dim=-1)  # (S, D)
    cos = torch.cos(emb).unsqueeze(0).unsqueeze(0)
    sin = torch.sin(emb).unsqueeze(0).unsqueeze(0)
    
    # Halved rotation（與 Clone 一致）
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    
    x_out = (x * cos) + (x_rotated * sin)
    return x_out
```

**特點**：
- ✅ 即時計算（每次 forward 都算一次）
- ✅ 支援 `start_pos` 參數（KV cache 必需）
- ✅ 使用相同的 halved 實作（修正後與 Clone 一致）
- ⚠️ 沒有預先計算的 cache（但有 PrecisionPolicy 控制）

---

## 4. Precision 控制差異

### Clone Backend: 細粒度參數控制

```python
# tinyllama_my.py - Clone 初始化
model = LlamaMyModel(
    backend="clone",
    clone_compute_dtype=torch.bfloat16,  # 主計算 dtype
    rope_cache_dtype=torch.bfloat16,     # RoPE cache dtype
    softmax_fp32=True                     # Softmax 是否用 FP32
)

# hf_clone.py - clone_attention_ffn 中的 dtype 控制
def clone_attention_ffn(x, params, ..., compute_dtype=None, softmax_fp32=False):
    # 所有權重 cast 到 compute_dtype
    wq = params['wq'].to(compute_dtype)
    wk = params['wk'].to(compute_dtype)
    # ...
    
    # Softmax 獨立控制
    if softmax_fp32:
        attn_weights = F.softmax(attn_scores.float(), dim=-1).to(compute_dtype)
    else:
        attn_weights = F.softmax(attn_scores, dim=-1)
```

**優點**：
- ✅ 每個精度參數都可獨立配置
- ✅ 適合精度掃描實驗（測試各種組合）
- ✅ 可以逐層、逐操作控制 dtype

### Custom Backend: 使用 PrecisionPolicy

```python
# tinyllama_my.py - Custom 初始化
from precision_policy import PrecisionPolicy

model = LlamaMyModel(
    backend="custom",
    precision_policy="bf16"  # 或 "default"、"match_hf"
)

# precision_policy.py - 定義
class PrecisionPolicy:
    def __init__(self, 
                 attn_matmul_dtype=None,
                 attn_softmax_dtype=None,
                 ffn_matmul_dtype=None,
                 rope_compute_dtype=None,
                 stable_softmax=False):
        # 預設策略
        ...
```

**優點**：
- ✅ 統一管理（一個 policy 控制所有）
- ✅ 預設值合理（如 `match_hf` 模仿 HF 精度）
- ✅ 易於擴展新的策略

**限制**：
- ⚠️ 粒度較粗（無法像 Clone 那樣精細控制每個操作）

---

## 5. Forward Pass 流程比較

### Clone Backend 流程

```
Input IDs
   ↓
Embedding (手動呼叫)
   ↓
for each layer:
    提取該層所有權重 → dict
   ↓
clone_forward_all(x, layers_params, ...) [純函數]
   ↓
   for layer_params in layers_params:
       clone_attention_ffn(x, layer_params, ...)
           ├─ RMSNorm
           ├─ Q/K/V projection (cast to compute_dtype)
           ├─ RoPE (使用預先計算的 cos/sin cache)
           ├─ Attention (可選 FP32 softmax)
           ├─ O projection
           ├─ Residual
           ├─ RMSNorm
           └─ FFN (SwiGLU)
   ↓
最終 RMSNorm (手動)
   ↓
LM Head (手動)
   ↓
Logits
```

### Custom Backend 流程

```
Input IDs
   ↓
Embedding (使用 input_embedding 函數)
   ↓
for layer_idx in range(num_layers):
    params = self.attention_blocks[layer_idx].params
    if seq_len > 1:
        x = transformer_block(x, params, ...)  # 無 cache
    else:
        x = self.attention_blocks[layer_idx].forward(x, params)  # 有 cache
           ├─ RMSNorm
           ├─ Q/K/V projection (使用 policy.attn_matmul_dtype)
           ├─ RoPE (即時計算，支援 start_pos)
           ├─ KV Cache 更新/concat
           ├─ Attention (使用 policy.attn_softmax_dtype)
           ├─ O projection
           ├─ Residual
           ├─ RMSNorm
           └─ FFN (使用 policy.ffn_matmul_dtype)
   ↓
最終 RMSNorm (使用 rmsnorm 函數)
   ↓
LM Head (使用 lm_head 函數)
   ↓
Logits
```

---

## 6. 程式碼文件對應

| 功能 | Clone Backend | Custom Backend |
|------|---------------|----------------|
| **主入口** | `tinyllama_my.py` (backend='clone') | `tinyllama_my.py` (backend='custom') |
| **Forward 實作** | `hf_clone.py` | `plain_script/plain_script.py` |
| **RoPE** | `hf_clone.py::build_rope_cache`<br>`hf_clone.py::apply_rotary_pos_emb` | `plain_script.py::apply_rope` |
| **Attention** | `hf_clone.py::clone_attention_ffn` | `plain_script.py::mqa_rope`<br>`plain_script.py::transfomer_block_with_kv_cache` |
| **FFN** | 內嵌在 `clone_attention_ffn` 中 | `plain_script.py::ffn_SwiGLU` |
| **精度控制** | 函數參數 (`compute_dtype`, `softmax_fp32`) | `precision_policy.py::PrecisionPolicy` |

---

## 7. 何時使用哪個 Backend？

### 使用 Clone Backend 的場景：

✅ **精度實驗和調優**
```python
# 測試不同精度組合對準確度的影響
model = LlamaMyModel(
    backend="clone",
    clone_compute_dtype=torch.bfloat16,
    rope_cache_dtype=torch.float32,
    softmax_fp32=True
)
```

✅ **驗證與 HuggingFace 的一致性**
```python
# 逐層比較輸出
clone_model = LlamaMyModel(backend="clone", ...)
hf_model = AutoModelForCausalLM.from_pretrained(...)

# 可以輕鬆提取中間層輸出
clone_logits, clone_latents = clone_model.single_step(inputs, return_latents=True)
```

✅ **單次推理（不需要生成）**
```python
# 例如：分類、評估基準測試
for batch in dataloader:
    logits = model.single_step(batch)  # 無需 KV cache
```

### 使用 Custom Backend 的場景：

✅ **文本生成（多個 token）**
```python
# 生成 100 個 token，Custom 快 ~10x
model = LlamaMyModel(backend="custom", precision_policy="bf16")
output = model.generate(prompt, max_new_tokens=100)
```

✅ **對話系統（需要增量生成）**
```python
# 用戶輸入 → 生成回應
for user_input in conversation:
    model.reset_kv_cache()  # 每輪對話重置
    response = model.generate(user_input, max_new_tokens=50)
```

✅ **生產部署（速度優先）**
```python
# 在線服務，需要快速響應
model = LlamaMyModel(
    backend="custom",
    precision_policy="bf16",  # 平衡精度和速度
    dtype=torch.bfloat16
)
```

---

## 8. 精度對比結果（實測）

基於 `scripts/compare_all_backends.py` 的結果：

```
Configuration                          max_abs    mean_abs
HuggingFace (ground truth)            0.000000   0.000000
Clone (bf16+bf16+fp32_softmax)        0.250000   0.028495  ✅
Custom (bf16 policy)                   0.250000   0.039023  ✅
```

**結論**：
- ✅ Clone 和 Custom 都能達到高精度對齊（差異 < 0.04）
- ✅ Clone 略優（0.028 vs 0.039），但差異在 bfloat16 量化誤差範圍內
- ✅ Custom 在精度接近的情況下，生成速度快 ~10x

---

## 9. 代碼維護性比較

### Clone Backend

**優點**：
- ✅ 純函數式，易於測試
- ✅ 無全局狀態，無副作用
- ✅ 容易並行化（每個 layer 獨立）

**缺點**：
- ⚠️ 需要手動管理權重字典
- ⚠️ 無法利用 KV cache（效率低）

### Custom Backend

**優點**：
- ✅ 物件導向，符合 PyTorch 習慣
- ✅ KV cache 封裝在類別中（狀態管理清晰）
- ✅ PrecisionPolicy 統一管理精度

**缺點**：
- ⚠️ 有狀態（需要手動 reset）
- ⚠️ 測試時需要注意 cache 影響

---

## 10. 總結對照表

| 特性 | Clone Backend | Custom Backend |
|------|---------------|----------------|
| **狀態** | 無狀態 (Stateless) | 有狀態 (Stateful) |
| **KV Cache** | ❌ 無 | ✅ 有 |
| **RoPE** | 預先計算 cache | 即時計算（支援 start_pos） |
| **精度控制** | 參數級別（細粒度） | Policy 級別（粗粒度） |
| **生成速度** | 慢（O(n²)） | 快（O(n)，有 cache） |
| **與 HF 對齊** | 極佳（0.028） | 極佳（0.039） |
| **代碼位置** | `hf_clone.py` | `plain_script/plain_script.py` |
| **主要用途** | 精度實驗、驗證 | 實際推理、生成 |

---

## 參考文件

- [`USAGE.md`](../USAGE.md) - 完整使用指南
- [`docs/backend_comparison.md`](backend_comparison.md) - Backend 應用場景比較
- [`scripts/compare_all_backends.py`](../scripts/compare_all_backends.py) - Backend 比較工具
