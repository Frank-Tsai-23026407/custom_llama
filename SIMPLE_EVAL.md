# Simple Evaluation Guide

This guide shows how to run simple evaluations on TinyLlama models using two methods:
1. **lm-evaluation-harness** (standard framework)
2. **Custom evaluation scripts** (project-specific)

---

## Method 1: Using lm-evaluation-harness

### Installation

```bash
cd lm-evaluation-harness
pip install -e .
```

### Basic Usage

#### Single Task Evaluation

```bash
lm_eval --model hf \
    --model_args pretrained=TinyLlama/TinyLlama_v1.1 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

#### Multiple Tasks

```bash
lm_eval --model hf \
    --model_args pretrained=TinyLlama/TinyLlama_v1.1 \
    --tasks hellaswag,arc_easy,arc_challenge,winogrande,boolq \
    --device cuda:0 \
    --batch_size 8
```

#### Limit Number of Examples (Quick Test)

```bash
lm_eval --model hf \
    --model_args pretrained=TinyLlama/TinyLlama_v1.1 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --limit 100
```

### Evaluate Quantized Models

#### AWQ Quantized Models

```bash
# Mix-Precision AWQ (m5, b128)
lm_eval --model hf \
    --model_args pretrained=model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m5 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8

# Fix-Precision AWQ (m5, b128)
lm_eval --model hf \
    --model_args pretrained=model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m5 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8
```

### Save Results to File

```bash
lm_eval --model hf \
    --model_args pretrained=TinyLlama/TinyLlama_v1.1 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --output_path results/ \
    --log_samples
```

### Available Tasks

Common tasks for language model evaluation:
- `hellaswag`: Commonsense reasoning
- `arc_easy`, `arc_challenge`: Science question answering
- `winogrande`: Pronoun resolution
- `boolq`: Boolean question answering
- `piqa`: Physical commonsense reasoning
- `openbookqa`: Open book question answering

List all available tasks:
```bash
lm_eval --tasks list
```

---

## Method 2: Using Custom Evaluation Scripts

### Overview

Custom scripts in `awq/` directory provide precision-aware evaluation with custom backends.

### Single Model Evaluation

#### Basic HellaSwag Evaluation

```bash
python awq/tinyllama_my_bfp_hellaswag.py \
    --model_path model/TinyLlama_1.1v
```

#### With Specific Precision Policy

```bash
python awq/tinyllama_my_bfp_hellaswag.py \
    --model_path model/TinyLlama_1.1v \
    --precision bf16
```

Available precision policies:
- `default`: Use tensor intrinsic dtypes
- `bf16`: Force all operations to bfloat16
- `match_hf`: Match HuggingFace behavior

#### Limit Examples (Quick Test)

```bash
python awq/tinyllama_my_bfp_hellaswag.py \
    --model_path model/TinyLlama_1.1v \
    --limit 100
```

### Evaluate AWQ Quantized Models

#### Mix-Precision AWQ Models

```bash
# m2 (2-bit mantissa)
python awq/tinyllama_my_bfp_hellaswag.py \
    --model_path model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m2

# m5 (5-bit mantissa)
python awq/tinyllama_my_bfp_hellaswag.py \
    --model_path model/TinyLlama_1.1v-awq-quantized-mix-precision-b128-m5
```

#### Fix-Precision AWQ Models

```bash
# m2 (2-bit mantissa)
python awq/tinyllama_my_bfp_hellaswag.py \
    --model_path model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m2

# m5 (5-bit mantissa)
python awq/tinyllama_my_bfp_hellaswag.py \
    --model_path model/TinyLlama_1.1v-awq-quantized-fix-precision-b128-m5
```

### Other Tasks

#### BoolQ

```bash
python awq/tinyllama_my_bfp_boolq.py \
    --model_path model/TinyLlama_1.1v
```

#### ARC-Easy

```bash
python awq/tinyllama_my_bfp_arc_e.py \
    --model_path model/TinyLlama_1.1v
```

#### ARC-Challenge

```bash
python awq/tinyllama_my_bfp_arc_c.py \
    --model_path model/TinyLlama_1.1v
```

#### PIQA

```bash
python awq/tinyllama_my_bfp_piqa.py \
    --model_path model/TinyLlama_1.1v
```

#### Winogrande

```bash
python awq/tinyllama_my_bfp_winogrande.py \
    --model_path model/TinyLlama_1.1v
```

#### OpenBookQA

```bash
python awq/tinyllama_my_bfp_obqa.py \
    --model_path model/TinyLlama_1.1v
```

### Batch Evaluation Scripts

#### Quick Precision Sweep (100 samples)

```bash
cd awq
bash run_precision_sweep_quick.sh
```

This evaluates multiple precision configurations on a small subset:
- 8 precision configs
- AWQ Mix-Precision/Fix-Precision models (b128, m2-m5)
- ~100 samples per task
- Results in `awq/awq/log/precision_sweep_quick/`

#### Full Precision Sweep (HellaSwag)

```bash
cd awq
bash run_precision_sweep_hellaswag.sh
```

This evaluates all precision configurations on full HellaSwag:
- 8 precision configs
- 16 AWQ models (Mix/Fix-Precision, b32/b128, m2-m5)
- ~10,000 samples
- Results in `awq/log/precision_sweep/`

#### Integration Test

```bash
cd awq
bash test_awq_integration.sh
```

Quick test of 2 AWQ models (mix-precision-m2 + fix-precision-m2) with 5 samples each.

---

## Comparison: lm-eval vs Custom

### lm-evaluation-harness

**Pros:**
- Standard framework, widely used
- Many tasks available out-of-box
- Consistent evaluation protocol
- Easy result comparison with other models

**Cons:**
- Less control over precision/dtype
- Harder to customize evaluation flow
- May not support custom quantization directly

### Custom Scripts

**Pros:**
- Full control over precision policies
- Custom backend support (BFP, AWQ)
- Easier to debug and modify
- Project-specific optimizations

**Cons:**
- Need to implement each task
- Less standardized
- May differ from official benchmarks

---

## Quick Start Examples

### Example 1: Quick Test with lm-eval

```bash
# Test on 100 samples
lm_eval --model hf \
    --model_args pretrained=TinyLlama/TinyLlama_v1.1 \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size 8 \
    --limit 100
```

### Example 2: Quick Test with Custom Script

```bash
# Test on 100 samples
python awq/tinyllama_my_bfp_hellaswag.py \
    --model_path model/TinyLlama_1.1v \
    --limit 100
```

### Example 3: Compare AWQ Models

```bash
# Using lm-eval
for model in mix-precision fix-precision; do
    lm_eval --model hf \
        --model_args pretrained=model/TinyLlama_1.1v-awq-quantized-${model}-b128-m5 \
        --tasks hellaswag \
        --device cuda:0 \
        --batch_size 8 \
        --output_path results/${model}/
done
```

```bash
# Using custom scripts
for model in mix-precision fix-precision; do
    python awq/tinyllama_my_bfp_hellaswag.py \
        --model_path model/TinyLlama_1.1v-awq-quantized-${model}-b128-m5 \
        | tee results/${model}_hellaswag.log
done
```

---

## Tips

1. **Start small**: Use `--limit 100` or quick sweep scripts first
2. **Monitor GPU**: Use `nvidia-smi` to check memory usage
3. **Batch size**: Adjust based on GPU memory (typical: 4-16)
4. **Save logs**: Always redirect output to files for later analysis
5. **Compare carefully**: Ensure same number of samples when comparing

---

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Use smaller model or quantized version
- Clear cache: `torch.cuda.empty_cache()`

### Slow Evaluation
- Increase `--batch_size` (if memory allows)
- Use `--limit` for quick tests
- Check if model is on GPU

### Different Results
- Verify same number of samples
- Check precision/dtype settings
- Ensure same random seed if applicable

---

## Related Documentation

- [EVAL.md](EVAL.md): Comprehensive evaluation documentation
- [awq/AWQ_MODELS_HELLASWAG.md](awq/AWQ_MODELS_HELLASWAG.md): AWQ model evaluation details
- [awq/NEW_FILES_SUMMARY.md](awq/NEW_FILES_SUMMARY.md): Overview of evaluation scripts
- [lm_evaluation_script.md](lm_evaluation_script.md): Technical evaluation details
