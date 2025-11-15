# Project File Structure

This document reflects the current repository layout and the purpose of the main folders and files.

—

## Root

```
custom_llama/
├── FILE_STRUCTURE.md
├── SIMPLE_EVAL.md
├── requirements.txt
├── debug/
├── docs/
├── llama_backend/
├── model/
├── model_analysis/
├── quantize_model_script/
└── testbench/
```

—

## llama_backend/ — Core backend

Custom TinyLlama backend with precision policy and optional clone backend.

```
llama_backend/
├── USAGE.md
├── llama_my.py
├── precision_policy.py
├── utils.py
├── clone/
│   ├── clone_backend.py
│   ├── hf_clone.py
│   └── hf_rope.py
└── custom/               # (empty)
```

—

## debug/ — Debug and tests

Small scripts used to validate attention, precision policies, and ground-truth comparisons.

```
debug/
├── compare_qkv_attention.py
├── test_precision_policy.py
└── tinyllama_gt.py
```

—

## docs/ — Documentation

Notes and comparisons relevant to the backends and debugging.

```
docs/
├── backend_comparison.md
├── custom_vs_clone_code_comparison.md
├── debug_files_inventory.md
└── git_submodule_guide.md
```

—

## model/ — Models and variants

Base model and multiple AWQ-quantized variants (mix/fix precision across block sizes and magnitude settings).

```
model/
├── TinyLlama_1.1v/
├── TinyLlama_1.1v-awq-quantized/
├── TinyLlama_1.1v-awq-quantized-mix-precision-b{32,64,128}-m{2,3,4,5}/
└── TinyLlama_1.1v-awq-quantized-fix-precision-b{32,64,128}-m{2,3,4,5}/
```

—

## model_analysis/ — Plots and analysis

```
model_analysis/
├── plot_element_contribution/
└── plot_weight/
```

—

## quantize_model_script/ — Quantization scripts

AWQ and block quantization utilities.

```
quantize_model_script/
├── activation_aware_weight_quantization.py
├── block_quantization.py
├── fix_precision_awq_tinyllama.py
└── mix_precision_awq_tinyllama.py
```

—

## testbench/ — AWQ experiments and runs

Docs, run scripts, and evaluation helpers for AWQ and BFP runs.

```
testbench/
├── AWQ.md
├── AWQ_INTEGRATION_COMPLETE.txt
├── AWQ_MODELS_HELLASWAG.md
├── NEW_FILES_SUMMARY.md
├── PRECISION_SWEEP_SUMMARY.md
├── README_PRECISION_SWEEP.md
├── awq/
├── log/
├── run_all.sh
├── run_block_floating_point.sh
├── run_dynamic_awq_experiments.sh
├── run_precision_sweep_hellaswag.sh
├── run_precision_sweep_quick.sh
├── run_static_awq_evaluation.sh
├── run_static_awq_experiments.sh
├── test_awq_integration.sh
├── tinyllama_my_bfp_arc_c.py
├── tinyllama_my_bfp_arc_e.py
├── tinyllama_my_bfp_boolq.py
├── tinyllama_my_bfp_hellaswag.py
├── tinyllama_my_bfp_obqa.py
├── tinyllama_my_bfp_piqa.py
├── tinyllama_my_bfp_winogrande.py
└── validate_modifications.sh
```

—

## Top-level docs

- `SIMPLE_EVAL.md`: Simple evaluation guide.
- `requirements.txt`: Python dependencies.

—

Last Updated: November 15, 2025
