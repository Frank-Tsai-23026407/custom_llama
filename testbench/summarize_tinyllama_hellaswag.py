#!/usr/bin/env python3
import argparse
import csv
import os
import re
from collections import defaultdict
from glob import glob
from typing import Dict, Optional

# Patterns to capture accuracy lines from HellaSwag logs
# Examples:
#   "Accuracy (acc): 0.4480"
#   "Accuracy (acc_norm): 0.6027"
ACCURACY_RE = re.compile(
    r"^Accuracy\s*\((?P<metric>[^\)]+)\)\s*:\s*(?P<value>[0-9]*\.?[0-9]+)\s*$",
    re.IGNORECASE
)


def parse_log_for_accuracies(path: str) -> Dict[str, float]:
    """Parse a log file and return a mapping metric->value."""
    results: Dict[str, float] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = ACCURACY_RE.match(line.strip())
                if not m:
                    continue
                metric = m.group("metric").strip().lower()
                try:
                    val = float(m.group("value"))
                except ValueError:
                    continue
                results[metric] = val
    except FileNotFoundError:
        pass
    return results


def latest_log_for_task(log_dir: str, task: str = "hellaswag") -> Optional[str]:
    """Return the latest log file path for a given task within a model's log dir."""
    candidates = glob(os.path.join(log_dir, f"{task}.*.log"))
    if not candidates:
        return None
    # Choose latest by modification time
    return max(candidates, key=os.path.getmtime)


def parse_model_name(dir_name: str) -> tuple:
    """Parse directory name to extract model configuration.
    
    Examples:
        'tinyllama_TinyLlama_1.1v-2d-comprehensive_awq_fix_block_128x1_mantissa_4'
        -> ('awq_fix', '128x1', '4')
        
        'bfp_runtime_128x1_m4'
        -> ('bfp_runtime', '128x1', '4')
    """
    if dir_name.startswith("bfp_runtime_"):
        # Runtime BFP format: bfp_runtime_128x1_m4
        match = re.match(r"bfp_runtime_(\d+x\d+)_m(\d+)", dir_name)
        if match:
            return ("bfp_runtime", match.group(1), match.group(2))
    else:
        # Static model format: tinyllama_TinyLlama_1.1v-2d-comprehensive_awq_fix_block_128x1_mantissa_4
        match = re.search(r"(awq_fix|awq_mix|bfp)_block_(\d+x\d+)_mantissa_(\d+)", dir_name)
        if match:
            return (match.group(1), match.group(2), match.group(3))
    
    return (dir_name, "", "")


def discover_model_log_dirs(base_dir: str):
    """Yield (model_name, model_log_dir) for each model folder under base_dir."""
    if not os.path.isdir(base_dir):
        return
    
    # Process static model directories
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p) and name != "runtime_bfp_2d":
            yield name, p
    
    # Process runtime_bfp_2d subdirectories
    runtime_dir = os.path.join(base_dir, "runtime_bfp_2d")
    if os.path.isdir(runtime_dir):
        for name in sorted(os.listdir(runtime_dir)):
            p = os.path.join(runtime_dir, name)
            if os.path.isdir(p):
                yield name, p


def build_summary(base_dir: str):
    """Build a nested mapping: model_type -> block_shape -> mantissa -> {metric: value}"""
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for model_name, model_dir in discover_model_log_dirs(base_dir):
        log_path = latest_log_for_task(model_dir, "hellaswag")
        if not log_path:
            continue
        
        metrics = parse_log_for_accuracies(log_path)
        if not metrics:
            continue
        
        model_type, block_shape, mantissa = parse_model_name(model_name)
        summary[model_type][block_shape][mantissa] = metrics

    return summary


def write_csv(summary, out_csv: str):
    """Write summary to CSV file."""
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_type", "block_shape", "mantissa", "acc", "acc_norm"])
        
        for model_type in sorted(summary.keys()):
            for block_shape in sorted(summary[model_type].keys()):
                for mantissa in sorted(summary[model_type][block_shape].keys()):
                    metrics = summary[model_type][block_shape][mantissa]
                    acc = metrics.get("acc", "")
                    acc_norm = metrics.get("acc_norm", "")
                    writer.writerow([
                        model_type,
                        block_shape,
                        mantissa,
                        f"{acc:.4f}" if isinstance(acc, float) else "",
                        f"{acc_norm:.4f}" if isinstance(acc_norm, float) else ""
                    ])


def write_markdown(summary, out_md: str):
    """Write summary to Markdown table."""
    os.makedirs(os.path.dirname(out_md) if os.path.dirname(out_md) else ".", exist_ok=True)
    
    lines = []
    lines.append("# TinyLlama HellaSwag Evaluation Summary\n")
    
    for model_type in sorted(summary.keys()):
        # Collect all rows for the current model_type to sort them later
        table_rows = []
        for block_shape in sorted(summary[model_type].keys()):
            # Safe sort key for mantissa
            def mantissa_key(m):
                try:
                    return int(m)
                except ValueError:
                    return 0

            for mantissa in sorted(summary[model_type][block_shape].keys(), key=mantissa_key):
                metrics = summary[model_type][block_shape][mantissa]
                acc = metrics.get("acc", 0.0)
                acc_norm = metrics.get("acc_norm", 0.0)
                table_rows.append((block_shape, mantissa, acc, acc_norm))

        # Sort rows by acc_norm in descending order
        table_rows.sort(key=lambda row: row[3], reverse=True)

        # Append sorted rows to the markdown output
        lines.append(f"## {model_type}\n")
        lines.append("| Block Shape | Mantissa | Accuracy (acc) | Accuracy (acc_norm) |")
        lines.append("| --- | --- | --- | --- |")
        
        for block_shape, mantissa, acc, acc_norm in table_rows:
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else ""
            acc_norm_str = f"{acc_norm:.4f}" if isinstance(acc_norm, float) else ""
            lines.append(f"| {block_shape} | {mantissa} | {acc_str} | {acc_norm_str} |")
        lines.append("")
    
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Summarize accuracies from TinyLlama HellaSwag logs."
    )
    parser.add_argument(
        "--base_dir",
        default="testbench/log/tinyllama_hellaswag_only",
        help="Base log directory"
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="CSV output path (default: <base_dir>/summary.csv)"
    )
    parser.add_argument(
        "--out_md",
        default=None,
        help="Markdown output path (default: <base_dir>/summary.md)"
    )
    args = parser.parse_args()

    out_csv = args.out_csv or os.path.join(args.base_dir, "summary.csv")
    out_md = args.out_md or os.path.join(args.base_dir, "summary.md")

    summary = build_summary(args.base_dir)
    if not summary:
        print(f"No accuracies found under {args.base_dir}")
        return

    write_csv(summary, out_csv)
    write_markdown(summary, out_md)
    print(f"Wrote: {out_csv}\nWrote: {out_md}")


if __name__ == "__main__":
    main()
