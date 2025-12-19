#!/usr/bin/env python3
import argparse
import csv
import os
import re
from collections import defaultdict
from glob import glob
from typing import Dict, Tuple, Optional

# Known task labels used by run scripts
KNOWN_TASKS = [
    "hellaswag",
    "arc_challenge",
    "arc_easy",
    "boolq",
    "obqa",
    "piqa",
    "winogrande",
]

# Patterns to capture accuracy-like lines.
# Examples handled:
#   "Accuracy: 0.5123"
#   "Accuracy (acc): 0.5123"
#   "Accuracy (acc_norm): 0.4988"
#   "Correct predictions (acc): 100"  (ignored, we want the Accuracy line)
ACCURACY_RE = re.compile(r"^Accuracy\s*(?:\((?P<metric>[^\)]+)\))?\s*:\s*(?P<value>[0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)


def parse_log_for_accuracies(path: str) -> Dict[str, float]:
    """Parse a log file and return a mapping metric->value.
    If a metric is unspecified (like just 'Accuracy:'), store under 'acc'.
    """
    results: Dict[str, float] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = ACCURACY_RE.match(line.strip())
                if not m:
                    continue
                metric = (m.group("metric") or "acc").strip().lower()
                try:
                    val = float(m.group("value"))
                except ValueError:
                    continue
                results[metric] = val
    except FileNotFoundError:
        pass
    return results


def latest_log_for_task(log_dir: str, task: str) -> Optional[str]:
    """Return the latest log file path for a given task within a model's log dir."""
    candidates = glob(os.path.join(log_dir, f"{task}.*.log"))
    if not candidates:
        return None
    # Choose latest by modification time
    return max(candidates, key=os.path.getmtime)


def discover_model_log_dirs(base_dir: str):
    """Yield (model_name, model_log_dir) for each model folder under base_dir."""
    if not os.path.isdir(base_dir):
        return
    for name in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            yield name, p


def build_summary(base_dir: str):
    """Build a nested mapping: model -> task -> {metric: value}"""
    summary: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for model_name, model_dir in discover_model_log_dirs(base_dir):
        for task in KNOWN_TASKS:
            log_path = latest_log_for_task(model_dir, task)
            if not log_path:
                continue
            metrics = parse_log_for_accuracies(log_path)
            if metrics:
                summary[model_name][task].update(metrics)

    return summary


def collect_all_columns(summary: Dict[str, Dict[str, Dict[str, float]]]):
    """Collect a sorted list of columns for CSV/MD output.
    Columns pattern: model, then for each task, task_acc and task_acc_norm if present anywhere.
    """
    columns = ["model"]
    # Determine which metrics appear across models per task
    task_metrics: Dict[str, set] = {t: set() for t in KNOWN_TASKS}
    for _, tasks in summary.items():
        for task, metrics in tasks.items():
            for m in metrics.keys():
                task_metrics.setdefault(task, set()).add(m)

    for task in KNOWN_TASKS:
        metrics = task_metrics.get(task, set())
        # Always prefer to show 'acc' if present, then 'acc_norm' if present
        ordered = []
        if "acc" in metrics:
            ordered.append("acc")
        if "acc_norm" in metrics:
            ordered.append("acc_norm")
        # Include any other unexpected metric keys deterministically
        for m in sorted(metrics - set(ordered)):
            ordered.append(m)
        for m in ordered:
            columns.append(f"{task}:{m}")
    return columns


def write_csv(summary, out_csv: str):
    columns = collect_all_columns(summary)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for model in sorted(summary.keys()):
            row = [model]
            for col in columns[1:]:
                task, metric = col.split(":", 1)
                val = summary.get(model, {}).get(task, {}).get(metric)
                row.append(f"{val:.4f}" if isinstance(val, float) else "")
            writer.writerow(row)


def write_markdown(summary, out_md: str):
    columns = collect_all_columns(summary)
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    lines = []
    # Header
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    # Rows
    for model in sorted(summary.keys()):
        cells = [model]
        for col in columns[1:]:
            task, metric = col.split(":", 1)
            val = summary.get(model, {}).get(task, {}).get(metric)
            cells.append(f"{val:.4f}" if isinstance(val, float) else "")
        lines.append("| " + " | ".join(cells) + " |")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize accuracies from Llama-3.2-1B logs.")
    parser.add_argument("--base_dir", default="testbench/log/llama-3.2-1b", help="Base log directory")
    parser.add_argument("--out_csv", default=None, help="CSV output path (default: <base_dir>/summary.csv)")
    parser.add_argument("--out_md", default=None, help="Markdown output path (default: <base_dir>/summary.md)")
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
