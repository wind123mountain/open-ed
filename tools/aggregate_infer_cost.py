#!/usr/bin/env python3
"""Aggregate multiple inference-cost measurement runs (median ± std per metric).

Usage:
    python3 tools/aggregate_infer_cost.py infer_cost_multi.jsonl
    python3 tools/aggregate_infer_cost.py infer_cost_multi.jsonl --out-md table.md

Input: JSONL with N runs per model_config (output of measure_infer_cost.py).
Output: Aggregated stats per model_config + Markdown table ready for paper.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict


# Metrics to aggregate (display order matters for the final table).
# Throughput / memory keys depend on the run's batch_size; we discover
# them at load time below.
BASE_METRICS = [
    ("params_B", "Params (B)", "{:.3f}"),
]

# Auto-discovered later: throughput_bs{N}_tokens_per_sec, samples_per_sec, peak_memory_bs{N}_gb


def stats(values):
    """Return median, mean, std, min, max for a list of values."""
    if not values:
        return None
    n = len(values)
    return {
        "n": n,
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if n > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def discover_metrics(runs):
    """Build the metric list dynamically from run keys (handles bs=N variations)."""
    sample = runs[0]
    metrics = list(BASE_METRICS)
    # Extract batch size from any throughput key
    bs = sample.get("batch_size")
    if bs is not None:
        metrics += [
            (f"throughput_bs{bs}_tokens_per_sec",  f"Tokens/s bs={bs}",         "{:.1f}"),
            (f"throughput_bs{bs}_samples_per_sec", f"Samples/s bs={bs}",        "{:.3f}"),
            (f"peak_memory_bs{bs}_gb",             f"Peak mem bs={bs} (GB)",    "{:.2f}"),
        ]
    return metrics


def aggregate(jsonl_path):
    """Group runs by model_config, compute stats per metric."""
    groups = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            groups[d["model_config"]].append(d)

    if not groups:
        raise ValueError(f"No runs found in {jsonl_path}")
    metrics_list = discover_metrics(next(iter(groups.values())))

    result = {}
    for cfg, runs in groups.items():
        name = runs[0]["name"]
        per_metric = {}
        for key, _, _ in metrics_list:
            vals = [r[key] for r in runs if key in r]
            per_metric[key] = stats(vals)
        result[cfg] = {
            "name": name,
            "n_runs": len(runs),
            "metrics": per_metric,
        }
    result["_metrics_schema"] = metrics_list
    return result


def fmt_with_std(s, fmt):
    if s is None: return "n/a"
    if s["n"] == 1:
        return fmt.format(s["median"])
    return f"{fmt.format(s['median'])} ± {fmt.format(s['std'])}"


def render_markdown(agg, order=("teacher", "student_eventkd", "student_sft")):
    metrics_list = agg["_metrics_schema"]
    lines = []
    cfgs = [k for k in agg if k != "_metrics_schema"]
    keys = [k for k in order if k in cfgs] + [k for k in cfgs if k not in order]
    headers = ["Model"] + [m[1] for m in metrics_list]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for k in keys:
        a = agg[k]
        row = [a["name"] + f" (n={a['n_runs']})"]
        for metric, _, fmt in metrics_list:
            row.append(fmt_with_std(a["metrics"].get(metric), fmt))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_ratio_row(agg, num_key, den_key="teacher"):
    """Compute student/teacher ratio for each metric (using median)."""
    if num_key not in agg or den_key not in agg:
        return None
    metrics_list = agg["_metrics_schema"]
    n = agg[num_key]["metrics"]
    d = agg[den_key]["metrics"]
    cells = [f"{agg[num_key]['name'].split('(')[0].strip()} / {agg[den_key]['name'].split('(')[0].strip()}"]
    for key, _, _ in metrics_list:
        if n.get(key) and d.get(key) and d[key]["median"] != 0:
            r = n[key]["median"] / d[key]["median"]
            cells.append(f"{r:.2f}×")
        else:
            cells.append("n/a")
    return "| " + " | ".join(cells) + " |"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="JSONL output of measure_infer_cost.py (multiple runs)")
    p.add_argument("--out-md", help="Also write Markdown table to this path")
    p.add_argument("--out-json", help="Also write aggregated stats as JSON")
    args = p.parse_args()

    agg = aggregate(args.input)

    metrics_list = agg["_metrics_schema"]
    print("=" * 80)
    print(f"Aggregated from {args.input}")
    print("=" * 80)
    for k, a in agg.items():
        if k == "_metrics_schema":
            continue
        print(f"\n[{k}] {a['name']}  (n_runs={a['n_runs']})")
        for metric, label, fmt in metrics_list:
            s = a["metrics"].get(metric)
            if s is None:
                continue
            if s["n"] == 1:
                print(f"  {label:30s}  {fmt.format(s['median'])}")
            else:
                rng = f"[{fmt.format(s['min'])}, {fmt.format(s['max'])}]"
                print(f"  {label:30s}  median={fmt.format(s['median'])}  "
                      f"mean={fmt.format(s['mean'])}  std={fmt.format(s['std'])}  range={rng}")

    print("\n" + "=" * 80)
    print("Markdown table (median ± std)")
    print("=" * 80)
    md = render_markdown(agg)
    print(md)

    # Also print student/teacher ratios
    print()
    print("Ratio rows (median, vs teacher):")
    for student in ("student_eventkd", "student_sft"):
        row = render_ratio_row(agg, student)
        if row:
            print(row)

    if args.out_md:
        with open(args.out_md, "w") as f:
            f.write(md + "\n")
        print(f"\nWrote {args.out_md}", file=sys.stderr)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"Wrote {args.out_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
