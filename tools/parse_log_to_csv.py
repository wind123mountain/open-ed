"""
Parse finetune log.txt files and write all eval results to a CSV file.

Supports two modes:
  --log <log.txt>          Parse a single log file
  --log-dir <run_dir>      Parse all <run_dir>/{0,1,...}/log.txt files (CL mode)

Usage:
    python tools/parse_log_to_csv.py --log-dir results/qwen3/v6/ace_0_0.6B_cl --out results.csv
    python tools/parse_log_to_csv.py --log <log.txt> --out <results.csv>
"""

import argparse
import ast
import csv
import re
import sys
from pathlib import Path


EXP_RE   = re.compile(r"={10,}\s+EXP at (.+?)\s+={10,}")
EVAL_RE  = re.compile(
    r"^(dev|test)\s*\|\s*avg_loss:\s*([\d.]+)\s*\|\s*(\{.*\})\s*$"
)
TRAIN_RE = re.compile(r"train \| epoch\s+(\d+)")

ED_FIELDNAMES = [
    "task_id", "exp_time", "eval_idx", "eval_type", "split", "avg_loss",
    "exact_match", "rougeL",
    "trigger_text_precision", "trigger_text_recall", "trigger_text_f1",
    "trigger_precision", "trigger_recall", "trigger_f1",
    "trigger_cls_acc",
    "trigger_tp", "trigger_fp", "trigger_fn",
    "argument_precision", "argument_recall", "argument_f1",
    "argument_tp", "argument_fp", "argument_fn",
]

RE_FIELDNAMES = [
    "task_id", "exp_time", "eval_idx", "eval_type", "split", "avg_loss",
    "exact_match", "rougeL",
    "relation_text_precision", "relation_text_recall", "relation_text_f1",
    "relation_precision", "relation_recall", "relation_f1",
    "relation_cls_acc",
    "relation_tp", "relation_fp", "relation_fn",
]


def parse_log(log_path: str, task_id_override=None):
    rows = []
    current_exp_time = None
    eval_idx = 0
    task_id = -1 if task_id_override is None else task_id_override
    current_epoch = -1
    seen_train_line = False

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            exp_match = EXP_RE.search(line)
            if exp_match:
                current_exp_time = exp_match.group(1).strip()
                eval_idx = 0
                current_epoch = -1
                seen_train_line = False
                if task_id_override is None:
                    task_id += 1
                continue

            train_match = TRAIN_RE.search(line)
            if train_match:
                current_epoch = int(train_match.group(1))
                seen_train_line = True
                continue

            eval_match = EVAL_RE.match(line)
            if eval_match and current_exp_time:
                split    = eval_match.group(1)
                avg_loss = float(eval_match.group(2))
                metrics  = ast.literal_eval(eval_match.group(3))

                # Determine eval_type from context
                if not seen_train_line:
                    eval_type = "pre_train"
                else:
                    eval_type = f"epoch_{current_epoch}_{split}"

                is_re = "relation" in metrics

                base = {
                    "task_id": task_id,
                    "exp_time": current_exp_time,
                    "eval_idx": eval_idx,
                    "eval_type": eval_type,
                    "split": split,
                    "avg_loss": avg_loss,
                    "exact_match": metrics.get("exact_match", ""),
                    "rougeL": metrics.get("rougeL", ""),
                }

                if is_re:
                    rel  = metrics.get("relation",  {})
                    rtxt = metrics.get("relation_text", {})
                    rc   = metrics.get("relation_counts", {})
                    base.update({
                        "relation_text_precision": rtxt.get("precision", ""),
                        "relation_text_recall":    rtxt.get("recall", ""),
                        "relation_text_f1":        rtxt.get("f1", ""),
                        "relation_precision": rel.get("precision", ""),
                        "relation_recall":    rel.get("recall", ""),
                        "relation_f1":        rel.get("f1", ""),
                        "relation_cls_acc": metrics.get("relation_cls_acc", ""),
                        "relation_tp": rc.get("tp", ""),
                        "relation_fp": rc.get("fp", ""),
                        "relation_fn": rc.get("fn", ""),
                    })
                else:
                    trig  = metrics.get("trigger",  {})
                    ttxt  = metrics.get("trigger_text", {})
                    arg   = metrics.get("argument", {})
                    tc    = metrics.get("trigger_counts",  {})
                    ac    = metrics.get("argument_counts", {})
                    base.update({
                        "trigger_text_precision": ttxt.get("precision", ""),
                        "trigger_text_recall":    ttxt.get("recall", ""),
                        "trigger_text_f1":        ttxt.get("f1", ""),
                        "trigger_precision": trig.get("precision", ""),
                        "trigger_recall":    trig.get("recall", ""),
                        "trigger_f1":        trig.get("f1", ""),
                        "trigger_cls_acc": metrics.get("trigger_cls_acc", ""),
                        "trigger_tp": tc.get("tp", ""),
                        "trigger_fp": tc.get("fp", ""),
                        "trigger_fn": tc.get("fn", ""),
                        "argument_precision": arg.get("precision", ""),
                        "argument_recall":    arg.get("recall", ""),
                        "argument_f1":        arg.get("f1", ""),
                        "argument_tp": ac.get("tp", ""),
                        "argument_fp": ac.get("fp", ""),
                        "argument_fn": ac.get("fn", ""),
                    })

                rows.append(base)
                eval_idx += 1

    return rows


def parse_log_dir(log_dir: str):
    """Parse all {task_id}/log.txt files in a CL run directory."""
    rows = []
    log_dir = Path(log_dir)
    task_dirs = sorted(
        [d for d in log_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name)
    )
    for task_dir in task_dirs:
        log_file = task_dir / "log.txt"
        if log_file.exists():
            task_id = int(task_dir.name)
            rows.extend(parse_log(str(log_file), task_id_override=task_id))
    return rows


def write_csv(rows, out_path=None):
    # Auto-detect ED vs RE based on first row's keys
    if rows and "relation_f1" in rows[0]:
        fieldnames = RE_FIELDNAMES
    else:
        fieldnames = ED_FIELDNAMES
    fh = open(out_path, "w", newline="") if out_path else sys.stdout
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    if out_path:
        fh.close()
        print(f"Saved {len(rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse finetune log.txt → CSV")
    parser.add_argument("--log", default=None, help="Path to a single log.txt")
    parser.add_argument("--log-dir", default=None, help="CL run directory containing {task_id}/log.txt files")
    parser.add_argument("--out", default=None,  help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    if args.log_dir:
        rows = parse_log_dir(args.log_dir)
    elif args.log:
        rows = parse_log(args.log)
    else:
        parser.error("Either --log or --log-dir is required")

    if not rows:
        print("No eval entries found.", file=sys.stderr)
        sys.exit(1)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
