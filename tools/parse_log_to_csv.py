"""
Parse a finetune log.txt and write all eval results to a CSV file.

Each EXP block that contains eval lines is treated as one task run.
Within a task, eval entries are numbered sequentially (0 = initial eval
before training, 1 = after epoch 1, etc.).

Usage:
    python tools/parse_log_to_csv.py --log <log.txt> --out <results.csv>
    python tools/parse_log_to_csv.py --log <log.txt>          # prints to stdout
"""

import argparse
import ast
import csv
import re
import sys
from pathlib import Path


EXP_RE  = re.compile(r"={10,}\s+EXP at (.+?)\s+={10,}")
EVAL_RE = re.compile(
    r"^(dev|test)\s*\|\s*avg_loss:\s*([\d.]+)\s*\|\s*(\{.*\})\s*$"
)

FIELDNAMES = [
    "task_id", "exp_time", "eval_idx", "split", "avg_loss",
    "exact_match", "rougeL",
    "trigger_precision", "trigger_recall", "trigger_f1",
    "trigger_tp", "trigger_fp", "trigger_fn",
    "argument_precision", "argument_recall", "argument_f1",
    "argument_tp", "argument_fp", "argument_fn",
]


def parse_log(log_path: str):
    rows = []
    current_exp_time = None
    eval_idx = 0
    task_id = -1

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            exp_match = EXP_RE.search(line)
            if exp_match:
                current_exp_time = exp_match.group(1).strip()
                eval_idx = 0      # reset counter for new run
                continue

            eval_match = EVAL_RE.match(line)
            if eval_match and current_exp_time:
                split    = eval_match.group(1)
                avg_loss = float(eval_match.group(2))
                metrics  = ast.literal_eval(eval_match.group(3))

                # Increment task_id only on eval_idx==0 (first eval of a run)
                if eval_idx == 0:
                    task_id += 1

                trig  = metrics.get("trigger",  {})
                arg   = metrics.get("argument", {})
                tc    = metrics.get("trigger_counts",  {})
                ac    = metrics.get("argument_counts", {})

                rows.append({
                    "task_id": task_id,
                    "exp_time": current_exp_time,
                    "eval_idx": eval_idx,
                    "split": split,
                    "avg_loss": avg_loss,
                    "exact_match": metrics.get("exact_match", ""),
                    "rougeL": metrics.get("rougeL", ""),
                    "trigger_precision": trig.get("precision", ""),
                    "trigger_recall":    trig.get("recall", ""),
                    "trigger_f1":        trig.get("f1", ""),
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
                eval_idx += 1

    return rows


def write_csv(rows, out_path=None):
    fh = open(out_path, "w", newline="") if out_path else sys.stdout
    writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
    if out_path:
        fh.close()
        print(f"Saved {len(rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse finetune log.txt → CSV")
    parser.add_argument("--log", required=True, help="Path to log.txt")
    parser.add_argument("--out", default=None,  help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    rows = parse_log(args.log)
    if not rows:
        print("No eval entries found in log.", file=sys.stderr)
        sys.exit(1)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
