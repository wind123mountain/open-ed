"""Measure inference cost for teacher / student models on ACE05 (rebuttal table).

Reports per-model: param count, latency (bs=1), throughput (bs=N), peak GPU
memory (separately for bs=1 and bs=N), and length stats. Run once per model;
combine the JSONL lines into the final paper table.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 tools/measure_infer_cost.py \
        --model-config teacher --out infer_cost.jsonl
    CUDA_VISIBLE_DEVICES=1 python3 tools/measure_infer_cost.py \
        --model-config student_eventkd --out infer_cost.jsonl
    CUDA_VISIBLE_DEVICES=1 python3 tools/measure_infer_cost.py \
        --model-config student_sft --out infer_cost.jsonl

Notes:
- LoRA is merged into the base via merge_and_unload() so the measurement
  reflects deployment cost (no PEFT wrapper overhead at inference).
- Greedy decoding (do_sample=False) is used for stable, reproducible timings.
- eos_token_id matches finetune.py's eval setup ([chat_eos, 151643]) so that
  Qwen3 generation stops at the same tokens as the paper's reported eval.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


# Eval-time eos hardcoded in finetune.py (Qwen-specific; 151643 = <|endoftext|>).
# Replicated here for fair comparison with paper-reported eval cost.
QWEN_EOS_ENDOFTEXT = 151643


MODEL_CONFIGS = {
    "teacher": {
        "name": "Qwen3-4B + LoRA SFT",
        "base": "Qwen/Qwen3-4B-Instruct-2507",
        "peft": "results/qwen3/sft_4B_ace/e5-bs2-lr0.0001-G8-N2-NN1-lora-32-64-0.05/490",
    },
    "student_eventkd": {
        "name": "Qwen3-0.6B + EventKD (ours)",
        "base": "Qwen/Qwen3-0.6B",
        "peft": "results/qwen3/span_distillm/0.6B_4B_ace_srkl_2/490",
    },
    "student_sft": {
        "name": "Qwen3-0.6B + SFT baseline",
        "base": "Qwen/Qwen3-0.6B",
        "peft": "results/qwen3/sft_0.6B_ace_lora/e5-bs8-lr5e-05-G2-N2-NN1-lora-8-64-0.1/490",
    },
}


def load_prompts(path, n):
    """Take first N pre-rendered chat-template prompts from processed test.jsonl."""
    with open(path) as f:
        records = [json.loads(line) for line in f]
    prompts = [r["prompt"] for r in records[:n]]
    if len(prompts) < n:
        print(f"WARNING: only {len(prompts)} prompts available (requested {n}).",
              file=sys.stderr)
    return prompts


def load_model(cfg, device, dtype):
    if not (cfg.get("peft") and os.path.isdir(cfg["peft"])):
        raise FileNotFoundError(
            f"LoRA adapter not found at {cfg.get('peft')!r}. "
            "Pull the checkpoint first or update MODEL_CONFIGS.")
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base"], dtype=dtype, device_map={"": device})
    model = PeftModel.from_pretrained(base, cfg["peft"])
    model = model.merge_and_unload()
    print(f"  loaded LoRA from {cfg['peft']} and merged into base",
          file=sys.stderr)
    model.eval()
    return model


def reset_cuda(device):
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def warmup(model, tok, prompts, gen_cfg, device, batch_size, max_prompt, n_warm):
    """Run n_warm batches at the given batch_size to JIT-compile CUDA kernels
    and pre-populate the allocator pool for the upcoming measurement phase."""
    if not prompts:
        return
    batch = prompts[:batch_size]
    while len(batch) < batch_size:
        batch.append(prompts[0])  # repeat to fill batch size
    for _ in range(n_warm):
        ids = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_prompt).to(device)
        with torch.no_grad():
            model.generate(**ids, generation_config=gen_cfg)
    torch.cuda.synchronize(device)


def measure_latency_bs1(model, tok, prompts, gen_cfg, device, max_prompt):
    """Per-sample wall time for batch size 1. Returns list of per-sample dicts."""
    out = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt", padding=False,
                      truncation=True, max_length=max_prompt).to(device)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            seq = model.generate(**ids, generation_config=gen_cfg)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            n_new = seq.size(1) - ids["input_ids"].size(1)
            out.append({
                "wall_ms": elapsed_ms,
                "new_tokens": int(n_new),
                "ms_per_token": elapsed_ms / max(n_new, 1),
            })
    return out


def measure_throughput(model, tok, prompts, gen_cfg, device, batch_size, max_prompt):
    """Total wall time for batched generation; returns dict of aggregate stats."""
    n_batches = len(prompts) // batch_size
    if n_batches == 0:
        raise ValueError(
            f"n_samples ({len(prompts)}) < batch_size ({batch_size}); "
            "increase --n-samples or decrease --batch-size.")
    total_samples = 0
    total_new_tokens = 0
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for b in range(n_batches):
            batch = prompts[b * batch_size : (b + 1) * batch_size]
            ids = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_prompt).to(device)
            seq = model.generate(**ids, generation_config=gen_cfg)
            new_tokens = seq.size(1) - ids["input_ids"].size(1)
            total_new_tokens += int(new_tokens) * batch_size
            total_samples += batch_size
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - t0
    return {
        "n_batches": n_batches,
        "total_samples": total_samples,
        "total_new_tokens_approx": total_new_tokens,  # approx: max length per batch × bs
        "total_seconds": duration,
        "samples_per_sec": total_samples / duration if duration > 0 else 0.0,
        "tokens_per_sec_approx": total_new_tokens / duration if duration > 0 else 0.0,
    }


def measure(args):
    cfg = MODEL_CONFIGS[args.model_config]
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES from shell controls actual GPU
    dtype = torch.bfloat16

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"[{args.model_config}] {cfg['name']}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(cfg["base"], padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = load_model(cfg, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())

    prompts = load_prompts(args.data, args.n_samples)
    if len(prompts) < args.batch_size:
        raise ValueError(
            f"Loaded only {len(prompts)} prompts but batch_size={args.batch_size}.")
    print(f"  {len(prompts)} prompts loaded from {args.data}", file=sys.stderr)

    # Build eos list matching finetune.py's eval (Qwen-specific)
    eos_ids = [tok.eos_token_id]
    if QWEN_EOS_ENDOFTEXT not in eos_ids:
        eos_ids.append(QWEN_EOS_ENDOFTEXT)

    gen_cfg = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_ids,
        return_dict_in_generate=False,
    )

    # ===== PHASE 1: latency at batch_size=1 =====
    print(f"  warmup {args.warmup} batches at bs=1...", file=sys.stderr)
    warmup(model, tok, prompts, gen_cfg, device,
           batch_size=1, max_prompt=args.max_prompt_length, n_warm=args.warmup)
    reset_cuda(device)

    print(f"  measuring latency bs=1 over {len(prompts)} samples...",
          file=sys.stderr)
    lat = measure_latency_bs1(model, tok, prompts, gen_cfg, device,
                              args.max_prompt_length)
    peak_bs1_gb = torch.cuda.max_memory_allocated(device) / 1e9

    # ===== PHASE 2: throughput at batch_size=N =====
    bs = args.batch_size
    print(f"  warmup {args.warmup} batches at bs={bs}...", file=sys.stderr)
    warmup(model, tok, prompts, gen_cfg, device,
           batch_size=bs, max_prompt=args.max_prompt_length, n_warm=args.warmup)
    reset_cuda(device)

    print(f"  measuring throughput bs={bs}...", file=sys.stderr)
    thr = measure_throughput(model, tok, prompts, gen_cfg, device, bs,
                             args.max_prompt_length)
    peak_bsN_gb = torch.cuda.max_memory_allocated(device) / 1e9

    return {
        "name": cfg["name"],
        "model_config": args.model_config,
        # --- size ---
        "params_B": round(n_params / 1e9, 4),
        "params_raw": n_params,
        # --- latency (bs=1) ---
        "latency_bs1_ms_median": round(statistics.median(L["wall_ms"] for L in lat), 1),
        "latency_bs1_ms_mean": round(statistics.mean(L["wall_ms"] for L in lat), 1),
        "ms_per_token_median": round(statistics.median(L["ms_per_token"] for L in lat), 2),
        "new_tokens_median_bs1": int(statistics.median(L["new_tokens"] for L in lat)),
        "peak_memory_bs1_gb": round(peak_bs1_gb, 2),
        # --- throughput (bs=N) ---
        f"throughput_bs{bs}_samples_per_sec": round(thr["samples_per_sec"], 2),
        f"throughput_bs{bs}_tokens_per_sec_approx": round(thr["tokens_per_sec_approx"], 1),
        f"throughput_bs{bs}_n_batches": thr["n_batches"],
        f"throughput_bs{bs}_total_samples": thr["total_samples"],
        f"throughput_bs{bs}_total_seconds": round(thr["total_seconds"], 2),
        "peak_memory_bs%d_gb" % bs: round(peak_bsN_gb, 2),
        # --- meta ---
        "n_samples": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_length": args.max_prompt_length,
        "warmup_batches": args.warmup,
        "decode": "greedy",
        "dtype": "bfloat16",
        "eos_token_ids": eos_ids,
        "lora_merged": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", required=True, choices=list(MODEL_CONFIGS),
                   help="Which model to measure")
    p.add_argument("--data", default="processed_data/ace/qwen/test.jsonl",
                   help="Test jsonl with pre-rendered chat-template prompts")
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for throughput measurement")
    p.add_argument("--max-new-tokens", type=int, default=308,
                   help="Cap on generation length (= 768 - max_prompt_length)")
    p.add_argument("--max-prompt-length", type=int, default=460)
    p.add_argument("--warmup", type=int, default=5,
                   help="Warmup batches per phase (bs=1 and bs=N each)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="Append JSONL line to this file")
    args = p.parse_args()

    result = measure(args)
    line = json.dumps(result, ensure_ascii=False)
    print(line)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
