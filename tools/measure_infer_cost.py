#!/usr/bin/env python3
"""Measure architectural inference cost (tokens/s + memory) on ACE05.

Latency-style metrics are intentionally omitted because they depend on
generation behavior (when each model emits eos), which varies with the
training method and confounds comparison between models that share the
same architecture (e.g. Student EventKD vs Student SFT).

What we measure (architecture-only, fair across all variants):
  1. Parameter count
  2. Tokens / second at a fixed batch size (forced fixed generation
     length — same #tokens per sample for every model, so total time
     reflects pure per-step compute)
  3. Samples / second (derived = tokens/s / fixed_gen_len)
  4. Peak GPU memory during the throughput run

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 tools/measure_infer_cost.py \
        --model-config teacher --out infer_cost.jsonl
    CUDA_VISIBLE_DEVICES=1 python3 tools/measure_infer_cost.py \
        --model-config student_eventkd --out infer_cost.jsonl
    CUDA_VISIBLE_DEVICES=1 python3 tools/measure_infer_cost.py \
        --model-config student_sft --out infer_cost.jsonl
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


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
    with open(path) as f:
        records = [json.loads(line) for line in f]
    if len(records) < n:
        print(f"WARNING: only {len(records)} prompts available (requested {n}).",
              file=sys.stderr)
    return [r["prompt"] for r in records[:n]]


def load_model(cfg, device, dtype):
    if not (cfg.get("peft") and os.path.isdir(cfg["peft"])):
        raise FileNotFoundError(
            f"LoRA adapter not found at {cfg.get('peft')!r}.")
    base = AutoModelForCausalLM.from_pretrained(
        cfg["base"], dtype=dtype, device_map={"": device})
    model = PeftModel.from_pretrained(base, cfg["peft"])
    model = model.merge_and_unload()
    print(f"  loaded LoRA from {cfg['peft']} and merged into base",
          file=sys.stderr)
    model.eval()
    return model


def measure_throughput(model, tok, prompts, gen_cfg, device, batch_size,
                       max_prompt, fixed_gen_len):
    """Forced fixed-length generation. Each batch produces batch_size * fixed_gen_len
    new tokens. Total time / total tokens = pure per-step throughput."""
    n_batches = len(prompts) // batch_size
    if n_batches == 0:
        raise ValueError(
            f"n_samples ({len(prompts)}) < batch_size ({batch_size}).")
    total_samples = 0
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for b in range(n_batches):
            batch = prompts[b * batch_size : (b + 1) * batch_size]
            ids = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_prompt).to(device)
            model.generate(**ids, generation_config=gen_cfg)
            total_samples += batch_size
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - t0
    total_new_tokens = total_samples * fixed_gen_len
    return {
        "n_batches": n_batches,
        "total_samples": total_samples,
        "total_new_tokens": total_new_tokens,
        "total_seconds": duration,
        "samples_per_sec": total_samples / duration if duration > 0 else 0.0,
        "tokens_per_sec": total_new_tokens / duration if duration > 0 else 0.0,
    }


def measure(args):
    cfg = MODEL_CONFIGS[args.model_config]
    device = "cuda:0"
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

    # Forced fixed-length generation: every model emits exactly fixed_gen_len
    # tokens per sample. No eos shortcut. No generation-length confound.
    gen_cfg = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.fixed_gen_len,
        min_new_tokens=args.fixed_gen_len,
        pad_token_id=tok.pad_token_id,
        eos_token_id=None,
        return_dict_in_generate=False,
    )

    bs = args.batch_size
    print(f"  warmup {args.warmup} batches at bs={bs} (fixed gen={args.fixed_gen_len})...",
          file=sys.stderr)
    for _ in range(args.warmup):
        warm_batch = prompts[:bs]
        ids = tok(warm_batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_prompt_length).to(device)
        with torch.no_grad():
            model.generate(**ids, generation_config=gen_cfg)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print(f"  measuring throughput bs={bs}...", file=sys.stderr)
    thr = measure_throughput(model, tok, prompts, gen_cfg, device, bs,
                             args.max_prompt_length, args.fixed_gen_len)
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    return {
        "name": cfg["name"],
        "model_config": args.model_config,
        # --- size ---
        "params_B": round(n_params / 1e9, 4),
        "params_raw": n_params,
        # --- throughput (fixed length, fair architectural comparison) ---
        f"throughput_bs{bs}_tokens_per_sec": round(thr["tokens_per_sec"], 1),
        f"throughput_bs{bs}_samples_per_sec": round(thr["samples_per_sec"], 3),
        f"throughput_bs{bs}_total_samples": thr["total_samples"],
        f"throughput_bs{bs}_total_new_tokens": thr["total_new_tokens"],
        f"throughput_bs{bs}_total_seconds": round(thr["total_seconds"], 3),
        f"throughput_bs{bs}_n_batches": thr["n_batches"],
        # --- memory ---
        f"peak_memory_bs{bs}_gb": round(peak_mem_gb, 3),
        # --- meta (reproducibility) ---
        "n_samples": len(prompts),
        "batch_size": bs,
        "fixed_gen_len": args.fixed_gen_len,
        "max_prompt_length": args.max_prompt_length,
        "warmup_batches": args.warmup,
        "decode": "greedy_forced_fixed_length",
        "dtype": "bfloat16",
        "lora_merged": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", required=True, choices=list(MODEL_CONFIGS),
                   help="Which model to measure")
    p.add_argument("--data", default="processed_data/ace/qwen/test.jsonl",
                   help="Test jsonl with pre-rendered chat-template prompts")
    p.add_argument("--n-samples", type=int, default=64,
                   help="Number of prompts (must be multiple of batch_size)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--fixed-gen-len", type=int, default=128,
                   help="Force every sample to generate exactly this many tokens "
                        "(eliminates generation-length confound between models)")
    p.add_argument("--max-prompt-length", type=int, default=460)
    p.add_argument("--warmup", type=int, default=5)
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
