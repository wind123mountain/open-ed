| Model | Params (B) | Tokens/s bs=16 | Samples/s bs=16 | Peak mem bs=16 (GB) |
|---|---|---|---|---|
| Qwen3-4B + LoRA SFT (n=3) | 4.022 | 151.9 ± 0.6 | 1.187 ± 0.005 | 9.25 ± 0.00 |
| Qwen3-0.6B + EventKD (ours) (n=3) | 0.596 | 264.6 ± 4.2 | 2.067 ± 0.033 | 2.17 ± 0.00 |
| Qwen3-0.6B + SFT baseline (n=3) | 0.596 | 267.0 ± 3.4 | 2.086 ± 0.026 | 2.17 ± 0.00 |