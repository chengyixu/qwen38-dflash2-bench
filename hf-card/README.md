---
language: en
license: mit
library_name: mlx
base_model:
- Qwen/Qwen3.8-27B
- z-lab/Qwen3.8-27B-DFlash2
tags:
- qwen3.8
- dflash2
- speculative-decoding
- block-diffusion
- apple-silicon
- mlx
- 4-bit
- long-context
- agentic
- omlx
- benchmark
inference: false
---

# LocalFlash: Qwen3.8-27B + DFlash2 serving configuration for Apple Silicon

This repository documents a **measured, reproducible deployment recipe** (not new weights):
Qwen3.8-27B at MLX 4-bit, accelerated by the **DFlash 2** block-diffusion drafter under the
z-lab **oMLX fork**, tuned for long-context coding-agent workloads on an M4 Max / 64 GB.

## Measured results (M4 Max, 64 GB, macOS 27.0)

| Metric | llama.cpp baseline | This configuration |
|---|---|---|
| Decode (median) | ~15 tok/s | **809 tok/s** |
| TTFT @ 32.5k fresh prompt | 435 s every turn | 486 s once |
| TTFT @ cached prefix turn | — | **8–16 s** |
| Context window | 32k | 262k native |
| Needle recall @ 70k | — | 5/5 ordered |

Raw measurement records: [GitHub repo](https://github.com/chengyixu/qwen38-dflash2-bench) → `results/raw/`.

## Files

No weight files are hosted here — use the upstream artifacts:

- Target: [`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)
- Drafter: [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) (BF16; quantize at load with the engine config below — pre-quantized drafter files have no consumer loader yet, see `bench/convert_dflash_4bit.py`)
- Server: [`z-lab/omlx-fork` release `0.6.2-dflash2`](https://github.com/z-lab/omlx-fork/releases)

This repo ships the **research package**:

```
├── README.md            this card (recipe + measured results)
├── main.pdf             the full paper (methodology, tables, pitfalls)
├── bench/               calibrated benchmark & QA harnesses
├── deploy/              production controller + engine config
└── results/raw/*.jsonl  39 raw measurement records
```

## Engine configuration

Place as `~/.omlx/model_settings.json`:

```json
{
  "version": 1,
  "models": {
    "Qwen3.8-27B-4bit": {
      "dflash_enabled": true,
      "dflash_draft_model": "/Users/<you>/Models/mlx/Qwen3.8-27B-DFlash2",
      "dflash_draft_quant_enabled": true,
      "dflash_draft_quant_weight_bits": 4,
      "dflash_draft_quant_activation_bits": 16,
      "dflash_draft_quant_group_size": 64,
      "dflash_block_size": 5,
      "dflash_verify_mode": null,
      "dflash_in_memory_cache": true,
      "display_name": "Qwen3.8-27B 4bit + DFlash2"
    }
  }
}
```

Notes: block size ≤5 per z-lab guidance for quantized targets; adaptive verify;
L1 prefix cache is what turns multi-turn agent sessions from minutes-per-turn
into seconds-per-turn.

## Integrity requirement

Hash-verify both models against the hub manifests before first serve. Two of our
three shards arrived corrupt after resumed downloads — with valid safetensors
headers and correct sizes. The model loads and then emits fluent gibberish.
`bench/hf_fetch.py` in the GitHub repo wraps download + verification.

## Sampling (per Qwen3.8 model card)

Thinking mode: `temperature=1.0, top_p=0.95, top_k=20`; reasoning effort
xhigh/medium/low via the server's chat-template kwargs.

## Citation

```bibtex
@misc{localflash2026,
  title={Losing the Draft, Keeping the Speed},
  author={chengyixu},
  year={2026},
  url={https://github.com/chengyixu/qwen38-dflash2-bench}
}
```
