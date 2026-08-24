# LocalFlash: Qwen3.8-27B + DFlash2 on Apple Silicon

[![Paper](https://img.shields.io/badge/paper-PDF-blue)](paper/main.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)]()
[![Runtime](https://img.shields.io/badge/runtime-oMLX%20z--lab%20fork%200.6.2-orange)]()

End-to-end engineering study and reproducible artifacts for serving **Qwen3.8-27B**
at 4-bit precision with **DFlash 2 block-diffusion speculative decoding** on an
Apple M4 Max (64 GB unified memory) — optimized for long-context coding-agent
workloads (≥50k tokens native).

## Headline results

| Metric | llama.cpp baseline | **LocalFlash (this work)** |
|---|---|---|
| Decode (median, short ctx) | ~15 tok/s | **809 tok/s** (54×) |
| TTFT @ 32.5k fresh prompt | 435 s *every turn* | 486 s **once** |
| TTFT @ cached prefix turn | n/a (>32k cap) | **8–16 s** (26–52×) |
| Context window | 32k hard cap | **262k native** |
| Long-ctx recall probe | untested | 5/5 ordered needles @ 70k |

All numbers measured on identical synthetic workloads; raw JSON-lines in
[`results/raw/`](results/raw/). Full methodology, negative results, and a
measurement-pitfall post-mortem in [`paper/main.pdf`](paper/main.pdf).

## Why prefix caching matters more than tok/s

Coding agents re-submit a growing shared context every tool-call turn. A stack
that wins single-turn decode but re-prefills per turn is unusable locally:

```
Turn 1 (cold 32.5k):  428 s   ← paid once
Turn 2 (+35 tokens):  8.2 s   ← L1 prefix cache hit, 52× faster than baseline
Turn 3 (follow-up):  15.8 s   ← still cached
```

This reproduces FreeToken's (arXiv:2608.16157) key finding on unified memory:
state reuse, not kernel speed, governs perceived agentic latency.

## Repo layout

```
├── paper/            arXiv-style LaTeX paper (compiles with tectonic)
├── bench/
│   ├── bench_api.py       calibrated workload generator + streaming benchmarker
│   ├── qa_check.py        long-context needle & code-generation gates
│   ├── agent_turns.py     multi-turn prefix-reuse protocol
│   └── hf_fetch.py        resumable, hash-verifying model downloader
├── deploy/
│   ├── local-llm.command              production controller (status/start/stop/test/logs)
│   └── omlx-model-settings.example.json  DFlash engine configuration
└── results/raw/       39 raw measurement records (JSONL)
```

## Quickstart (reproduce)

```bash
# 0. The controller requires your server API key:
export LOCAL_LLM_API_KEY="<your omlx api key>"   # stored in ~/.omlx/settings.json

# 1. Install the z-lab oMLX fork (0.6.2-dflash2) from
#    https://github.com/z-lab/omlx-fork/releases
# 2. Fetch weights (hash-verified):
python bench/hf_fetch.py mlx-community/Qwen3.8-27B-4bit ~/Models/mlx/Qwen3.8-27B-4bit
python bench/hf_fetch.py z-lab/Qwen3.8-27B-DFlash2      ~/Models/mlx/Qwen3.8-27B-DFlash2

# 3. Configure the engine (drafter auto-quantized 4-bit, block size 5)
cp deploy/omlx-model-settings.example.json ~/.omlx/model_settings.json

# 4. Serve + verify
deploy/local-llm.command start && deploy/local-llm.command test

# 5. Benchmark (single-instance check first! see paper §5)
python bench/bench_api.py --base-url http://127.0.0.1:7870 \
    --model Qwen3.8-27B-4bit --prompt-tokens 128 --gen-tokens 384 \
    --api-key "$OMLX_API_KEY" --runs 3 --label short
```

## Configuration that won

```json
{
  "dflash_enabled": true,
  "dflash_draft_model": "~/Models/mlx/Qwen3.8-27B-DFlash2",
  "dflash_draft_quant_enabled": true,
  "dflash_draft_quant_weight_bits": 4,
  "dflash_draft_quant_group_size": 64,
  "dflash_block_size": 5,
  "dflash_verify_mode": null,
  "dflash_in_memory_cache": true
}
```

Block size ≤5 follows z-lab guidance for quantized targets (MLX quantized-matmul
efficiency degrades at wider verify shapes).

## Rejected alternatives (measured, not vibes)

| Stack | Result |
|---|---|
| llama.cpp build 10566 + DFlash2 GGUF | drafter tensor-layout mismatch → AR fallback; 353 tok/s decode but 373 s prefill |
| Vanilla mlx-lm | ~10 tok/s, no serving features |
| Mainline oMLX + MTP head | far slower than fork's DFlash engine |
| FreeToken | CUDA-only; its *idea* (state reuse) is realized here via prefix cache |

## Measurement pitfalls (read before benchmarking)

1. **Single-instance invariant**: a package-manager service silently rebound our
   port and invalidated two full benchmark batches (decode read 19 tok/s instead
   of 809). Always verify process identity + swap pressure before runs.
2. **Hash-verify weights**: interrupted+resumed downloads produced shards with
   valid headers but corrupt payloads — model loads, then emits gibberish.
   Only SHA-256 against the hub manifest catches it.
3. **Budget thinking tokens**: quality probes must cap above the model's own
   reasoning or they manufacture failures.

## Citation

```bibtex
@misc{localflash2026,
  title={Losing the Draft, Keeping the Speed: Block-Diffusion Speculative
         Decoding and Prefix-Cached Serving for Long-Context Agentic LLM
         Workloads on Apple Silicon},
  author={chengyixu},
  year={2026},
  url={https://github.com/chengyixu/qwen38-dflash2-bench}
}
```

## Credits

- [z-lab/dflash](https://github.com/z-lab/dflash) & [Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) — the drafter
- [z-lab/omlx-fork](https://github.com/z-lab/omlx-fork) — serving engine
- [jundot/omlx](https://github.com/jundot/omlx) — upstream server
- [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) — target weights
- [FreeToken (arXiv:2608.16157)](https://arxiv.org/abs/2608.16157) — agentic-serving insights

MIT License. Secrets are redacted; the deployment API key appears nowhere in this repository.
