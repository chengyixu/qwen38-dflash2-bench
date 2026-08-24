#!/usr/bin/env python3
"""Streaming chat-completion benchmark: measures TTFT (prefill), decode tok/s, memory-free.

Usage:
  bench_api.py --base-url http://127.0.0.1:7870 --model ID --prompt-tokens 24000 \
      --gen-tokens 256 --label llama-baseline [--temperature 1.0] [--top-p 0.95] [--runs 1]

Builds a synthetic ~N-token prompt, streams the response, reports per-run JSON lines:
{label, run, prompt_tokens, gen_tokens, ttft_s, prefill_tps, decode_tps, total_s}
"""
import argparse, json, sys, time, urllib.request

WORDS = ("system module function class return value buffer kernel pipeline latency throughput "
         "memory compute tensor vector matrix layer network gradient optimizer schedule queue "
         "cache index segment offset pointer register thread process signal protocol session "
         "packet socket stream frame block chunk region domain context scope profile metric "
         "benchmark evaluate transform encode decode embed attention normalize aggregate").split()

def build_prompt(target_tokens: int) -> str:
    # Plain prose; calibrated 1.483 tokens/word against Qwen3 tokenizer
    n_words = int(target_tokens / 1.483)
    parts = []
    i = 0
    while len(parts) < n_words:
        w1 = WORDS[i % len(WORDS)]
        w2 = WORDS[(i * 7 + 3) % len(WORDS)]
        parts.append(f"{w1} {w2}")
        i += 1
    body = " ".join(parts)
    return ("Below is a technical log excerpt. Read it carefully.\n\n" + body +
            "\n\nEnd of excerpt. Without summarizing the excerpt, answer: what is 17 * 23? "
            "Reply with just the number.")

def run_once(url, model, prompt, gen_tokens, temperature, top_p, run_idx, label, api_key=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": gen_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    t0 = time.perf_counter()
    ttft = None
    out_chars = 0
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                out_chars += len(choices[0]["delta"]["content"])
    total = time.perf_counter() - t0
    ptoks = usage.get("prompt_tokens") if usage else None
    gtoks = usage.get("completion_tokens") if usage else None
    prefill_tps = round(ptoks / ttft, 2) if (ptoks and ttft) else None
    decode_tps = round((gtoks - 1) / (total - ttft), 2) if (gtoks and ttft and total > ttft) else None
    result = {
        "label": label, "run": run_idx,
        "prompt_tokens": ptoks, "gen_tokens": gtoks,
        "ttft_s": round(ttft, 3) if ttft else None,
        "prefill_tps": prefill_tps, "decode_tps": decode_tps,
        "total_s": round(total, 2),
    }
    print(json.dumps(result), flush=True)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=100)
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--label", default="bench")
    args = ap.parse_args()

    url = args.base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    prompt = build_prompt(args.prompt_tokens)
    results = [run_once(url, args.model, prompt, args.gen_tokens,
                        args.temperature, args.top_p, i, args.label, api_key=args.api_key)
               for i in range(args.runs)]
    # summary (median decode)
    decodes = sorted(r["decode_tps"] for r in results if r["decode_tps"])
    ttfts = sorted(r["ttft_s"] for r in results if r["ttft_s"] is not None)
    if decodes:
        print(json.dumps({"label": args.label, "summary": True,
                          "median_decode_tps": decodes[len(decodes)//2],
                          "min_decode_tps": decodes[0], "max_decode_tps": decodes[-1],
                          "median_ttft_s": ttfts[len(ttfts)//2] if ttfts else None}), flush=True)

if __name__ == "__main__":
    main()
