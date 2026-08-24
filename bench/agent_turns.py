#!/usr/bin/env python3
"""Agentic multi-turn prefix-cache test: big base context + follow-up turns."""
import json, sys, time, urllib.request
sys.path.insert(0, "/private/tmp/commandcode-501/-Users-wilsonxu-Models/34f4e0c2-cc8e-4c61-b935-9b45fc904635/scratchpad")
from bench_api import build_prompt

def stream_chat(url, key, model, messages, max_tokens=128):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 1.0, "top_p": 0.95, "stream": True,
               "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(), headers=headers)
    t0 = time.perf_counter()
    ttft = None
    usage = None
    text = ""
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                obj = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text += ch[0]["delta"]["content"]
    return {"ttft_s": round(ttft, 2) if ttft else None,
            "total_s": round(time.perf_counter() - t0, 2),
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "gen_tokens": usage.get("completion_tokens") if usage else None,
            "head": text[:60].replace("\n", " ")}

def main():
    url, key, model = sys.argv[1], sys.argv[2], sys.argv[3]
    base_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 24000

    base_doc = build_prompt(base_tokens)
    msgs = [{"role": "user", "content": base_doc +
             "\n\nQuestion 1: what is 17 * 23? Just the number."}]
    print(json.dumps({"turn": 1, "kind": "cold_base",
                      **stream_chat(url, key, model, msgs)}), flush=True)

    msgs.append({"role": "assistant", "content": "391"})
    msgs.append({"role": "user", "content": "Question 2: what is 31 * 47? Just the number."})
    print(json.dumps({"turn": 2, "kind": "cached_prefix",
                      **stream_chat(url, key, model, msgs)}), flush=True)

    msgs.append({"role": "assistant", "content": "1457"})
    msgs.append({"role": "user", "content": ("Question 3: In one short sentence, what topic does "
                                             "the technical log above cover?")})
    print(json.dumps({"turn": 3, "kind": "cached_prefix_longer_answer",
                      **stream_chat(url, key, model, msgs, max_tokens=200)}), flush=True)

if __name__ == "__main__":
    main()
