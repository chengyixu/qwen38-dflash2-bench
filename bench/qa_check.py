#!/usr/bin/env python3
"""Quality checks: long-context needle + coding sanity. Usage: qa_check.py BASE_URL MODEL [PROMPT_TOKENS]"""
import json, sys, time, urllib.request
sys.path.insert(0, "/private/tmp/commandcode-501/-Users-wilsonxu-Models/34f4e0c2-cc8e-4c61-b935-9b45fc904635/scratchpad")
from bench_api import run_once, build_prompt, WORDS

def needle_prompt(total_tokens: int) -> str:
    """Bury 5 magic numbers in filler; ask for all five. Uses calibrated vocab."""
    n_words = int(total_tokens / 1.483)
    secrets = ["ZEBRA-7741", "LANTERN-3382", "QUARTZ-9056", "HARBOR-1173", "MEADOW-6649"]
    parts = []
    anchors = [n_words // 6, n_words // 3, n_words // 2, (2 * n_words) // 3, (5 * n_words) // 6]
    ai = 0
    for i in range(n_words):
        if ai < len(anchors) and i == anchors[ai]:
            parts.append(f"RECORD:{secrets[ai]}")
            ai += 1
        else:
            parts.append(f"{WORDS[i % len(WORDS)]} {WORDS[(i * 7 + 3) % len(WORDS)]}")
    body = " ".join(parts)
    q = ("\n\nFrom the log above, list every RECORD value you found, comma-separated, "
         "in order of appearance. Reply with only the values.")
    return "Technical log follows.\n\n" + body + q

def chat(url, model, prompt, max_tokens=200, api_key=None):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "stream": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        obj = json.load(r)
    dt = time.perf_counter() - t0
    return obj["choices"][0]["message"].get("content") or "", obj.get("usage", {}), dt

def main():
    url, model = sys.argv[1], sys.argv[2]
    ptoks = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
    api_key = sys.argv[4] if len(sys.argv) > 4 else None

    # Test 1: needle in haystack
    expected = "ZEBRA-7741, LANTERN-3382, QUARTZ-9056, HARBOR-1173, MEADOW-6649"
    prompt = needle_prompt(ptoks)
    print(f"[needle] prompt ~{ptoks} tokens...", flush=True)
    text, usage, dt = chat(url, model, prompt, max_tokens=600, api_key=api_key)
    found = sum(1 for s in expected.split(", ") if s in text)
    print(json.dumps({"test": "needle", "ctx_target": ptoks, "prompt_tokens": usage.get("prompt_tokens"),
                      "found": f"{found}/5", "pass": found == 5, "latency_s": round(dt, 1),
                      "answer_head": text[:160].replace("\n", " ")}), flush=True)

    # Test 2: coding sanity (greedy, deterministic-ish)
    code_q = ("Write a Python function `def moving_average(nums: list[float], k: int) -> list[float]` "
              "that returns the sliding-window averages with window size k. Handle k<=0 and windows "
              "larger than the input by returning []. Use an O(n) algorithm. Reply with code only.")
    print("[code] running...", flush=True)
    text2, usage2, dt2 = chat(url, model, code_q, max_tokens=400, api_key=api_key)
    ok = ("def moving_average" in text2 and "return []" in text2.replace("'", '"') or "return []" in text2)
    has_window = "k" in text2 and ("sum" in text2 or "cumsum" in text2 or "acc" in text2)
    print(json.dumps({"test": "coding", "has_signature": "def moving_average" in text2,
                      "has_guard": "return []" in text2, "plausible_on": has_window,
                      "gen_tokens": usage2.get("completion_tokens"), "latency_s": round(dt2, 1),
                      "answer_head": text2[:200].replace("\n", " ")}), flush=True)

if __name__ == "__main__":
    main()
