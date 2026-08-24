#!/usr/bin/env python3
"""Persistent HF snapshot downloader: retries with resume until complete."""
import sys, time
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
for attempt in range(1, 41):
    try:
        p = snapshot_download(repo, local_dir=dest, max_workers=4)
        print(f"COMPLETE {p}", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"attempt {attempt} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        time.sleep(min(10 * attempt, 60))
print("FAILED after all attempts", flush=True)
sys.exit(1)
