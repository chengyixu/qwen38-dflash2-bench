#!/usr/bin/env python3
"""Convert a DFlash2 drafter to pre-quantized MLX 4-bit using the same path as
the oMLX fork's runtime (nn.quantize + verify-linears), then save to disk.

Usage: convert_dflash_4bit.py SRC_DST_DIR OUT_DIR
"""
import sys, shutil, json
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])

RES = "/Applications/oMLX.app/Contents/Resources"
import os
os.environ.setdefault("PYTHONPATH", "")
import importlib.util

def _add(p):
    if p not in sys.path:
        sys.path.insert(0, p)

_add(RES)
_add(f"{RES}/Python/framework-mlx-base/lib/python3.11/site-packages")
_add(f"{RES}/Python/cpython-3.11/lib/python3.11/site-packages")

from dflash_mlx.runtime.loading import load_draft_bundle  # noqa: E402

model, meta = load_draft_bundle(str(SRC), lazy=False, draft_quant="w4a16:gs64")
print("loaded:", meta["draft_quant"], "dtype:", meta["draft_load_dtype"])

OUT.mkdir(parents=True, exist_ok=True)

# Save weights: serialize quantized parameter tree to a single safetensors file
import mlx.core as mx
from mlx.utils import tree_flatten

flat = dict(tree_flatten(model.parameters()))
arrays = {k: v for k, v in flat.items() if isinstance(v, mx.array)}
mx.save_safetensors(
    str(OUT / "model.safetensors"),
    arrays,
    metadata={"format": "mlx"},
)
print("tensors saved:", len(arrays))

# Copy config/tokenizer from source
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
          "merges.txt", "generation_config.json", "chat_template.jinja"]:
    s = SRC / f
    if s.exists():
        shutil.copy(s, OUT / f)

print("saved to", OUT)
