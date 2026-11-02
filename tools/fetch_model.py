#!/usr/bin/env python
"""Download the SigLIP 2 checkpoint into the project-local HF cache.

huggingface.co is unreachable from the lab network, so HF_ENDPOINT must point
at a mirror; env.sh already exports https://hf-mirror.com.
"""
import os
import sys

from huggingface_hub import snapshot_download

MODEL = sys.argv[1] if len(sys.argv) > 1 else "google/siglip2-base-patch16-384"
PATTERNS = ["*.json", "*.txt", "*.safetensors", "*.model", "tokenizer*",
            "preprocessor*", "*.md"]

print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT", "(default)"))
print("HF_HOME     =", os.environ.get("HF_HOME", "(default)"))
path = snapshot_download(MODEL, allow_patterns=PATTERNS)
print("SNAPSHOT:", path)
total = 0
for name in sorted(os.listdir(path)):
    fp = os.path.join(path, name)
    if os.path.isfile(fp):
        total += os.path.getsize(fp)
        print("  %-32s %10.1f KB" % (name, os.path.getsize(fp) / 1024))
print("total %.1f MB" % (total / 1e6))
