#!/usr/bin/env python
"""Probe hf-mirror availability of SeedVR2 weights, then start the 3B download.

Run inside the flashvsr env (has huggingface_hub). Safe to re-run: downloads
resume. Prints PROBE lines first so the log tells us availability fast even
if the big download then takes an hour.
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import hf_hub_download, snapshot_download

CACHE = os.path.expanduser("~/weights/seedvr2")
os.makedirs(CACHE, exist_ok=True)

for repo in ("ByteDance-Seed/SeedVR2-3B", "ByteDance-Seed/SeedVR2-7B"):
    try:
        p = hf_hub_download(repo_id=repo, filename="README.md",
                            local_dir=os.path.join(CACHE, "probe"))
        print(f"PROBE OK   {repo}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"PROBE FAIL {repo}: {type(e).__name__} {str(e)[:200]}",
              flush=True)

print("PROBE done — starting 3B snapshot download (resumable)", flush=True)
try:
    path = snapshot_download(repo_id="ByteDance-Seed/SeedVR2-3B",
                             local_dir=os.path.join(CACHE, "SeedVR2-3B"),
                             max_workers=4)
    print(f"SNAPSHOT OK {path}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"SNAPSHOT FAIL: {type(e).__name__} {str(e)[:300]}", flush=True)
