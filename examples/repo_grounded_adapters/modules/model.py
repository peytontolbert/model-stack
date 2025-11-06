import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import torch


""" 

def _ensure_snapshot(model_id: str, cache_dir: str) -> str:
    # If local dir with config.json, use directly
    if os.path.isdir(model_id) and os.path.isfile(os.path.join(model_id, "config.json")):
        return model_id
    # Try existing HF snapshot in cache_dir
    org_name = model_id.strip().split("/")[-2:]
    if len(org_name) == 2:
        org, name = org_name
        dir1 = os.path.join(cache_dir, f"models--{org}--{name}", "snapshots")
        cands = []
        if os.path.isdir(dir1):
            cands.extend([os.path.join(dir1, d) for d in os.listdir(dir1)])
        cands = [p for p in cands if os.path.isfile(os.path.join(p, "config.json"))]
        if cands:
            cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return cands[0]
    # Otherwise, try to download
    from huggingface_hub import snapshot_download  # type: ignore
    return snapshot_download(repo_id=model_id, cache_dir=cache_dir)

"""