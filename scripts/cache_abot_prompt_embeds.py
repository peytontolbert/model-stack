#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute ABot-World prompt embeddings into the lazy T5 cache.")
    parser.add_argument("--repo-path", default="/data/repositories/ABot-World")
    parser.add_argument("--tokenizer-path", default="/arxiv/models/acvlab--ABot-World-0-5B-LF/google/umt5-xxl/")
    parser.add_argument("--encoder-pth", default="/arxiv/models/acvlab--ABot-World-0-5B-LF/models_t5_umt5-xxl-enc-bf16.pth")
    parser.add_argument("--prompt", action="append", required=True, help="Prompt to encode. Repeat for batch prompts.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cpu", help="T5 execution device: cpu or cuda:0. Output cache is stored on CPU.")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    repo = Path(args.repo_path)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ["ABOT_WORLD_LAZY_T5"] = "1"
    os.environ["ABOT_WORLD_T5_DEVICE"] = args.device
    if args.cache_dir:
        os.environ["ABOT_WORLD_PROMPT_CACHE_DIR"] = args.cache_dir

    from utils.wan_wrapper import WanTextEncoder

    encoder = WanTextEncoder(tokenizer_path=args.tokenizer_path, encoder_pth_path=args.encoder_pth)
    cache_path = encoder._cache_path(args.prompt)
    existed_before = cache_path.is_file()
    out = encoder(args.prompt, device=None)
    payload = {
        "status": "ok",
        "cache_path": str(cache_path),
        "existed_before": existed_before,
        "prompt_count": len(args.prompt),
        "prompt_embeds_shape": list(out["prompt_embeds"].shape),
        "prompt_embeds_dtype": str(out["prompt_embeds"].dtype),
        "t5_loaded": encoder.text_encoder is not None,
    }
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) ++ "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
