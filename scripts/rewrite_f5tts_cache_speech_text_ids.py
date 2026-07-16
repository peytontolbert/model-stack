#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from distill_f5tts_12_to_4_q4 import DEFAULT_VOCAB, build_model, text_to_ids
from tts_text_normalizer import normalize_f5tts_speech_text


def iter_rows(metadata_path: Path):
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite cached F5TTS text IDs to app-normalized speech text.")
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    source_cache = Path(args.source_cache)
    output_cache = Path(args.output_cache)
    output_samples = output_cache / "samples"
    output_samples.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = build_model(Path(args.vocab), device)
    model.eval()

    rewritten = 0
    total = 0
    metadata_out = output_cache / "metadata.jsonl"
    metadata_out.unlink(missing_ok=True)
    with metadata_out.open("w", encoding="utf-8") as handle:
        for row in iter_rows(source_cache / "metadata.jsonl"):
            if row.get("event") == "done":
                continue
            if "path" not in row:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                continue

            total += 1
            source_path = source_cache / str(row["path"])
            target_path = output_cache / str(row["path"])
            target_path.parent.mkdir(parents=True, exist_ok=True)
            payload = torch.load(source_path, map_location="cpu")

            ref_text = str(payload.get("ref_text") or row.get("ref_text") or "").strip()
            gen_text = str(payload.get("gen_text") or row.get("gen_text") or "").strip()
            speech_gen_text = normalize_f5tts_speech_text(gen_text)
            speech_full_text = (ref_text + " " + speech_gen_text).strip()
            if speech_gen_text != gen_text:
                rewritten += 1

            payload["text"] = [speech_full_text]
            payload["gen_text"] = gen_text
            payload["speech_gen_text"] = speech_gen_text
            payload["text_ids"] = text_to_ids(model, [speech_full_text], device).detach().cpu().to(torch.int32)
            torch.save(payload, target_path)

            out_row = dict(row)
            out_row["speech_gen_text"] = speech_gen_text
            out_row["normalized_f5tts_text"] = bool(speech_gen_text != gen_text)
            handle.write(json.dumps(out_row, sort_keys=True) + "\n")

        handle.write(
            json.dumps(
                {
                    "event": "done",
                    "source_cache": str(source_cache),
                    "rows": total,
                    "rewritten_rows": rewritten,
                },
                sort_keys=True,
            )
            + "\n"
        )

    config_src = source_cache / "cache_config.json"
    if config_src.exists():
        shutil.copyfile(config_src, output_cache / "source_cache_config.json")
    (output_cache / "rewrite_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": total, "rewritten_rows": rewritten, "output_cache": str(output_cache)}, indent=2))


if __name__ == "__main__":
    main()
