import os
import json
from typing import List, Dict


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local snapshot path")
    p.add_argument("--cache-dir", default="/data/transformer_10/checkpoints")
    p.add_argument("--samples", type=int, default=0, help="Sample count from a default suite (0 uses all)")
    p.add_argument("--backend", default="local", choices=["hf","local","pure"], help="Which local tokenizer backend to compare against HF")
    args = p.parse_args()

    from model.hf_snapshot import ensure_snapshot
    snap = ensure_snapshot(args.model, args.cache_dir)

    # HF reference tokenizer
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError("Install transformers to run the comparison script") from e
    hf_tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, cache_dir=args.cache_dir)
    if getattr(hf_tok, "pad_token", None) is None:
        try:
            hf_tok.pad_token = hf_tok.eos_token
        except Exception:
            pass

    # Local tokenizer
    if args.backend == "pure":
        from data.tokenizer import PureLlamaTokenizer
        local_tok = PureLlamaTokenizer(snap)
    else:
        from data.tokenizer import LocalLlamaTokenizer
        local_tok = LocalLlamaTokenizer(snap)

    tests: List[str] = [
        "Hello, world",
        "The quick brown fox jumps over 13 lazy dogs.",
        "  leading and  multiple   spaces  ",
        "tabs\tand\nnewlines\n",
        "emoji 😀😃😄 😁🖖🏽",
        "中文測試 — mixed-width punctuation、！",
        "العَرَبِيَّةُ لغةٌ جميلةٌ",
        "कृत्रिम बुद्धिमत्ता से नमस्ते",
        "geschichten999",
        "<|begin_of_text|> special token check <|end_of_text|>",
    ]
    if args.samples and args.samples > 0:
        tests = tests[: int(args.samples)]

    results: Dict[str, Dict] = {}
    ok = 0
    for t in tests:
        hf_ids = list(hf_tok.encode(t, add_special_tokens=False))
        loc_ids = local_tok.encode(t)
        enc_eq = (hf_ids == loc_ids)
        # Decode both from HF ids for stable baseline
        hf_dec = hf_tok.decode(hf_ids)
        loc_dec_from_hf = local_tok.decode(hf_ids)
        # Decode both from local ids
        loc_dec = local_tok.decode(loc_ids)
        hf_dec_from_loc = hf_tok.decode(loc_ids)
        results[t] = {
            "enc_equal": enc_eq,
            "hf_ids": hf_ids[:64],
            "loc_ids": loc_ids[:64],
            "hf_dec": hf_dec,
            "loc_dec_from_hf": loc_dec_from_hf,
            "loc_dec": loc_dec,
            "hf_dec_from_loc": hf_dec_from_loc,
        }
        ok += int(enc_eq)

    out = {
        "snapshot": snap,
        "hf_special": {
            "bos": getattr(hf_tok, "bos_token_id", None),
            "eos": getattr(hf_tok, "eos_token_id", None),
            "unk": getattr(hf_tok, "unk_token_id", None),
            "pad": getattr(hf_tok, "pad_token_id", None),
        },
        "local_special": {
            "bos": local_tok.bos_token_id,
            "eos": local_tok.eos_token_id,
            "unk": local_tok.unk_token_id,
            "pad": local_tok.pad_token_id,
        },
        "tests": results,
        "enc_equal_count": ok,
        "enc_total": len(tests),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


