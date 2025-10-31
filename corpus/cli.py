from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import json

from .build import build_corpus
from .manifest import CorpusManifest


def _load_tokenizer(name_or_path: Optional[str]):
    from data.tokenizer import get_tokenizer
    return get_tokenizer(name_or_path)


def cmd_build(args: argparse.Namespace) -> None:
    tok = _load_tokenizer(args.tokenizer)
    man = build_corpus(
        args.input,
        args.outdir,
        tokenizer=tok,
        shard_size_tokens=int(args.shard_size_tokens),
        apply_pii_redaction=bool(args.redact_pii),
        dedup=bool(args.dedup),
    )
    print(json.dumps({
        "total_tokens": man.total_tokens,
        "num_shards": len(man.shards),
        "outdir": str(args.outdir),
    }, indent=2))


def cmd_stats(args: argparse.Namespace) -> None:
    man = CorpusManifest.load(Path(args.manifest))
    print(json.dumps({
        "total_tokens": man.total_tokens,
        "num_shards": len(man.shards),
        "tokenizer": man.tokenizer,
    }, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m corpus.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build token shards + manifest from raw text/jsonl")
    b.add_argument("--input", required=True, help="Input directory with .txt/.jsonl (jsonl uses 'text' field)")
    b.add_argument("--outdir", required=True, help="Output directory for shards + manifest.json")
    b.add_argument("--tokenizer", default=None, help="HF name/path or None for whitespace fallback")
    b.add_argument("--shard-size-tokens", type=int, default=(1<<20))
    b.add_argument("--redact-pii", action="store_true")
    b.add_argument("--dedup", action="store_true")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("stats", help="Print basic stats from a corpus manifest")
    s.add_argument("--manifest", required=True)
    s.set_defaults(func=cmd_stats)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()


