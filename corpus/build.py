from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional
import json
import numpy as np

from .manifest import CorpusManifest, ShardInfo
from .pii import redact_pii


def _iter_texts_from_dir(root: str | Path) -> Iterator[str]:
    root = Path(root)
    for p in sorted(root.rglob("*.txt")):
        try:
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if s:
                        yield s
        except Exception:
            continue
    for p in sorted(root.rglob("*.jsonl")):
        try:
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        s = str(obj.get("text", "")).strip()
                        if s:
                            yield s
                    except Exception:
                        continue
        except Exception:
            continue


def build_corpus(
    input_path: str | Path,
    out_dir: str | Path,
    *,
    tokenizer,
    shard_size_tokens: int = 1 << 20,
    apply_pii_redaction: bool = True,
    dedup: bool = False,
) -> CorpusManifest:
    """Tokenize input text files and write token shards + manifest.

    - input_path: directory with .txt and/or .jsonl (with {"text": ...})
    - tokenizer: object with encode(str) -> List[int] and an identifying config via .info()
    - shard_size_tokens: roll over to a new shard after this many tokens
    - apply_pii_redaction: redact simple PII prior to tokenization
    - dedup: if True, exact content-hash dedup across lines
    """
    from .dedup import dedup_exact

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    texts = _iter_texts_from_dir(input_path)
    if apply_pii_redaction:
        texts = (redact_pii(t) for t in texts)
    if dedup:
        texts = (t for (t, is_unique) in dedup_exact(texts) if is_unique)

    shard_idx = 0
    token_buf: list[int] = []
    shard_infos: list[ShardInfo] = []
    total = 0

    def _flush() -> None:
        nonlocal shard_idx, token_buf, shard_infos
        if not token_buf:
            return
        arr = np.asarray(token_buf, dtype=np.int32)
        shard_name = f"shard_{shard_idx:06d}.bin"
        shard_path = outp / shard_name
        arr.tofile(str(shard_path))
        shard_infos.append(ShardInfo(path=str(shard_path), num_tokens=int(arr.size)))
        shard_idx += 1
        token_buf = []

    for t in texts:
        ids = tokenizer.encode(t)
        token_buf.extend(int(x) for x in ids)
        total += len(ids)
        if len(token_buf) >= int(shard_size_tokens):
            _flush()

    _flush()

    man = CorpusManifest(
        version=1,
        tokenizer=tokenizer.info(),
        shards=shard_infos,
        total_tokens=int(total),
    )
    man.dump(outp / "manifest.json")
    return man


__all__ = ["build_corpus"]


