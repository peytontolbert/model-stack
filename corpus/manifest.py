from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class ShardInfo:
    path: str
    num_tokens: int
    num_sequences: Optional[int] = None


@dataclass
class CorpusManifest:
    version: int
    tokenizer: Dict[str, Any]
    shards: List[ShardInfo]
    total_tokens: int

    @staticmethod
    def load(path: str | Path) -> "CorpusManifest":
        with Path(path).open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
        shards = [ShardInfo(**s) for s in obj.get("shards", [])]
        return CorpusManifest(
            version=int(obj["version"]),
            tokenizer=dict(obj.get("tokenizer", {})),
            shards=shards,
            total_tokens=int(obj.get("total_tokens", 0)),
        )

    def dump(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "version": int(self.version),
                    "tokenizer": self.tokenizer,
                    "total_tokens": int(self.total_tokens),
                    "shards": [asdict(s) for s in self.shards],
                },
                fh,
                indent=2,
                sort_keys=True,
            )


__all__ = ["ShardInfo", "CorpusManifest"]

# corpus/manifest.py
from dataclasses import dataclass
@dataclass
class Shard:
    uri: str
    bytes: int
    license: str
    language: str
    split: str  # train/val/test
# Emits manifest.jsonl consumed by data
