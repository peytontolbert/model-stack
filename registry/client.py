from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from governance.signature import sha256_file


@dataclass
class ArtifactRecord:
    id: str
    name: str
    path: str
    checksum: str
    stage: str
    metadata: Dict[str, Any]


class ArtifactRegistry:
    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.json"
        if not self._index_path.exists():
            self._write_index({"artifacts": []})

    def _read_index(self) -> Dict[str, Any]:
        with open(self._index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_index(self, data: Dict[str, Any]) -> None:
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _new_id(self, name: str, checksum: str) -> str:
        h = hashlib.sha1(f"{name}:{checksum}".encode()).hexdigest()[:12]
        return f"{name}:{h}"

    def register(self, artifact_path: str | os.PathLike[str], name: str, metadata: Optional[Dict[str, Any]] = None, stage: str = "pending") -> ArtifactRecord:
        p = Path(artifact_path)
        if not p.exists():
            raise FileNotFoundError(p)
        checksum = sha256_file(p)
        art_id = self._new_id(name, checksum)
        rec = ArtifactRecord(id=art_id, name=name, path=str(p.resolve()), checksum=checksum, stage=stage, metadata=metadata or {})
        idx = self._read_index()
        idx["artifacts"].append(asdict(rec))
        self._write_index(idx)
        return rec

    def list(self, stage: Optional[str] = None, name: Optional[str] = None) -> List[ArtifactRecord]:
        idx = self._read_index()
        out: List[ArtifactRecord] = []
        for r in idx.get("artifacts", []):
            if stage and r.get("stage") != stage:
                continue
            if name and r.get("name") != name:
                continue
            out.append(ArtifactRecord(**r))
        return out

    def get(self, art_id: str) -> Optional[ArtifactRecord]:
        idx = self._read_index()
        for r in idx.get("artifacts", []):
            if r.get("id") == art_id:
                return ArtifactRecord(**r)
        return None

    def promote(self, art_id: str, stage: str) -> Optional[ArtifactRecord]:
        idx = self._read_index()
        for r in idx.get("artifacts", []):
            if r.get("id") == art_id:
                r["stage"] = stage
                self._write_index(idx)
                return ArtifactRecord(**r)
        return None

    def verify(self, art_id: str) -> bool:
        rec = self.get(art_id)
        if not rec:
            return False
        p = Path(rec.path)
        if not p.exists():
            return False
        return sha256_file(p) == rec.checksum

    def retain_last_n(self, name: str, stage: str, n: int) -> None:
        idx = self._read_index()
        arts = [r for r in idx.get("artifacts", []) if r.get("name") == name and r.get("stage") == stage]
        if len(arts) <= n:
            return
        # Sort by insertion order (older first), drop extras
        to_keep = set(a["id"] for a in arts[-n:])
        new_list = [r for r in idx.get("artifacts", []) if not (r.get("name") == name and r.get("stage") == stage and r.get("id") not in to_keep)]
        idx["artifacts"] = new_list
        self._write_index(idx)


