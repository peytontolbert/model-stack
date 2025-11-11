from __future__ import annotations

import os
import json
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

def _ensure_path(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    pth = os.path.abspath(os.path.expanduser(os.path.expandvars(p)))
    return pth if os.path.exists(pth) else None

def load_manifest(adapters_dir: str) -> Dict:
    mf = os.path.join(adapters_dir, "manifest.json")
    try:
        return json.loads(open(mf, "r", encoding="utf-8").read())
    except Exception:
        return {}

def resolve_cache_path(manifest: Dict, key: str, default_path: str) -> str:
    try:
        p = manifest.get("caches", {}).get(key, {}).get("path")
        p = _ensure_path(p)
        if p:
            return p
    except Exception:
        pass
    return default_path

def iter_jsonl(path: str) -> Iterator[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except Exception:
                    continue
    except Exception:
        return iter(())

def load_symbol_index(adapters_dir: str) -> List[Dict]:
    mf = load_manifest(adapters_dir)
    sym_path = resolve_cache_path(mf, "symbol_index", os.path.join(adapters_dir, "symbol_index.jsonl"))
    return list(iter_jsonl(sym_path))

def load_windows_index(adapters_dir: str) -> List[Dict]:
    mf = load_manifest(adapters_dir)
    win_path = resolve_cache_path(mf, "windows_index", os.path.join(adapters_dir, "windows_index.jsonl"))
    return list(iter_jsonl(win_path))

def load_facts(adapters_dir: str) -> List[Dict]:
    mf = load_manifest(adapters_dir)
    facts_path = resolve_cache_path(mf, "facts", os.path.join(adapters_dir, "facts.jsonl"))
    return list(iter_jsonl(facts_path))

def pick_files_from_windows(program_root: str, windows: List[Dict], prompt: str, k: int = 8) -> List[str]:
    # Lightweight scoring: overlap of prompt tokens with filename and uri; prefer unique files
    import re
    toks = set([t for t in re.findall(r"[A-Za-z0-9_]+", (prompt or "").lower()) if len(t) >= 3])
    scored: List[Tuple[float, str]] = []
    seen = set()
    for w in windows:
        rel = str(w.get("path") or "").replace("\\", "/")
        if not rel:
            continue
        if rel in seen:
            continue
        base = os.path.basename(rel).lower()
        uri = str(w.get("uri") or "").lower()
        score = 0.0
        for t in toks:
            if t in base:
                score += 1.0
            if t in uri:
                score += 0.5
        if score > 0.0:
            scored.append((score, rel))
            seen.add(rel)
    scored.sort(key=lambda x: x[0], reverse=True)
    files = [rel for (_s, rel) in scored[: max(1, k)]]
    return files



