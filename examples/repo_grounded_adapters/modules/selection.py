# question-aware selection, zoom-by-symbol, function-first candidate collection, name-match helpers (lift from run_repo_adapter.py)
import os
import re
from typing import Dict, List, Optional, Tuple

from examples.repo_grounded_adapters.code_graph import CodeGraph


def _re_escape(s: str) -> str:
    try:
        import re as _re

        return _re.escape(s)
    except Exception:
        return s


def modules_from_symbols(repo_root: str, seeds: List[str], *, radius: int = 1, top_k: int = 8, ignore: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    g = CodeGraph.load_or_build(repo_root, ignore=[s for s in (ignore or []) if s])
    modules_set: Dict[str, bool] = {}
    files_set: Dict[str, bool] = {}
    for s in seeds:
        try:
            for sym in g.find_symbol(s):
                modules_set[sym.module] = True
                files_set[os.path.relpath(sym.file, g.root)] = True
        except Exception:
            continue
    if radius > 0 and modules_set:
        cur = list(modules_set.keys())
        seen = set(cur)
        for _ in range(max(0, int(radius))):
            nxt: List[str] = []
            for m in cur:
                for dep in g.module_imports.get(m, []):
                    if dep not in seen:
                        seen.add(dep)
                        nxt.append(dep)
            cur = nxt
        for m in seen:
            modules_set[m] = True
            f = g.file_for_module(m)
            if f:
                files_set[os.path.relpath(f, g.root)] = True
    modules = sorted(modules_set.keys())[: top_k]
    files = sorted(files_set.keys())[: max(top_k, 8)]
    return modules, files


def question_aware_modules_and_files(repo_root: str, prompt: str, *, top_k: int = 8, ignore: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    g = CodeGraph.load_or_build(repo_root, ignore=[s for s in (ignore or []) if s])
    try:
        toks = [t for t in re.findall(r"[A-Za-z0-9_]+", (prompt or "").lower()) if len(t) >= 3]
    except Exception:
        toks = []
    score_by_file: Dict[str, int] = {}
    for t in toks[:10]:
        try:
            for (rel, _ln, _txt) in g.search_refs(_re_escape(t)):
                score_by_file[rel] = score_by_file.get(rel, 0) + 1
        except Exception:
            continue
    ranked_files = [fp for fp, _ in sorted(score_by_file.items(), key=lambda x: x[1], reverse=True)]
    files = ranked_files[: max(top_k, 8)]
    modules_set: Dict[str, bool] = {}
    for f in files:
        m = g.module_for_file(f)
        if m:
            modules_set[m] = True
    modules = sorted(modules_set.keys())[: top_k]
    return modules, files


def name_matches_prompt(name: Optional[str], prompt_tokens: set) -> bool:
    if not name:
        return False
    n = name.lower()
    if n in " ".join(sorted(prompt_tokens)):
        return True
    parts = [p for p in n.replace("_", " ").split() if p]
    if not parts:
        return False
    inter = sum(1 for p in parts if p in prompt_tokens)
    need = 2 if len(parts) >= 2 else 1
    cat = "".join(parts)
    return bool(inter >= need or cat in prompt_tokens)


def prompt_token_set(prompt_q: str) -> set:
    try:
        import re as _re
        toks = _re.findall(r"[A-Za-z0-9_]+", (prompt_q or "").lower())
        return set(toks)
    except Exception:
        return set()
