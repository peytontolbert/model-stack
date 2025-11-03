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


def _signature_tokens(sig: Optional[str]) -> List[str]:
    if not sig:
        return []
    try:
        toks = [t for t in re.findall(r"[A-Za-z0-9_]+", sig) if len(t) >= 2]
        return toks
    except Exception:
        return []


def _doc_head_tokens(doc: Optional[str]) -> List[str]:
    if not doc:
        return []
    try:
        head = (doc or "").splitlines()[0].lower()[:200]
        return [t for t in re.findall(r"[a-z0-9_]+", head) if len(t) >= 3]
    except Exception:
        return []


def _load_self_queries(path: Optional[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not path:
        return out
    try:
        import json
        with open(path, "r", encoding="utf-8") as fh:
            for ln in fh:
                try:
                    obj = json.loads(ln)
                    sym = str(obj.get("symbol", ""))
                    qs = [str(q) for q in (obj.get("queries") or [])]
                    if sym and qs:
                        out[sym] = qs[:5]
                except Exception:
                    continue
    except Exception:
        return {}
    return out


def rerank_modules_and_files(
    repo_root: str,
    prompt: str,
    modules: List[str],
    files: List[str],
    *,
    ignore: Optional[List[str]] = None,
    self_queries_path: Optional[str] = None,
    weights: Tuple[float, float, float, float, float] = (0.35, 0.25, 0.15, 0.15, 0.10),
) -> Tuple[List[str], List[str]]:
    """Hybrid reranker over initial module/file candidates.

    weights: (w_sig, w_call, w_cov, w_doc, w_vis)
    """
    g = CodeGraph.load_or_build(repo_root, ignore=[s for s in (ignore or []) if s])
    p_tokens = prompt_token_set(prompt)
    w_sig, w_call, w_cov, w_doc, w_vis = [float(x) for x in weights]

    # Self-queries
    sq = _load_self_queries(self_queries_path)
    sq_tokens_by_sym: Dict[str, set] = {}
    for sym, qs in sq.items():
        toks = set()
        for q in qs:
            toks.update(prompt_token_set(q))
        sq_tokens_by_sym[sym] = toks

    # Pre-compute per-module features
    sig_by_mod: Dict[str, set] = {}
    doc_by_mod: Dict[str, set] = {}
    cov_by_mod: Dict[str, int] = {}
    call_neighbors: Dict[str, set] = {}
    vis_by_mod: Dict[str, float] = {}

    for fqn, s in g.symbols_by_fqn.items():
        m = s.module
        sig_by_mod.setdefault(m, set()).update(_signature_tokens(s.signature))
        doc_by_mod.setdefault(m, set()).update(_doc_head_tokens(s.doc))
        vis_by_mod[m] = 1.0 if (not m.endswith(".__init__") and not m.endswith(".tests")) else 0.8
    # coverage proxy: pytest nodes count
    for m, nodes in g.pytest_nodes_by_module.items():
        cov_by_mod[m] = len(nodes or [])
    # call neighborhoods
    for caller, callee in g.calls:
        cm = caller.rsplit(".", 1)[0] if "." in caller else caller
        call_neighbors.setdefault(cm, set()).add(callee)

    def _module_score(m: str) -> float:
        sig = sig_by_mod.get(m, set())
        doc = doc_by_mod.get(m, set())
        sig_overlap = len(sig & p_tokens) / float(max(1, len(sig))) if sig else 0.0
        doc_overlap = len(doc & p_tokens) / float(max(1, len(doc))) if doc else 0.0
        call_compat = float(len(call_neighbors.get(m, set()))) / 25.0  # crude proxy, cap later
        cov = float(min(1.0, (cov_by_mod.get(m, 0) / 5.0)))
        vis = float(vis_by_mod.get(m, 1.0))
        # self-query boost if any symbol under module matches prompt tokens
        sq_boost = 0.0
        for fqn in g.defs_in(m):
            toks = sq_tokens_by_sym.get(fqn)
            if toks and (toks & p_tokens):
                sq_boost = max(sq_boost, 0.15)
                break
        score = (
            w_sig * sig_overlap + w_call * min(1.0, call_compat) + w_cov * cov + w_doc * doc_overlap + w_vis * vis + sq_boost
        )
        return float(max(0.0, min(2.0, score)))

    mod_scored = [(m, _module_score(m)) for m in modules]
    mod_scored.sort(key=lambda x: x[1], reverse=True)
    modules_new = [m for m, _s in mod_scored][: max(1, len(modules))]

    # File scores inherit module score; break ties by filename token overlap
    file_scored: List[Tuple[str, float]] = []
    for f in files:
        m = g.module_for_file(f)
        base = 0.0
        if m:
            base = next((s for (mm, s) in mod_scored if mm == m), 0.0)
        fn_toks = set([t for t in re.findall(r"[a-z0-9_]+", os.path.basename(f).lower()) if len(t) >= 3])
        fn_overlap = len(fn_toks & p_tokens) / float(max(1, len(fn_toks))) if fn_toks else 0.0
        file_scored.append((f, float(base + 0.05 * fn_overlap)))
    file_scored.sort(key=lambda x: x[1], reverse=True)
    files_new = [f for f, _s in file_scored][: max(1, len(files))]

    return modules_new, files_new
