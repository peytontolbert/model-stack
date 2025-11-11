# citation regex check, pytest test runner

from typing import Dict, List, Any
import subprocess
from ..code_graph import CodeGraph
import os


def verify_with_tests(g: CodeGraph, module: str, *, repo_root: str, env: Dict[str, str]) -> bool:
    """Run mapped tests for a module if available; return True if all selected tests pass.

    Uses CodeGraph's tests mapping (best-effort). If no tests mapped, returns True as a soft pass.
    """
    nodes = g.pytest_nodes_by_module.get(module, [])
    # Also include module-level mapping
    if not nodes:
        nodes = g.tests_for_module(module)
    if not nodes:
        return True  # no tests to run; accept
    # Build pytest command
    cmd = ["pytest", "-q"]
    # Prefer node ids if present (file::Class::test), else module paths
    for n in nodes[:8]:  # cap for speed
        cmd.append(n)
    try:
        proc = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
        return int(proc.returncode) == 0
    except Exception:
        return False


def extract_citations(text: str) -> List[str]:
    try:
        import re
        rx = re.compile(r"(?:path:\s*)?([A-Za-z0-9_./\-]+?\.\w+):(\d+)(?:-(\d+))?")
        return [m.group(1) for m in rx.finditer(text or "")]
    except Exception:
        return []


def has_citations(s: str, per_para: bool) -> bool:
    import re as _re
    rx = _re.compile(r"(?:path:\s*)?[A-Za-z0-9_./-]+?\.\w+:\d+(?:-\d+)?")
    if not rx.search(s):
        return False
    if per_para:
        paras = [p.strip() for p in s.split("\n\n") if p.strip()]
        return all(rx.search(p) for p in paras)
    return True


def normalize_citations(text: str) -> List[tuple[str, int, int]]:
    """Return normalized (path, a, b) triples for all citations in text.

    If end is missing, b==a. Paths are returned as-is (caller can make them repo-relative).
    """
    try:
        import re
        rx = re.compile(r"(?:path:\s*)?([A-Za-z0-9_./\-]+?\.\w+):(\d+)(?:-(\d+))?")
        out: List[tuple[str, int, int]] = []
        for m in rx.finditer(text or ""):
            p = m.group(1)
            try:
                a = int(m.group(2) or "0")
            except Exception:
                a = 0
            try:
                b = int(m.group(3) or a)
            except Exception:
                b = a
            out.append((p, min(a, b), max(a, b)))
        return out
    except Exception:
        return []


def extract_typed_facts(text: str, g: CodeGraph) -> List[dict]:
    """Best-effort, lightweight fact extraction from answer text.

    Emits dicts like {kind, symbol, span} where:
      - kind in {"signature", "mentions", "returns"}
      - symbol is best-effort FQN or name
      - span optionally refers to a cited (path,a-b) if the symbol can be mapped later
    """
    out: List[dict] = []
    try:
        import re
        # signature claims: show name and parenthesized params
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", text or ""):
            name = m.group(1)
            params = m.group(2)
            # map to FQN if possible (short name match)
            fqns = [s.fqn for s in g.find_symbol(name)] if name else []
            out.append({
                "kind": "signature",
                "symbol": fqns[0] if fqns else name,
                "params": params,
            })
        # explicit returns claims (very naive)
        for m in re.finditer(r"returns\s+([A-Za-z_][A-Za-z0-9_.]*)", (text or "").lower()):
            typ = m.group(1)
            out.append({"kind": "returns", "type": typ})
        # code mentions in backticks
        for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_\.]+)`", text or ""):
            sym = m.group(1)
            out.append({"kind": "mentions", "symbol": sym})
    except Exception:
        return out
    return out


def extract_symbol_mentions(text: str) -> List[str]:
    """Collect probable symbol names from text using simple patterns."""
    names: List[str] = []
    try:
        import re
        # Backticked identifiers or dotted names
        for m in re.finditer(r"`([A-Za-z_][A-Za-z0-9_\.]+)`", text or ""):
            names.append(m.group(1).split(".")[-1])
        # Bare function/class name followed by '('
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", text or ""):
            names.append(m.group(1))
    except Exception:
        return names
    # De-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _abs_paths_like(root: str, files: List[str]) -> List[str]:
    out: List[str] = []
    for rel in files or []:
        if not rel:
            continue
        if os.path.isabs(rel):
            out.append(os.path.abspath(rel))
        else:
            out.append(os.path.abspath(os.path.join(root, rel)))
    return out


def _to_repo_relative(root: str, abs_path: str) -> str:
    try:
        root_abs = os.path.abspath(root or "")
        p_abs = os.path.abspath(abs_path or "")
        if p_abs.startswith(root_abs):
            rel = p_abs[len(root_abs):].lstrip(os.sep)
            return rel.replace("\\", "/")
        return abs_path.replace("\\", "/")
    except Exception:
        return abs_path.replace("\\", "/")


def _merge_overlapping_spans(spans: List[tuple[str, int, int]]) -> List[tuple[str, int, int]]:
    by_file: Dict[str, List[tuple[int, int]]] = {}
    for p, a, b in spans:
        by_file.setdefault(p, []).append((min(a, b), max(a, b)))
    merged: List[tuple[str, int, int]] = []
    for p, ranges in by_file.items():
        ranges.sort()
        cur_a, cur_b = ranges[0]
        for a, b in ranges[1:]:
            if a <= cur_b + 1:
                cur_b = max(cur_b, b)
            else:
                merged.append((p, cur_a, cur_b))
                cur_a, cur_b = a, b
        merged.append((p, cur_a, cur_b))
    return merged


def resolve_claim_spans(text: str, g: CodeGraph, idx: object, files_scope: List[str]) -> List[tuple[str, int, int]]:
    """Resolve spans for claims using anchor-first mapping, then similarity fallback."""
    try:
        scope_abs = set(_abs_paths_like(g.root, files_scope))
    except Exception:
        scope_abs = set()
    spans: List[tuple[str, int, int]] = []
    names = extract_symbol_mentions(text or "")
    # Anchor-first by symbol lookup
    for name in names:
        try:
            cands = [s for s in g.find_symbol(name) if (not scope_abs or os.path.abspath(s.file) in scope_abs)]
        except Exception:
            cands = []
        if cands:
            # Prefer shortest span (tighter definition)
            s = min(cands, key=lambda z: (int(getattr(z, "end_line", getattr(z, "line", 1))) - int(getattr(z, "line", 1))), default=None)
            if s:
                rel = _to_repo_relative(g.root, s.file)
                spans.append((rel, int(getattr(s, "line", 1)), int(getattr(s, "end_line", getattr(s, "line", 1)))))
    # Similarity fallback via SymbolIndex if nothing found
    if not spans and idx is not None:
        try:
            # Late import type to avoid hard dependency
            top = getattr(idx, "query")(text or "", 3)
            for it in (top or [])[:3]:
                rel = it.get("rel") or _to_repo_relative(g.root, it.get("file", ""))
                a = int(it.get("line", 1)); b = int(it.get("end_line", a))
                spans.append((rel, min(a, b), max(a, b)))
        except Exception:
            pass
    # Merge overlaps and de-dup
    seen = set()
    spans = _merge_overlapping_spans([s for s in spans if s not in seen and not seen.add(s)])
    return spans


def validate_spans(cites: List[tuple[str, int, int]], g: CodeGraph) -> List[tuple[str, int, int]]:
    """Clamp to file bounds and ensure 1-based inclusive ranges."""
    ok: List[tuple[str, int, int]] = []
    for p, a, b in cites:
        a0 = max(1, int(a)); b0 = max(a0, int(b))
        # Best-effort cap to file length
        try:
            abs_fp = os.path.join(g.root, p) if not os.path.isabs(p) else p
            with open(abs_fp, "r", encoding="utf-8", errors="ignore") as fh:
                n = sum(1 for _ in fh)
            b0 = min(b0, int(n))
        except Exception:
            pass
        ok.append((p, a0, b0))
    return ok


def _file_from_citation_path(g: CodeGraph, files_scope: List[str], cite_path: str) -> str | None:
    """Resolve a citation path to a repo-relative path within scope, allowing basename-only matches."""
    if not cite_path:
        return None
    # If absolute, try to make it repo-relative
    if os.path.isabs(cite_path):
        return _to_repo_relative(g.root, cite_path)
    # Direct match in scope
    scope_rel = [f if not os.path.isabs(f) else _to_repo_relative(g.root, f) for f in (files_scope or [])]
    if cite_path in scope_rel:
        return cite_path
    # Basename match in scope
    base = os.path.basename(cite_path)
    cands = [r for r in scope_rel if os.path.basename(r) == base]
    if len(cands) == 1:
        return cands[0]
    # As a fallback, accept the path as-is (caller will attempt to open)
    return cite_path


def _nearest_symbol_span_for_line(g: CodeGraph, rel_path: str, a: int, b: int) -> tuple[int, int] | None:
    """Find a symbol span in file that overlaps [a,b], else nearest by distance to midpoint."""
    try:
        abs_fp = os.path.join(g.root, rel_path) if not os.path.isabs(rel_path) else rel_path
        # Filter symbols for this file
        syms: List[Any] = []
        try:
            syms_all = list(getattr(g, "symbols_by_fqn", {}).values())
        except Exception:
            syms_all = []
        for s in syms_all or []:
            try:
                if os.path.abspath(getattr(s, "file", "")) == os.path.abspath(abs_fp):
                    syms.append(s)
            except Exception:
                continue
        if not syms:
            return None
        a0, b0 = int(a), int(b)
        # Overlap first
        overlaps = [s for s in syms if not (int(s.end_line) < a0 or int(s.line) > b0)]
        if overlaps:
            s = min(overlaps, key=lambda z: (int(z.end_line) - int(z.line)))
            return int(s.line), int(s.end_line)
        # Nearest by midpoint distance
        mid = (a0 + b0) // 2
        s = min(syms, key=lambda z: min(abs(mid - int(z.line)), abs(mid - int(z.end_line))))
        return int(s.line), int(s.end_line)
    except Exception:
        return None


def repair_citations(text: str, g: CodeGraph, files_scope: List[str]) -> List[tuple[str, int, int]]:
    """Map cited path:a-b to nearest anchored symbol spans within scope, de-dup and validate."""
    cites = normalize_citations(text or "")
    if not cites:
        return []
    repaired: List[tuple[str, int, int]] = []
    for (p_raw, a, b) in cites:
        try:
            rel = _file_from_citation_path(g, files_scope, p_raw)
            if not rel:
                continue
            span = _nearest_symbol_span_for_line(g, rel, a, b)
            if not span:
                # Keep original range if no symbol found
                repaired.append((rel, min(int(a), int(b)), max(int(a), int(b))))
                continue
            sa, sb = span
            repaired.append((rel, int(sa), int(sb)))
        except Exception:
            continue
    # Validate and merge
    repaired = validate_spans(repaired, g)
    # De-dup
    seen = set()
    uniq = []
    for r in repaired:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return _merge_overlapping_spans(uniq)


