# citation regex check, pytest test runner

from typing import Dict, List
import subprocess
from examples.repo_grounded_adapters.code_graph import CodeGraph


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