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