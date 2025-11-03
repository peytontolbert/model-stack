from typing import List
from examples.repo_grounded_adapters.code_graph import CodeGraph

def build_prompts_for_module(g: CodeGraph, module: str, max_q: int = 3) -> List[str]:
    """Construct a few simple, verifiable prompts for a module using symbol names and doc headers.

    Prefers questions that can be answered via local context/citations.
    """
    prompts: List[str] = []
    # Symbols defined in the module
    defs = list(g.defs_in(module) or [])
    # Ask for an explanation of the module
    prompts.append(f"Explain the key functions/classes in {module}. Cite path:line for each claim.")
    # Ask about up to two concrete defs
    for fqn in defs[:2]:
        name = fqn.split(".")[-1]
        prompts.append(f"What does `{name}` do in {module}? Show signature and cite path:line.")
    # Unique and bounded
    uniq: List[str] = []
    for p in prompts:
        if p not in uniq:
            uniq.append(p)
    return uniq[: max(1, int(max_q))]
