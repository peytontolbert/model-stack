from __future__ import annotations

import argparse
import json
import os
import subprocess
from typing import Dict, List, Tuple, Optional

from examples.repo_grounded_adapters.code_graph import CodeGraph
from examples.repo_grounded_adapters.repo_conditioned_adapter import (
    build_subgraph_embedding_from_graph,
    save_npz,
    generate_lora_from_embedding,
)


def _build_prompts_for_module(g: CodeGraph, module: str, max_q: int = 3) -> List[str]:
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


def _run_enhanced_runner(
    model: str,
    adapters_npz: str,
    repo: str,
    prompt: str,
    *,
    cache_dir: Optional[str] = None,
    device_map: str = "none",
) -> Tuple[int, str]:
    """Invoke the enhanced runner to produce an answer with citations; return (exit_code, stdout)."""
    cmd = [
        os.sys.executable,
        "-m",
        "examples.repo_grounded_adapters.run_llama_with_repo_adapter_enhanced",
        "--model",
        model,
        "--adapters",
        adapters_npz,
        "--repo",
        repo,
        "--prompt",
        prompt,
        "--pack-context",
        "--require-citations",
        "--device-map",
        device_map,
    ]
    if cache_dir:
        cmd.extend(["--cache-dir", cache_dir])
    try:
        out = subprocess.run(cmd, capture_output=True, text=True)
        return int(out.returncode), (out.stdout or "")
    except Exception as e:
        return 1, f"[error] runner failed: {e}"


def _verify_with_tests(g: CodeGraph, module: str, *, repo_root: str, env: Dict[str, str]) -> bool:
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


def _extract_citations(text: str) -> List[str]:
    try:
        import re

        rx = re.compile(r"(?:path:\s*)?([A-Za-z0-9_./\-]+?\.\w+):(\d+)(?:-(\d+))?")
        return [m.group(1) for m in rx.finditer(text or "")]
    except Exception:
        return []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--adapters", required=True, help="Base adapters.npz used during generation (prior)")
    p.add_argument("--out", required=True, help="Output directory for tuned adapters and buffer")
    p.add_argument("--ignore", action="append", default=None)
    p.add_argument("--max-prompts", type=int, default=3, help="Max prompts per module")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--device-map", default="none", choices=["auto", "none"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.out)))
    os.makedirs(out_dir, exist_ok=True)
    g = CodeGraph.load_or_build(args.repo, ignore=[s for s in (args.ignore or []) if s])

    # Build Q/A buffer
    buffer_path = os.path.join(out_dir, "distill.jsonl")
    verified_modules: Dict[str, bool] = {}
    env = os.environ.copy()
    # For each module, generate a few answers and verify
    with open(buffer_path, "w", encoding="utf-8") as fh:
        for module in sorted(g.modules.keys()):
            mi = g.modules[module]
            if mi.is_test:
                continue
            prompts = _build_prompts_for_module(g, module, max_q=int(args.max_prompts))
            module_verified = False
            for q in prompts:
                if args.dry_run:
                    ans_text = "[dry-run]"
                    ok = True
                else:
                    rc, ans_text = _run_enhanced_runner(
                        args.model,
                        args.adapters,
                        args.repo,
                        q,
                        cache_dir=args.cache_dir,
                        device_map=args.device_map,
                    )
                    if rc != 0:
                        ok = False
                    else:
                        # Basic citation sanity: at least one citation into this repo
                        cites = _extract_citations(ans_text)
                        ok = bool(cites)
                        # If tests are mapped, require them to pass
                        if ok:
                            ok = _verify_with_tests(g, module, repo_root=args.repo, env=env)
                fh.write(json.dumps({
                    "module": module,
                    "prompt": q,
                    "answer": ans_text,
                    "verified": bool(ok),
                }) + "\n")
                module_verified = module_verified or bool(ok)
            if module_verified:
                verified_modules[module] = True

    # Build a tuned adapter from verified modules via subgraph embedding (lightweight alternative to gradient fine-tune)
    include_mods = sorted(list(verified_modules.keys()))
    if not include_mods:
        # Fallback: no verified modules; do nothing further
        print(json.dumps({
            "status": "no_verified_modules",
            "buffer": buffer_path,
        }))
        return

    emb = build_subgraph_embedding_from_graph(
        g,
        dim=1536,
        seed=0,
        include_modules=include_mods,
        include_files=None,
        include_text=True,
        text_max_bytes=250000,
        max_text_tokens=0,
        text_weight=0.25,
        graph_prop_hops=0,
        graph_prop_damp=0.85,
    )
    # Infer simple defaults for generation
    # Try to approximate shapes via first-layer local map in enhanced runner at runtime; here we export without shapes
    # Let the consumer use shapes inferred at load time.
    # Choose conservative defaults (rank 8, zmean gate, standard targets)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    # We do not know d_model/num_layers here without loading model config; prefer consumer to mix these subgraph adapters later.
    # Export only embedding and a placeholder adapters set with d_model/num_layers set to 0 to denote structural metadata.
    adapters = {
        "layers": [],
        "rank": 8,
        "d_model": 0,
        "targets": targets,
        "gates": [],
    }
    manifest = {
        "repo": os.path.abspath(args.repo),
        "verified_modules": include_mods,
        "buffer": buffer_path,
        "schema_version": 1,
        "note": "Embedding computed from verified modules; adapters layers left empty for mixing downstream.",
    }
    save_npz(out_dir, embedding=emb, adapters=adapters, manifest=manifest)
    print(json.dumps({
        "status": "ok",
        "buffer": buffer_path,
        "verified_modules_count": len(include_mods),
        "out_dir": out_dir,
    }))


if __name__ == "__main__":
    main()


