from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# Resolve program_conditioned_adapter directory directly from this file
EX_DIR = Path(__file__).resolve().parents[2]
SMOKE_REPO = EX_DIR / "smoke_repo"
# Backend class name (implemented in this example package)
PG_BACKEND = "examples.program_conditioned_adapter.examples.scripts.python_repo_graph:PythonRepoGraph"
# Program configuration loader
from .program_config import load_program_config  # noqa: E402


def _run(cmd: list[str]) -> int:
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    cfg = load_program_config(str(SMOKE_REPO))
    adapters_dir = cfg.paths.adapters_dir
    adapters_dir.mkdir(parents=True, exist_ok=True)
    # Resolve backend module path if a bare class name is provided
    pg_backend = PG_BACKEND
    if ":" not in pg_backend:
        pg_backend = cfg.pg_backend
    # Emit consolidated repository knowledge (entities/edges/anchors) for grounding
    knowledge_path = cfg.paths.knowledge_path
    _run([
        sys.executable,
        "-m",
        "examples.program_conditioned_adapter.examples.python_repo_grounded_qa.emit_repository_knowledge",
        str(SMOKE_REPO),
        str(knowledge_path),
        pg_backend,
    ])
    # Build adapters and caches for the smoke repo (PCA-agnostic; PG injected via --pg-backend)
    rc = _run([
        sys.executable,
        "-m",
        "examples.program_conditioned_adapter.build",
        "--sources", str(SMOKE_REPO),
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--adapters-dir", str(adapters_dir),
        "--embed-dim", "256",
        "--include-text",
        "--text-max-bytes", "20000",
        "--pg-backend", pg_backend,
        "--contracts-require-citations",
        "--contracts-retrieval-policy", str(cfg.contracts.retrieval_policy),
        "--contracts-retrieval-temp", str(cfg.contracts.retrieval_temp),
        "--init-program-state",
        "--program-state-path", str(cfg.paths.program_state_path),
        "--seed", "0",
        "--verbose",
    ])
    if rc != 0:
        sys.exit(rc)
    # Run grounded QA (selection and stamping via ProgramGraph plugin)
    prompt = "Where is the add function defined and how is it used? Provide citations."
    rc = _run([
        sys.executable,
        "-m",
        "examples.program_conditioned_adapter.run",
        "--sources", str(SMOKE_REPO),
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--adapters-dir", str(adapters_dir),
        "--prompt", prompt,
        "--of-sources", "question",
        "--pack-context",
        "--pack-mode", "windows",
        "--context-tokens", "800",
        "--require-citations",
        "--structured",
        "--citations-enforce",
        "--pg-backend", pg_backend,
        "--retrieval-policy", str(cfg.contracts.retrieval_policy),
        "--retrieval-temp", str(cfg.contracts.retrieval_temp),
        "--program-state", str(cfg.paths.program_state_path),
        "--delta-cap", "0.05",
        "--verbose",
    ])
    sys.exit(rc)


if __name__ == "__main__":
    main()


