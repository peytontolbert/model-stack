from __future__ import annotations

import sys
import subprocess
from pathlib import Path

# Resolve program_conditioned_adapter directory directly from this file
EX_DIR = Path(__file__).resolve().parents[2]
SMOKE_REPO = EX_DIR / "smoke_repo"
PG_BACKEND = "examples.program_conditioned_adapter.examples.scripts.python_repo_graph:PythonRepoGraph"
# Program configuration
from .program_config import load_program_config  # noqa: E402


def _run(cmd: list[str]) -> int:
    print("[run]", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    cfg = load_program_config(str(SMOKE_REPO))
    adapters_dir = cfg.paths.adapters_dir
    adapters_dir.mkdir(parents=True, exist_ok=True)
    # Emit planning knowledge artifacts (entities/edges/artifacts-derived plans)
    rc = _run([
        sys.executable,
        str(Path(__file__).resolve().parent / "emit_planning_knowledge.py"),
        "--program", str(SMOKE_REPO),
        "--pg-backend", (cfg.pg_backend if ":" in cfg.pg_backend else PG_BACKEND),
        "--out-dir", str(cfg.paths.knowledge_dir),
        "--max-modules", "200",
        "--verbose",
    ])
    if rc != 0:
        sys.exit(rc)

    # Build adapters and caches for the smoke repo (PCA-agnostic; PG via --pg-backend)
    rc = _run([
        sys.executable,
        str(EX_DIR / "build.py"),
        "--sources", str(SMOKE_REPO),
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--adapters-dir", str(adapters_dir),
        "--embed-dim", "256",
        "--include-text",
        "--text-max-bytes", "20000",
        "--pg-backend", cfg.pg_backend if ":" in cfg.pg_backend else PG_BACKEND,
        "--graph-prop-hops", "2",
        "--graph-prop-damp", "0.85",
        "--contracts-require-citations",
        "--contracts-retrieval-policy", str(cfg.contracts.retrieval_policy),
        "--contracts-retrieval-temp", str(cfg.contracts.retrieval_temp),
        "--kbann-priors",
        "--knowledge-preset",
        "--auto-rank",
        "--rank-min", "8",
        "--rank-max", "16",
        "--init-program-state",
        "--program-state-path", str(cfg.paths.program_state_path),
        "--seed", "0",
        "--verbose",
    ])
    if rc != 0:
        sys.exit(rc)

    # Grounded planning prompt (facts-seeded steps + citations)
    prompt = (
        "Produce a grounded step-by-step plan to add a new CLI subcommand "
        "that lists all modules and their public functions. Include specific paths and "
        "line ranges to modify or create, referencing code entities with citations."
    )

    # Run grounded planning (structured mode; citations enforced)
    rc = _run([
        sys.executable,
        str(EX_DIR / "run.py"),
        "--sources", str(SMOKE_REPO),
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--adapters-dir", str(adapters_dir),
        "--prompt", prompt,
        "--of-sources", "question",
        "--pack-context",
        "--pack-mode", "windows",
        "--context-tokens", "1200",
        "--require-citations",
        "--structured",
        "--citations-enforce",
        "--pg-backend", cfg.pg_backend if ":" in cfg.pg_backend else PG_BACKEND,
        "--retrieval-policy", cfg.contracts.retrieval_policy,
        "--retrieval-temp", str(cfg.contracts.retrieval_temp),
        "--alpha-warmup",
        "--adapter-aware-decoding",
        "--program-state", str(cfg.paths.program_state_path),
        "--delta-cap", "0.05",
        "--verbose",
    ])
    sys.exit(rc)


if __name__ == "__main__":
    main()


