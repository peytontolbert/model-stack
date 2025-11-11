from __future__ import annotations

import sys
from pathlib import Path
import subprocess

EX_DIR = Path("/data/transformer_10/examples/program_conditioned_adapter")
SMOKE_PROG = Path("/data/transformer_10/examples/program_conditioned_adapter/smoke_repo")
ART_DIR = EX_DIR / "examples" / "python_repo_grounded_planning" / "artifacts" / "smoke_planning"

PG_BACKEND = "examples.program_conditioned_adapter.examples.scripts.python_repo_graph:PythonRepoGraph"
EMIT_MOD = "examples.program_conditioned_adapter.examples.python_repo_grounded_planning.emit_planning_knowledge"


def _run(argv: list[str]) -> int:
    print("[run]", " ".join(argv))
    return subprocess.call(argv)


def main() -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)

    rc = _run([
        sys.executable, "-m", "examples.program_conditioned_adapter.build",
        "--program", str(SMOKE_PROG),
        "--pg-backend", PG_BACKEND,
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--adapters-dir", str(ART_DIR),
        "--embed-dim", "256",
        "--include-text",
        "--text-max-bytes", "20000",
        "--graph-prop-hops", "1",
        "--graph-prop-damp", "0.85",
        "--code-recall-preset",
        "--auto-rank",
        "--rank-min", "4",
        "--rank-max", "16",
        "--seed", "0",
        "--verbose",
    ])
    if rc != 0:
        sys.exit(rc)

    rc = _run([
        sys.executable, "-m", EMIT_MOD,
        "--program", str(SMOKE_PROG),
        "--pg-backend", PG_BACKEND,
        "--out-dir", str(ART_DIR),
        "--max-modules", "200",
        "--verbose",
    ])
    if rc != 0:
        sys.exit(rc)

    prompt = (
        "Produce a grounded step-by-step plan to add a new CLI subcommand that lists all modules "
        "and their public functions. Include exact files or artifact URIs and line ranges to modify/create, "
        "and cite entities/windows."
    )

    rc = _run([
        sys.executable, "-m", "examples.program_conditioned_adapter.run",
        "--program", str(SMOKE_PROG),
        "--pg-backend", PG_BACKEND,
        "--model", "meta-llama/Llama-3.1-8B-Instruct",
        "--adapters-dir", str(ART_DIR),
        "--prompt", prompt,
        "--of-sources", "question",
        "--pack-context",
        "--pack-mode", "windows",
        "--context-tokens", "1200",
        "--require-citations",
        "--structured",
        "--citations-enforce",
        "--retrieval-policy", "sim:0.45,struct:0.35,plan:0.20",
        "--retrieval-temp", "0.7",
        "--alpha-warmup",
        "--adapter-aware-decoding",
        "--verbose",
    ])
    sys.exit(rc)


if __name__ == "__main__":
    main()


