from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProgramContracts:
    require_citations: bool = True
    citations_per_paragraph: bool = False
    retrieval_policy: str = "sim:0.6,struct:0.4,plan:0.2"
    retrieval_temp: float = 0.7


@dataclass(frozen=True)
class ProgramPaths:
    adapters_dir: Path
    knowledge_dir: Path
    program_state_path: Optional[Path] = None


@dataclass(frozen=True)
class ProgramConfig:
    program_id: str
    pg_backend: str  # dotted path module:Class
    paths: ProgramPaths
    contracts: ProgramContracts


def _detect_pg_backend(example_dir: Path) -> str:
    # Reuse the QA PythonRepoGraph backend by default
    return "examples.program_conditioned_adapter.examples.scripts.python_repo_graph:PythonRepoGraph"


def load_program_config(repo_root: str) -> ProgramConfig:
    repo = Path(repo_root).resolve()
    example_dir = Path(__file__).resolve().parent
    adapters_dir = example_dir / "artifacts" / "smoke_planning"
    knowledge_dir = adapters_dir  # colocate planning knowledge
    pg_backend = _detect_pg_backend(example_dir)
    program_id = repo.name or "repo"
    return ProgramConfig(
        program_id=program_id,
        pg_backend=pg_backend,
        paths=ProgramPaths(
            adapters_dir=adapters_dir,
            knowledge_dir=knowledge_dir,
            program_state_path=adapters_dir / ".program_state.json",
        ),
        contracts=ProgramContracts(),
    )


