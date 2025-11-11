from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProgramContracts:
    require_citations: bool = True
    citations_per_paragraph: bool = False
    retrieval_policy: str = "sim:0.6,struct:0.4"
    retrieval_temp: float = 0.7


@dataclass(frozen=True)
class ProgramPaths:
    adapters_dir: Path
    knowledge_path: Path
    program_state_path: Optional[Path] = None


@dataclass(frozen=True)
class ProgramConfig:
    program_id: str
    pg_backend: str  # dotted path module:Class
    paths: ProgramPaths
    contracts: ProgramContracts


def _detect_pg_backend(base_dir: Path) -> str:
    pkg = "examples.program_conditioned_adapter.examples.python_repo_grounded_qa"
    if (base_dir / "python_repo_graph.py").exists():
        return f"{pkg}.python_repo_graph:PythonRepoGraph"
    if (base_dir / "repo_graph.py").exists():
        return f"{pkg}.repo_graph:PythonRepoGraph"
    # Fallback to python_repo_graph by name
    return f"{pkg}.python_repo_graph:PythonRepoGraph"


def load_program_config(repo_root: str) -> ProgramConfig:
    repo = Path(repo_root).resolve()
    ex_dir = Path(__file__).resolve().parents[2]
    example_dir = Path(__file__).resolve().parent
    adapters_dir = ex_dir / "artifacts" / "smoke_base"
    knowledge_path = adapters_dir / "repository_knowledge.json"
    pg_backend = _detect_pg_backend(example_dir)
    program_id = repo.name or "repo"
    return ProgramConfig(
        program_id=program_id,
        pg_backend=pg_backend,
        paths=ProgramPaths(
            adapters_dir=adapters_dir,
            knowledge_path=knowledge_path,
            program_state_path=adapters_dir / ".program_state.json",
        ),
        contracts=ProgramContracts(),
    )


