#!/usr/bin/env python3
"""Verify that migration documentation covers the repository tree.

This script proves documentation readiness before migration by checking:

1. The required migration docs exist.
2. The research README indexes the required migration docs.
3. The module target-state matrix has valid exact-file rows.
4. Every non-`__init__` Python file in the repository is covered either:
   - exactly by the module matrix, or
   - by an explicit subtree rule from the full repository scope closure.
5. The documented subtree counts still match the repository tree.

Exit code is non-zero on any failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "research"
README = DOCS_DIR / "README.md"
MATRIX = DOCS_DIR / "transformer10-module-target-state-matrix.md"
SCOPE = DOCS_DIR / "transformer10-full-repository-scope-closure.md"

VALID_MATRIX_STATES = {
    "cuda_kernel",
    "cpp_runtime",
    "python_binding",
    "python_reference",
    "remove_or_merge",
    "defer",
}

REQUIRED_DOCS = [
    "transformer10-module-target-state-matrix.md",
    "transformer10-cpp-model-stack-api-spec.md",
    "transformer10-complete-tensor-math-function-inventory.md",
    "transformer10-training-backward-cuda-spec.md",
    "transformer10-data-checkpoint-tokenizer-cpp-spec.md",
    "transformer10-serving-engine-cpp-spec.md",
    "transformer10-autotune-eval-benchmark-spec.md",
    "transformer10-compression-quantization-runtime-spec.md",
    "transformer10-noncore-systems-targets.md",
    "transformer10-end-to-end-cpp-cuda-migration-runbook.md",
    "transformer10-full-repository-scope-closure.md",
    "transformer10-pytorch-decommission-checklist.md",
]


@dataclasses.dataclass(frozen=True)
class ScopeRule:
    path: str
    classification: str
    expected_count: int

    def matches(self, relpath: str) -> bool:
        if self.path.endswith("/**"):
            prefix = self.path[:-2]
            return relpath.startswith(prefix)
        return relpath == self.path


SCOPE_RULES = [
    ScopeRule("autotune/algorithms/**", "core_migration_target", 5),
    ScopeRule("autotune/bench/**", "core_migration_target", 5),
    ScopeRule("autotune/schedulers/**", "core_migration_target", 2),
    ScopeRule("autotune/storage/**", "core_migration_target", 1),
    ScopeRule("dist/parallel/**", "core_migration_target", 2),
    ScopeRule("dist/strategy/**", "python_reference", 3),
    ScopeRule("interpret/analysis/**", "python_binding", 3),
    ScopeRule("interpret/attn/**", "python_binding", 4),
    ScopeRule("interpret/attribution/**", "python_binding", 3),
    ScopeRule("interpret/causal/**", "python_binding", 5),
    ScopeRule("interpret/features/**", "python_binding", 5),
    ScopeRule("interpret/importance/**", "python_binding", 1),
    ScopeRule("interpret/metrics/**", "python_binding", 2),
    ScopeRule("interpret/neuron/**", "python_binding", 2),
    ScopeRule("interpret/probes/**", "python_binding", 1),
    ScopeRule("interpret/search/**", "python_binding", 1),
    ScopeRule("rag/components/**", "defer", 5),
    ScopeRule("rl/algorithms/**", "defer", 2),
    ScopeRule("tensor/masking/**", "core_migration_target", 1),
    ScopeRule("tensor/numerics/**", "core_migration_target", 2),
    ScopeRule("tensor/tests/**", "python_reference", 29),
    ScopeRule("blocks/examples/**", "python_reference", 1),
    ScopeRule("example.py", "python_reference", 1),
    ScopeRule("tools/**", "python_reference", 1),
    ScopeRule("examples/00_tiny_lm/**", "python_reference", 1),
    ScopeRule("examples/01_sft_dialog/**", "python_reference", 1),
    ScopeRule("examples/02_int8_export/**", "python_reference", 1),
    ScopeRule("examples/04_eval_coding/**", "python_reference", 1),
    ScopeRule("examples/05_repo_adapters/**", "python_reference", 1),
    ScopeRule("examples/06_repo_fast_weights/**", "python_reference", 1),
    ScopeRule("examples/07_tensor_numerics/**", "python_reference", 1),
    ScopeRule("examples/08_model_generate/**", "python_reference", 1),
    ScopeRule("examples/09_interpret_logit_lens/**", "python_reference", 1),
    ScopeRule("examples/10_autotune_search/**", "python_reference", 1),
    ScopeRule("examples/11_compress_quantize/**", "python_reference", 1),
    ScopeRule("examples/12_data_tokenize_shard/**", "python_reference", 1),
    ScopeRule("examples/debug_attention.py", "python_reference", 1),
    ScopeRule("examples/debug_parity.py", "python_reference", 1),
    ScopeRule("examples/debug_single_layer.py", "python_reference", 1),
    ScopeRule("examples/repo_grounded_adapters/**", "defer", 30),
    ScopeRule("examples/program_conditioned_adapter/**", "defer", 225),
    ScopeRule("examples/be_great/**", "defer", 9),
    ScopeRule("other_repos/ThunderKittens/**", "external_reference_repo", 94),
    ScopeRule("other_repos/cuda-kernels/**", "external_reference_repo", 1),
    ScopeRule("other_repos/extension-cpp/**", "external_reference_repo", 5),
    ScopeRule("other_repos/flash-attention/**", "external_reference_repo", 192),
    ScopeRule("other_repos/good-kernels/**", "external_reference_repo", 12),
    ScopeRule("other_repos/tiny-cuda-nn/**", "external_reference_repo", 9),
    ScopeRule("other_repos/tinygrad/**", "external_reference_repo", 658),
]


def all_python_files() -> list[str]:
    files: list[str] = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if path.name == "__init__.py":
            continue
        if rel.startswith("docs/"):
            continue
        if "/.git/" in rel or rel.startswith(".git/"):
            continue
        files.append(rel)
    return sorted(files)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_matrix_rows(text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    pattern = re.compile(r"^\|\s*`([^`]+\.py)`\s*\|\s*`([^`]+)`\s*\|", re.MULTILINE)
    for relpath, state in pattern.findall(text):
        rows[relpath] = state
    return rows


def verify_required_docs() -> list[str]:
    failures: list[str] = []
    readme_text = read_text(README)
    for name in REQUIRED_DOCS:
        path = DOCS_DIR / name
        if not path.exists():
            failures.append(f"missing required doc: {name}")
        if f"`{name}`" not in readme_text:
            failures.append(f"README is missing doc entry: {name}")
    return failures


def verify_matrix(repo_files: Iterable[str], matrix_rows: dict[str, str]) -> list[str]:
    failures: list[str] = []
    repo_set = set(repo_files)
    for relpath, state in sorted(matrix_rows.items()):
        if state not in VALID_MATRIX_STATES:
            failures.append(f"invalid matrix state for {relpath}: {state}")
        if relpath not in repo_set:
            failures.append(f"stale matrix entry, file missing from repo: {relpath}")
    return failures


def classify_by_scope_rule(relpath: str) -> ScopeRule | None:
    matches = [rule for rule in SCOPE_RULES if rule.matches(relpath)]
    if not matches:
        return None
    if len(matches) > 1:
        joined = ", ".join(rule.path for rule in matches)
        raise RuntimeError(f"ambiguous scope rules for {relpath}: {joined}")
    return matches[0]


def verify_scope_rules(repo_files: Iterable[str], exact_matrix_files: set[str]) -> tuple[list[str], dict[str, int]]:
    failures: list[str] = []
    actual_counts = {rule.path: 0 for rule in SCOPE_RULES}

    for relpath in repo_files:
        rule = classify_by_scope_rule(relpath)
        if rule is not None:
            actual_counts[rule.path] += 1

    for relpath in repo_files:
        if relpath in exact_matrix_files:
            continue
        rule = classify_by_scope_rule(relpath)
        if rule is None:
            failures.append(f"uncovered python file: {relpath}")
            continue
    for rule in SCOPE_RULES:
        actual = actual_counts[rule.path]
        if actual != rule.expected_count:
            failures.append(
                f"scope rule count drift for {rule.path}: expected {rule.expected_count}, found {actual}"
            )

    return failures, actual_counts


def print_summary(
    repo_files: list[str],
    matrix_rows: dict[str, str],
    scope_counts: dict[str, int],
    failures: list[str],
) -> None:
    exact_count = len(matrix_rows)
    scope_count = sum(scope_counts.values())
    scope_matches = {
        relpath
        for relpath in repo_files
        if classify_by_scope_rule(relpath) is not None
    }
    unique_covered = len(set(matrix_rows) | scope_matches)
    overlap = exact_count + scope_count - unique_covered
    print("Migration Documentation Coverage Report")
    print(f"repo_python_files: {len(repo_files)}")
    print(f"matrix_exact_files: {exact_count}")
    print(f"scope_rule_files: {scope_count}")
    print(f"covered_unique_files: {unique_covered}")
    print(f"coverage_overlap_files: {overlap}")
    print(f"required_docs: {len(REQUIRED_DOCS)}")
    if failures:
        print(f"status: FAIL ({len(failures)} issues)")
        print()
        print("Failures:")
        for item in failures:
            print(f"- {item}")
    else:
        print("status: PASS")
        print()
        print("Coverage proof:")
        print("- every required migration doc exists and is indexed in docs/research/README.md")
        print("- every exact core product module is covered by the module target-state matrix")
        print("- every remaining Python file is covered by an explicit subtree scope rule")
        print("- documented subtree counts still match the repository tree")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    repo_files = all_python_files()
    matrix_rows = parse_matrix_rows(read_text(MATRIX))

    failures: list[str] = []
    failures.extend(verify_required_docs())
    failures.extend(verify_matrix(repo_files, matrix_rows))
    scope_failures, scope_counts = verify_scope_rules(repo_files, set(matrix_rows))
    failures.extend(scope_failures)

    print_summary(repo_files, matrix_rows, scope_counts, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
