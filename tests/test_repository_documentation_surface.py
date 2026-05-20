from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CRITICAL_PACKAGE_READMES = [
    "attn/README.md",
    "dist/README.md",
    "docs/README.md",
    "experiments/README.md",
    "governance/README.md",
    "registry/README.md",
    "runtime/README.md",
    "safety/README.md",
    "tools/README.md",
]


def test_critical_non_example_packages_have_substantive_readmes() -> None:
    for relpath in CRITICAL_PACKAGE_READMES:
        text = (ROOT / relpath).read_text(encoding="utf-8").strip()
        assert len(text.splitlines()) >= 20, relpath
        assert text.startswith("# "), relpath
