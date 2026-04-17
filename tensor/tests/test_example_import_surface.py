from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_examples_prefer_top_level_model_alias_and_runtime_inspect() -> None:
    assert "from model.lm import TransformerLM" not in _read("README.md")
    assert "from model.lm import TransformerLM" not in _read("example.py")

    for relpath in [
        "examples/00_tiny_lm/run.py",
        "examples/01_sft_dialog/run.py",
        "examples/02_int8_export/run.py",
        "examples/04_eval_coding/run.py",
        "examples/08_model_generate/run.py",
        "examples/09_interpret_logit_lens/run.py",
        "examples/10_autotune_search/run.py",
        "examples/11_compress_quantize/run.py",
    ]:
        source = _read(relpath)
        assert "from model import TransformerLM" in source
        assert "from model.lm import TransformerLM" not in source

    for relpath in [
        "examples/repo_grounded_adapters/build.py",
        "examples/program_conditioned_adapter/build.py",
        "examples/repo_grounded_adapters/modules/runner.py",
        "examples/repo_grounded_adapters/modules/peft.py",
        "examples/repo_grounded_adapters/modules/tune.py",
        "examples/repo_grounded_adapters/modules/runtime.py",
        "examples/repo_grounded_adapters/run.py",
        "examples/program_conditioned_adapter/examples/python_repo_grounded_qa/modules/tune.py",
        "examples/program_conditioned_adapter/modules/runner.py",
        "examples/program_conditioned_adapter/modules/peft.py",
        "examples/program_conditioned_adapter/modules/runtime.py",
    ]:
        source = _read(relpath)
        assert "from runtime.inspect import" in source
        assert "from model.inspect import" not in source
        assert "from blocks.inspect import" not in source
