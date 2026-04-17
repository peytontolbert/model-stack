from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_trainer_checkpoint_path_uses_runtime_helpers() -> None:
    trainer = _read("train/trainer.py")
    assert "runtime_embedding(" in trainer
    assert "apply_native_norm(" in trainer
    assert "runtime_linear(" in trainer
    assert "self.model.embed(" not in trainer
    assert "self.model.norm(" not in trainer
    assert "self.model.lm_head(" not in trainer


def test_interpret_helpers_use_runtime_bindings() -> None:
    residual = _read("interpret/analysis/residual.py")
    assert "runtime_embedding(" in residual
    assert "model.embed(" not in residual

    attn_weights = _read("interpret/attn/weights.py")
    assert "runtime_embedding(" in attn_weights
    assert "apply_native_norm(" in attn_weights
    assert "runtime_linear(" in attn_weights
    assert "runtime_split_heads(" in attn_weights
    assert "runtime_apply_rotary(" in attn_weights
    assert "model.embed(" not in attn_weights
    assert "block.n1(" not in attn_weights
    assert "attn.w_q(" not in attn_weights
    assert "attn.w_k(" not in attn_weights

    head_patching = _read("interpret/causal/head_patching.py")
    assert "runtime_linear(" in head_patching
    assert "runtime_split_heads(" in head_patching
    assert "runtime_apply_rotary(" in head_patching
    assert "runtime_head_output_projection(" in head_patching
    assert "attn.w_q(" not in head_patching
    assert "attn.w_k(" not in head_patching
    assert "attn.w_v(" not in head_patching
    assert "attn.w_o(" not in head_patching

    ablate = _read("interpret/attn/ablate.py")
    assert "runtime_linear(" in ablate
    assert "runtime_split_heads(" in ablate
    assert "runtime_apply_rotary(" in ablate
    assert "runtime_head_output_projection(" in ablate
    assert "attn.w_q(" not in ablate
    assert "attn.w_k(" not in ablate
    assert "attn.w_v(" not in ablate
    assert "attn.w_o(" not in ablate
