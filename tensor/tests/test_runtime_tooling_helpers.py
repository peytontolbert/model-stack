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
    model_adapter = _read("interpret/model_adapter.py")
    assert "patched_embedding_output(" in model_adapter
    assert "runtime_embedding" in model_adapter
    assert "runtime_linear(" in model_adapter
    assert "runtime_split_heads(" in model_adapter
    assert "runtime_apply_rotary(" in model_adapter
    assert "runtime_head_output_projection(" in model_adapter

    residual = _read("interpret/analysis/residual.py")
    assert "ActivationTracer(" in residual
    assert "get_model_adapter(" in residual
    assert "model.embed(" not in residual

    attn_weights = _read("interpret/attn/weights.py")
    assert "patched_attention(" in attn_weights
    assert "get_model_adapter(" in attn_weights
    assert "attn.w_q(" not in attn_weights
    assert "attn.w_k(" not in attn_weights

    tracer = _read("interpret/tracer.py")
    assert "patched_embedding_output(" in tracer

    direct = _read("interpret/attribution/direct.py")
    assert "patched_embedding_output(" in direct

    grad_x_input = _read("interpret/attribution/grad_x_input.py")
    assert "patched_embedding_output(" in grad_x_input

    integrated_gradients = _read("interpret/attribution/integrated_gradients.py")
    assert "patched_embedding_output(" in integrated_gradients

    occlusion = _read("interpret/attribution/occlusion.py")
    assert "patched_embedding_output(" in occlusion

    head_patching = _read("interpret/causal/head_patching.py")
    assert "get_model_adapter(" in head_patching
    assert "attn.w_q(" not in head_patching
    assert "attn.w_k(" not in head_patching
    assert "attn.w_v(" not in head_patching
    assert "attn.w_o(" not in head_patching

    ablate = _read("interpret/attn/ablate.py")
    assert "patched_attention(" in ablate
    assert "get_model_adapter(" in ablate
    assert "attn.w_q(" not in ablate
    assert "attn.w_k(" not in ablate
    assert "attn.w_v(" not in ablate
    assert "attn.w_o(" not in ablate
