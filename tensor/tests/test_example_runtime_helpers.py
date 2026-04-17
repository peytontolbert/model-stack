from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_debug_examples_route_local_execution_through_runtime_helpers() -> None:
    debug_attention = _read("examples/debug_attention.py")
    assert "runtime_embedding(" in debug_attention
    assert "apply_native_norm(" in debug_attention
    assert "runtime_linear(" in debug_attention
    assert "local.embed(" not in debug_attention
    assert "local.blocks[0].n1(" not in debug_attention
    assert "local_attn.w_q(" not in debug_attention
    assert "local_attn.w_k(" not in debug_attention
    assert "local_attn.w_v(" not in debug_attention

    debug_single_layer = _read("examples/debug_single_layer.py")
    assert "runtime_embedding(" in debug_single_layer
    assert "apply_native_norm(" in debug_single_layer
    assert "runtime_linear(" in debug_single_layer
    assert "local.embed(" not in debug_single_layer
    assert "local.norm(" not in debug_single_layer
    assert "local.lm_head(" not in debug_single_layer

    debug_parity = _read("examples/debug_parity.py")
    assert "runtime_embedding(" in debug_parity
    assert "apply_native_norm(" in debug_parity
    assert "local.embed(" not in debug_parity
    assert "local.blocks[0].n1(" not in debug_parity


def test_trace_llama_parity_scripts_use_runtime_embedding() -> None:
    repo_trace = _read("examples/repo_grounded_adapters/scripts/trace_llama_parity.py")
    assert "runtime_embedding(" in repo_trace
    assert "local_input_embeds = runtime_embedding(" in repo_trace
    assert "local.embed(" not in repo_trace
    assert "input_embeds=local.embed(" not in repo_trace

    program_trace = _read("examples/program_conditioned_adapter/scripts/trace_llama_parity.py")
    assert "runtime_embedding(" in program_trace
    assert "local_input_embeds = runtime_embedding(" in program_trace
    assert "local.embed(" not in program_trace
    assert "input_embeds=local.embed(" not in program_trace
