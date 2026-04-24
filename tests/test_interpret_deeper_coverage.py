from __future__ import annotations

import types
from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import interpret.model_adapter as model_adapter_module
import runtime.seq2seq as runtime_seq2seq_module
from interpret.activation_cache import ActivationCache, CaptureSpec
from interpret.causal.patching import causal_trace_restore_fraction, output_patching
from interpret.causal.sae_patch import sae_feature_mask
from interpret.features.index import FeatureSlice, flatten_features, targets_from_tokens
from interpret.model_adapter import (
    ModelAdapter,
    ModelInputs,
    _clone_for_capture,
    _expanded_kv_heads,
    _prepare_attention_scores,
    _same_tensor_identity,
    _stack_or_default,
    eager_attention_forward,
    get_model_adapter,
    mlp_forward,
    patched_embedding_output,
    resolve_model_score,
)
from interpret.probes.dataset import ProbeFeatureSlice, build_probe_dataset
from runtime.attention_modules import EagerAttention
from runtime.causal import CausalLM
from runtime.encoder import EncoderModel
from runtime.seq2seq import EncoderDecoderLM
from specs.config import ModelConfig
from tensor.mlp import MLP


def _causal_model() -> CausalLM:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = CausalLM(cfg)
    model.eval()
    return model


def _encoder_model() -> EncoderModel:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = EncoderModel(cfg)
    model.eval()
    return model


def _seq2seq_model() -> EncoderDecoderLM:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=2, d_ff=32, vocab_size=32, attn_impl="eager")
    model = EncoderDecoderLM(cfg)
    model.eval()
    return model


class _LegacyCausal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.embed(input_ids)


class _Unsupported(nn.Module):
    pass


class _IdentityBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Identity()
        self.mlp = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _CausalNoMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.blocks = nn.ModuleList([_IdentityBlock()])
        self.norm = nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.embed(input_ids)


class _FakeDecoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn_block = types.SimpleNamespace(attn=None)
        self.cross_block = types.SimpleNamespace(cross=nn.Identity())

    def _ensure_self_attn(self) -> None:
        return None


class _LegacySeq2Seq(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc_embed = nn.Embedding(8, 4)
        self.dec_embed = nn.Embedding(8, 4)
        self.encoder = nn.ModuleList([nn.Identity()])
        self.decoder = nn.ModuleList([_FakeDecoderBlock()])
        self.enc_norm = nn.Identity()
        self.dec_norm = nn.Identity()

    def forward(
        self,
        enc_input_ids: torch.Tensor,
        dec_input_ids: torch.Tensor,
        enc_padding_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.dec_embed(dec_input_ids)


class _BadHandle:
    def remove(self) -> None:
        raise RuntimeError("expected remove failure")


class _DummyCache:
    def __init__(self, k_old: torch.Tensor, v_old: torch.Tensor) -> None:
        self.k_old = k_old
        self.v_old = v_old
        self.appended: tuple[torch.Tensor, torch.Tensor] | None = None

    def read(self, start: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_old, self.v_old

    def length(self) -> int:
        return int(self.k_old.shape[2])

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        self.appended = (k.clone(), v.clone())


def test_model_adapter_internal_helpers_and_selection_branches() -> None:
    x = torch.randn(1, 2, 3, requires_grad=True)
    assert _clone_for_capture(None, detach=True, move_to_cpu=False) is None
    captured = _clone_for_capture(x, detach=False, move_to_cpu=False)
    assert captured is not None
    assert captured.data_ptr() != x.data_ptr()

    if torch.cuda.is_available():
        gpu_x = x.to("cuda")
        moved = _clone_for_capture(gpu_x, detach=True, move_to_cpu=True)
        assert moved is not None
        assert moved.device.type == "cpu"

    assert _stack_or_default(None, "encoder") == "encoder"
    assert _stack_or_default(None, "causal") == "causal"
    assert _stack_or_default(None, "encoder_decoder") == "decoder"
    assert _same_tensor_identity(x, x)
    assert not _same_tensor_identity(x, x.clone())
    assert not _same_tensor_identity(x, None)

    legacy = ModelAdapter(_LegacyCausal())
    assert legacy.kind == "causal"
    with pytest.raises(TypeError):
        ModelAdapter(_Unsupported())

    seq2seq = _seq2seq_model()
    adapter = ModelAdapter(seq2seq)
    inputs = ModelInputs.encoder_decoder(torch.randint(0, 8, (1, 4)), torch.randint(0, 8, (1, 3)))

    assert adapter.output_module() is not None
    assert adapter.final_norm(stack="encoder") is seq2seq.enc_norm
    assert adapter.final_norm(stack="decoder") is seq2seq.dec_norm
    assert adapter.embedding_module(stack="encoder") is seq2seq.enc_embed
    assert adapter.embedding_module(stack="decoder") is seq2seq.dec_embed
    assert torch.equal(adapter.sequence_tokens(inputs, stack="encoder"), inputs.enc_input_ids)
    assert torch.equal(adapter.sequence_tokens(inputs, stack="decoder"), inputs.dec_input_ids)
    assert adapter.embedding_runtime_module() is runtime_seq2seq_module
    assert adapter.embedding_output(inputs, stack="decoder").shape[-1] == seq2seq.cfg.d_model

    assert len(adapter.block_targets(stack="encoder")) == seq2seq.cfg.n_layers
    assert len(adapter.block_targets(stack="decoder")) == seq2seq.cfg.n_layers * 2
    assert len(adapter.attention_targets(stack="encoder")) == seq2seq.cfg.n_layers
    assert len(adapter.attention_targets(stack="decoder", kind="cross")) == seq2seq.cfg.n_layers
    assert adapter.block_target(0, stack="decoder", kind="self").kind == "self"
    assert adapter.block_target(0, stack="decoder", kind="cross").kind == "cross"
    assert adapter.attention_target(0, stack="decoder", kind="cross").kind == "cross"

    with pytest.raises(KeyError):
        adapter.attention_target(99)
    with pytest.raises(KeyError):
        adapter.block_target(99)
    assert adapter.mlp_targets(stack="encoder")
    assert adapter.mlp_targets(stack="decoder", kind="missing") == []

    with pytest.raises(TypeError):
        ModelAdapter(_CausalNoMLP()).mlp_module(0)

    legacy_seq2seq = ModelAdapter(_LegacySeq2Seq())
    assert legacy_seq2seq.kind == "encoder_decoder"
    assert legacy_seq2seq.attention_targets(stack="decoder", kind="self") == []


def test_model_adapter_forward_validation_and_score_resolution_branches() -> None:
    with pytest.raises(ValueError):
        ModelAdapter(_causal_model()).forward(ModelInputs())
    with pytest.raises(ValueError):
        ModelAdapter(_encoder_model()).forward(ModelInputs())
    with pytest.raises(ValueError):
        ModelAdapter(_seq2seq_model()).forward(ModelInputs())

    causal = _causal_model()
    with pytest.raises(ValueError):
        resolve_model_score(causal, torch.randn(2, 3))

    score, token_id, feature_id = resolve_model_score(
        causal,
        torch.randn(1, 2, causal.cfg.vocab_size),
        score_fn=lambda outputs: 1.25,
    )
    assert score.ndim == 0
    assert token_id is None
    assert feature_id is None

    with pytest.raises(ValueError):
        resolve_model_score(
            causal,
            torch.randn(1, 2, causal.cfg.vocab_size),
            score_fn=lambda outputs: outputs[0],
        )

    encoder = _encoder_model()
    outputs = encoder(torch.randint(0, encoder.cfg.vocab_size, (1, 4)))
    score, token_id, feature_id = resolve_model_score(encoder, outputs, position=-1)
    assert score.ndim == 0
    assert token_id is None
    assert isinstance(feature_id, int)


def test_patched_embedding_output_shared_weight_second_call_branch() -> None:
    model = _seq2seq_model()
    model.dec_embed = model.enc_embed
    adapter = get_model_adapter(model)
    captured: list[torch.Tensor] = []

    with patched_embedding_output(adapter, stack="decoder", capture=lambda out: captured.append(out.detach().clone())):
        first = runtime_seq2seq_module.runtime_embedding(model.enc_embed.weight, torch.tensor([[1]]), None)
        second = runtime_seq2seq_module.runtime_embedding(model.dec_embed.weight, torch.tensor([[2]]), None)

    assert len(captured) == 1
    assert torch.equal(captured[0], second)
    assert not torch.equal(first, second)


def test_probe_dataset_and_feature_index_edge_branches() -> None:
    cache = ActivationCache()
    a = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    b = torch.arange(2 * 4 * 2, dtype=torch.float32).view(2, 4, 2)
    cache.store("a", a, CaptureSpec(move_to_cpu=False))
    cache.store("b2", torch.arange(2 * 3, dtype=torch.float32).view(2, 3), CaptureSpec(move_to_cpu=False))
    cache.store("b3", b, CaptureSpec(move_to_cpu=False))

    flat = flatten_features(cache, [FeatureSlice("missing"), FeatureSlice("b2", batch_index=1), FeatureSlice("b3", batch_index=1, time_slice=slice(0, 1))])
    assert flat.shape == (1, 5)

    with pytest.raises(ValueError):
        targets_from_tokens(torch.arange(5))
    assert targets_from_tokens(torch.arange(8).view(2, 4), batch_index=1, position=1).tolist() == [6]
    assert targets_from_tokens(torch.arange(8).view(2, 4), position=99).numel() == 0

    with pytest.raises(ValueError):
        build_probe_dataset(cache, [])
    with pytest.raises(KeyError):
        build_probe_dataset(cache, [ProbeFeatureSlice("missing")], input_ids=torch.arange(8).view(2, 4))
    with pytest.raises(ValueError):
        build_probe_dataset(cache, [ProbeFeatureSlice("a")])
    with pytest.raises(ValueError):
        build_probe_dataset(cache, [ProbeFeatureSlice("a")], input_ids=torch.arange(6).view(2, 3))
    with pytest.raises(ValueError):
        build_probe_dataset(cache, [ProbeFeatureSlice("a")], input_ids=torch.arange(8).view(2, 4), mask=torch.ones(1, 4))
    with pytest.raises(ValueError):
        build_probe_dataset(cache, [ProbeFeatureSlice("a")], targets=torch.ones(2, 4, 1))

    empty = build_probe_dataset(
        cache,
        [ProbeFeatureSlice("a")],
        input_ids=torch.arange(8).view(2, 4),
        mask=torch.zeros(2, 4, dtype=torch.long),
    )
    assert empty.x.shape == (0, 0)
    assert empty.y.numel() == 0

    ds = build_probe_dataset(
        cache,
        [ProbeFeatureSlice("a"), ProbeFeatureSlice("a", time_offset=-1)],
        input_ids=torch.arange(8).view(2, 4),
        mask=torch.ones(2, 4, dtype=torch.long),
        sample_size=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert ds.x.shape[0] == 2
    assert ds.y.shape[0] == 2


def test_output_patching_restore_fraction_and_sae_patch_edge_branches(monkeypatch) -> None:
    linear = nn.Linear(2, 2)
    monkeypatch.setattr(linear, "register_forward_hook", lambda hook: _BadHandle())
    with output_patching(nn.Sequential(linear), {"0": torch.ones(1, 2), "missing": torch.zeros(1, 2)}):
        _ = linear(torch.zeros(1, 2))

    encoder = _encoder_model()
    clean = torch.randint(0, encoder.cfg.vocab_size, (1, 4))
    corrupted = clean.clone()
    corrupted[0, -1] = (corrupted[0, -1] + 1) % encoder.cfg.vocab_size
    encoder.train()
    frac = causal_trace_restore_fraction(
        encoder,
        clean_input_ids=clean,
        corrupted_input_ids=corrupted,
        patch_points=["blocks.0", "missing"],
        score_fn=lambda outputs: outputs[0, -1, 0],
    )
    assert frac.shape == (1,)
    assert encoder.training

    from interpret.features.sae import SparseAutoencoder

    model = _causal_model()
    input_ids = torch.randint(0, model.cfg.vocab_size, (1, 5))
    sae = SparseAutoencoder(model.cfg.d_model, 6)
    base = model(input_ids)
    with sae_feature_mask(model, 0, sae):
        noop = model(input_ids)
    assert torch.allclose(base, noop)
    with sae_feature_mask(model, 0, sae, keep_codes=[0, 1]):
        kept = model(input_ids)
    with sae_feature_mask(model, 0, sae, boost_codes=[0], boost_factor=2.0):
        boosted = model(input_ids)
    with sae_feature_mask(model, 0, sae, drop_codes=[0], time_slice=slice(-1, None)):
        sliced = model(input_ids)
    assert kept.shape == boosted.shape == sliced.shape == base.shape
    assert not torch.allclose(base[:, -1], sliced[:, -1])

    block = model.blocks[0]
    monkeypatch.setattr(block, "register_forward_hook", lambda hook: _BadHandle())
    with sae_feature_mask(model, 0, sae, drop_codes=[0]):
        pass


def test_attention_and_mlp_helper_branches(monkeypatch) -> None:
    cfg = ModelConfig(d_model=16, n_heads=4, n_layers=1, d_ff=32, vocab_size=32, attn_impl="eager")

    attn = EagerAttention(cfg, n_kv_heads=2, use_rope=True)
    qh = torch.randn(1, attn.n_heads, 2, attn.head_dim)
    kh_all = torch.randn(1, attn.n_kv_heads, 2, attn.head_dim)
    repeated_k, repeated_v = _expanded_kv_heads(attn, kh_all, kh_all)
    assert repeated_k.shape[1] == attn.n_heads
    assert repeated_v.shape[1] == attn.n_heads
    logits, probs = _prepare_attention_scores(attn, qh, kh_all, None)
    assert logits.shape == probs.shape == (1, attn.n_heads, 2, 2)

    x = torch.randn(1, 2, attn.d_model)

    def _proj(width: int):
        return lambda tensor: torch.ones(tensor.shape[0], tensor.shape[1], width, device=tensor.device, dtype=tensor.dtype)

    attn.w_q.runtime_linear = _proj(attn.n_heads * attn.head_dim)  # type: ignore[attr-defined]
    attn.w_k.runtime_linear = _proj(attn.n_kv_heads * attn.head_dim)  # type: ignore[attr-defined]
    attn.w_v.runtime_linear = _proj(attn.n_kv_heads * attn.head_dim)  # type: ignore[attr-defined]
    attn.w_o.runtime_linear = _proj(attn.d_model)  # type: ignore[attr-defined]
    cache = _DummyCache(
        torch.zeros(1, attn.n_kv_heads, 1, attn.head_dim),
        torch.zeros(1, attn.n_kv_heads, 1, attn.head_dim),
    )
    captured = {}
    out = eager_attention_forward(attn, x, None, None, None, cache, capture=lambda snapshot: captured.setdefault("snapshot", snapshot))
    assert out.shape == (1, 2, attn.d_model)
    assert cache.appended is not None
    assert captured["snapshot"].probs is not None

    attn_cross = EagerAttention(cfg, n_kv_heads=2, use_rope=False)
    attn_cross.w_q.runtime_linear = _proj(attn_cross.n_heads * attn_cross.head_dim)  # type: ignore[attr-defined]
    attn_cross.w_k.runtime_linear = _proj(attn_cross.n_kv_heads * attn_cross.head_dim)  # type: ignore[attr-defined]
    attn_cross.w_v.runtime_linear = _proj(attn_cross.n_kv_heads * attn_cross.head_dim)  # type: ignore[attr-defined]
    out_cross = eager_attention_forward(attn_cross, x, x, x, None, None)
    assert out_cross.shape == (1, 2, attn_cross.d_model)

    attn_packed = EagerAttention(cfg, use_rope=False)
    attn_packed.eval()
    monkeypatch.setattr(attn_packed, "_packed_backend", lambda tensor: "fake")
    monkeypatch.setattr(
        attn_packed,
        "_ensure_packed_qkv",
        lambda backend, reference: (
            torch.empty(0),
            torch.empty(0),
            (attn_packed.n_heads * attn_packed.head_dim, attn_packed.n_kv_heads * attn_packed.head_dim, attn_packed.n_kv_heads * attn_packed.head_dim),
        ),
    )
    monkeypatch.setattr(
        model_adapter_module,
        "runtime_qkv_packed_heads_projection",
        lambda tensor, packed_weight, packed_bias, q_size, k_size, v_size, q_heads, kv_heads, backend: (
            torch.ones(tensor.shape[0], q_heads, tensor.shape[1], attn_packed.head_dim, dtype=tensor.dtype),
            torch.ones(tensor.shape[0], kv_heads, tensor.shape[1], attn_packed.head_dim, dtype=tensor.dtype),
            torch.ones(tensor.shape[0], kv_heads, tensor.shape[1], attn_packed.head_dim, dtype=tensor.dtype),
        ),
    )
    monkeypatch.setattr(model_adapter_module, "runtime_attention", lambda q, k, v, attn_mask=None, is_causal=False, scale=1.0: q)
    monkeypatch.setattr(attn_packed, "_ensure_packed_output", lambda backend, reference: (torch.eye(attn_packed.d_model), None))
    monkeypatch.setattr(
        model_adapter_module,
        "runtime_head_output_projection",
        lambda head_out, weight, bias, backend=None: torch.zeros(head_out.shape[0], head_out.shape[2], attn_packed.d_model, dtype=head_out.dtype),
    )
    out_packed = eager_attention_forward(attn_packed, x, None, None, None, None)
    assert out_packed.shape == (1, 2, attn_packed.d_model)

    x_grad = torch.randn(1, 3, 8, requires_grad=True)
    gated_fallback = MLP(8, 4, activation="swiglu")
    gated_fallback.activation_name = "mystery"
    snap: dict[str, object] = {}
    out_gated = mlp_forward(gated_fallback, x_grad, capture=lambda snapshot: snap.setdefault("gated", snapshot), patch_mid=lambda mid: mid + 1.0, keep_grad=True)
    assert out_gated.shape == (1, 3, 8)
    assert snap["gated"].mlp_mid is not None

    reglu = MLP(8, 4, activation="reglu")
    out_reglu = mlp_forward(reglu, x_grad, keep_grad=True)
    assert out_reglu.shape == (1, 3, 8)

    gated_leaky = MLP(8, 4, activation="swiglu")
    gated_leaky.activation_name = "leaky_relu_0p5_squared"
    out_gated_leaky = mlp_forward(gated_leaky, x_grad, keep_grad=True)
    assert out_gated_leaky.shape == (1, 3, 8)

    dense = MLP(8, 4, activation="gelu")
    out_dense = mlp_forward(dense, x_grad, keep_grad=True)
    assert out_dense.shape == (1, 3, 8)
