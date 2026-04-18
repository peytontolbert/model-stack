from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import interpret.cli as cli


class _NoParam(nn.Module):
    def forward(self, x):  # pragma: no cover - unused
        return x


def test_cli_load_helpers_cover_parsing_and_errors(tmp_path, monkeypatch) -> None:
    mod_path = tmp_path / "fake_factory_mod.py"
    mod_path.write_text(
        "import torch.nn as nn\n"
        "def make_model():\n"
        "    return nn.Linear(1, 1)\n"
        "def make_tuple():\n"
        "    return nn.Linear(1, 1), object()\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    plain = cli.load_model("fake_factory_mod:make_model")
    pair = cli.load_model("fake_factory_mod:make_tuple")
    assert isinstance(plain, nn.Module)
    assert isinstance(pair, nn.Module)

    assert torch.equal(cli.parse_tokens("1,2,3"), torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert cli.model_device(_NoParam()) == torch.device("cpu")
    assert cli.maybe_parse_tokens(None, torch.device("cpu")) is None

    model = nn.Linear(1, 1)
    with pytest.raises(ValueError):
        cli.load_forward_kwargs(type("Args", (), {"tokens": None, "enc_tokens": None, "dec_tokens": None})(), model)
    with pytest.raises(ValueError):
        cli.load_pair_kwargs(type("Args", (), {"clean": "1", "corrupted": None, "enc_clean": None, "dec_clean": None, "enc_corrupted": None, "dec_corrupted": None})(), model)
    with pytest.raises(ValueError):
        cli.load_pair_kwargs(type("Args", (), {"clean": None, "corrupted": None, "enc_clean": "1", "dec_clean": None, "enc_corrupted": "1", "dec_corrupted": "1"})(), model)

    forward = cli.load_forward_kwargs(type("Args", (), {"tokens": None, "enc_tokens": "1,2", "dec_tokens": "3,4", "stack": "decoder", "kind": "cross"})(), model)
    assert forward["stack"] == "decoder"
    assert forward["kind"] == "cross"
    pair_kwargs = cli.load_pair_kwargs(type("Args", (), {"clean": None, "corrupted": None, "enc_clean": "1,2", "dec_clean": "3,4", "enc_corrupted": "5,6", "dec_corrupted": "7,8", "stack": "decoder", "kind": "cross"})(), model)
    assert "clean_inputs" in pair_kwargs and "corrupted_inputs" in pair_kwargs


def test_cli_build_parser_covers_all_commands(monkeypatch, capsys) -> None:
    fake_model = nn.Linear(1, 1)
    monkeypatch.setattr(cli, "load_model", lambda _path: fake_model)
    monkeypatch.setattr(cli, "logit_lens", lambda *a, **k: {0: (torch.tensor([1]), torch.tensor([2.0]))})
    monkeypatch.setattr(cli, "causal_trace_restore_fraction", lambda *a, **k: torch.tensor([0.1, 0.2]))
    monkeypatch.setattr(cli, "attention_weights_for_layer", lambda *a, **k: torch.ones(1, 2, 3, 3))
    monkeypatch.setattr(cli, "attention_entropy_for_layer", lambda *a, **k: torch.ones(1, 2, 3))
    monkeypatch.setattr(cli, "attention_rollout", lambda *a, **k: torch.eye(3).unsqueeze(0))
    monkeypatch.setattr(cli, "mlp_lens", lambda *a, **k: {0: (torch.tensor([2]), torch.tensor([1.5]))})
    monkeypatch.setattr(cli, "causal_trace_heads_restore_table", lambda *a, **k: torch.ones(2, 4))
    monkeypatch.setattr(cli, "logit_diff_lens", lambda *a, **k: {0: 1.25})
    monkeypatch.setattr(cli, "head_grad_saliencies", lambda *a, **k: torch.ones(2, 4))
    monkeypatch.setattr(cli, "token_occlusion_importance", lambda *a, **k: torch.tensor([0.1, 0.2, 0.3]))
    monkeypatch.setattr(cli, "greedy_head_recovery", lambda *a, **k: {"selected": [(0, 1)], "curve": torch.tensor([0.1, 0.2])})
    monkeypatch.setattr(cli, "module_importance_scan", lambda *a, **k: [("blocks.0", 1.0)])
    monkeypatch.setattr(cli, "residual_norms", lambda *a, **k: {"pre": torch.ones(2), "post": torch.ones(2), "tokenwise_pre": torch.ones(1, 3, 2), "tokenwise_post": torch.ones(1, 3, 2)})
    monkeypatch.setattr(cli, "estimate_layer_costs", lambda *a, **k: {"total_flops": 123, "bytes_per_token": 456})
    monkeypatch.setattr(cli, "logit_change_with_mask", lambda *a, **k: 0.5)
    monkeypatch.setattr(cli, "component_logit_attribution", lambda *a, **k: {"embed": 1.0})
    monkeypatch.setattr(cli, "head_patch_sweep", lambda *a, **k: {"names": ["a", "b"], "scores": torch.ones(2, 3, 4)})
    monkeypatch.setattr(cli, "block_output_patch_sweep", lambda *a, **k: {"names": ["b0", "b1"], "scores": torch.ones(2, 4)})
    monkeypatch.setattr(cli, "path_patch_sweep", lambda *a, **k: {"source_modules": ["s0"], "receiver_modules": ["r0"], "target_restore": torch.ones(1, 1), "receiver_restore": torch.ones(1, 1), "effects": []})

    parser = cli.build_parser()
    cases = [
        ["logit-lens", "--model", "m:f", "--tokens", "1,2,3", "--topk", "2", "--stack", "causal", "--kind", "self"],
        ["causal-trace", "--model", "m:f", "--clean", "1,2", "--corrupted", "3,4", "--points", "blocks.0", "--topk", "1", "--stack", "causal", "--kind", "self"],
        ["attn-weights", "--model", "m:f", "--tokens", "1,2,3", "--layer", "0", "--stack", "causal", "--kind", "self"],
        ["attn-entropy", "--model", "m:f", "--tokens", "1,2,3", "--layer", "0", "--stack", "causal", "--kind", "self"],
        ["attn-rollout", "--model", "m:f", "--tokens", "1,2,3", "--stack", "causal", "--kind", "self"],
        ["mlp-lens", "--model", "m:f", "--tokens", "1,2,3", "--topk", "2", "--stack", "causal", "--kind", "self"],
        ["head-trace", "--model", "m:f", "--clean", "1,2", "--corrupted", "3,4", "--stack", "causal", "--kind", "self"],
        ["logit-diff-lens", "--model", "m:f", "--tokens", "1,2,3", "--target", "1", "--baseline", "0", "--stack", "causal", "--kind", "self"],
        ["head-saliency", "--model", "m:f", "--tokens", "1,2,3", "--stack", "causal", "--kind", "self"],
        ["occlude", "--model", "m:f", "--tokens", "1,2,3", "--mode", "prob", "--stack", "causal"],
        ["head-greedy", "--model", "m:f", "--enc-clean", "1,2", "--dec-clean", "3,4", "--enc-corrupted", "5,6", "--dec-corrupted", "7,8", "--k", "2", "--stack", "decoder", "--kind", "cross"],
        ["scan-modules", "--model", "m:f", "--tokens", "1,2,3", "--mode", "nll", "--stack", "causal"],
        ["residual-stats", "--model", "m:f", "--tokens", "1,2,3", "--stack", "causal"],
        ["flops", "--model", "m:f", "--seq", "8", "--batch", "2", "--dtype", "fp32"],
        ["mask-effect", "--model", "m:f", "--tokens", "1,2,3", "--type", "block", "--window", "4", "--block", "2", "--dilation", "3", "--stack", "causal"],
        ["component-attribution", "--model", "m:f", "--tokens", "1,2,3", "--stack", "causal"],
        ["head-sweep", "--model", "m:f", "--clean", "1,2", "--corrupted", "3,4", "--stack", "causal", "--kind", "self", "--topk", "2"],
        ["block-sweep", "--model", "m:f", "--clean", "1,2", "--corrupted", "3,4", "--stack", "causal", "--topk", "2"],
        ["path-sweep", "--model", "m:f", "--clean", "1,2", "--corrupted", "3,4", "--sources", "blocks.0,blocks.1", "--receivers", "blocks.2,blocks.3", "--topk", "2"],
    ]
    for argv in cases:
        args = parser.parse_args(argv)
        args.func(args)
    out = capsys.readouterr().out
    assert "layer 0" in out
    assert "recovery top1" in out
    assert "selected" in out
    assert "top_entries" in out
    assert "top_paths" in out


def test_cli_main_dispatches(monkeypatch, capsys) -> None:
    fake_model = nn.Linear(1, 1)
    monkeypatch.setattr(cli, "load_model", lambda _path: fake_model)
    monkeypatch.setattr(cli, "logit_lens", lambda *a, **k: {0: (torch.tensor([9]), torch.tensor([1.0]))})
    monkeypatch.setattr(sys, "argv", ["interpret.cli", "logit-lens", "--model", "m:f", "--tokens", "1,2"])
    cli.main()
    assert "layer 0" in capsys.readouterr().out
