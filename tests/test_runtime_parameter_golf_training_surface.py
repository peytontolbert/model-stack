import types

import numpy as np
import torch

from data.iterable import StreamingTokenIterable
from runtime.positional import resolve_rope_embedding, rope_yarn_inv_freq
from tensor.optim import Muon


def test_parameter_golf_yarn_inv_freq_uses_ramp() -> None:
    inv = rope_yarn_inv_freq(
        seq_len=4096,
        head_dim=16,
        base_theta=10000.0,
        factor=4.0,
        original_max_position_embeddings=1024,
        device="cpu",
    )
    base = 1.0 / (10000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
    ramp = (((torch.arange(0, 16, 2, dtype=torch.float32) / 16) - 0.25) / 0.75).clamp(0.0, 1.0)
    expected = base / (1.0 + ramp * 3.0)
    assert torch.allclose(inv, expected)


def test_resolve_rope_embedding_yarn_returns_reference_dtype() -> None:
    ref = torch.empty(1, 8, 1, dtype=torch.bfloat16)
    cos, sin = resolve_rope_embedding(
        reference=ref,
        head_dim=16,
        base_theta=10000.0,
        scaling_type="parameter_golf_yarn",
        scaling_factor=4.0,
        original_max_position_embeddings=1024,
    )
    assert cos.shape == (8, 16)
    assert sin.shape == (8, 16)
    assert cos.dtype == torch.bfloat16
    assert sin.dtype == torch.bfloat16


def test_streaming_token_iterable_partitions_by_rank_and_worker(tmp_path, monkeypatch) -> None:
    for idx in range(8):
        np.arange(idx * 4, idx * 4 + 4, dtype=np.int32).tofile(tmp_path / f"shard_{idx:02d}.bin")

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        torch.utils.data,
        "get_worker_info",
        lambda: types.SimpleNamespace(id=1, num_workers=2),
    )

    ds = StreamingTokenIterable(tmp_path, seq_len=4, shuffle_shards=False, repeat=False)
    windows = list(iter(ds))
    assert len(windows) == 2
    assert torch.equal(windows[0], torch.tensor([12, 13, 14, 15]))
    assert torch.equal(windows[1], torch.tensor([28, 29, 30, 31]))


def test_muon_optimizer_updates_matrix_params() -> None:
    p = torch.nn.Parameter(torch.eye(4))
    p.grad = torch.ones_like(p)
    opt = Muon([p], lr=0.01, momentum=0.9, backend_steps=1)
    before = p.detach().clone()
    opt.step()
    assert not torch.equal(p.detach(), before)
