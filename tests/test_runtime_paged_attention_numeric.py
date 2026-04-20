from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.kv_cache import PagedKVCache
from runtime.ops import paged_attention_decode as runtime_paged_attention_decode
from runtime.ops import speculative_accept as runtime_speculative_accept


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for paged attention numeric tests")
    return torch.device("cuda")


def _build_paged_kv(
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    lengths: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, kv_heads, max_len, head_dim = k_dense.shape
    max_blocks = (max_len + page_size - 1) // page_size
    total_pages = batch * max_blocks
    k_pages = torch.zeros(
        total_pages,
        kv_heads,
        page_size,
        head_dim,
        dtype=k_dense.dtype,
        device=k_dense.device,
    )
    v_pages = torch.zeros_like(k_pages)
    block_table = torch.arange(total_pages, device=k_dense.device, dtype=torch.long).view(batch, max_blocks)
    for batch_idx in range(batch):
        row_len = int(lengths[batch_idx].item())
        for token_idx in range(row_len):
            page_idx = int(block_table[batch_idx, token_idx // page_size].item())
            page_offset = token_idx % page_size
            k_pages[page_idx, :, page_offset, :] = k_dense[batch_idx, :, token_idx, :]
            v_pages[page_idx, :, page_offset, :] = v_dense[batch_idx, :, token_idx, :]
    return k_pages, v_pages, block_table


def _dense_reference(
    q: torch.Tensor,
    k_dense: torch.Tensor,
    v_dense: torch.Tensor,
    lengths: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    q_float = q.to(torch.float32)
    k_float = k_dense.to(torch.float32)
    v_float = v_dense.to(torch.float32)
    if q.shape[1] != k_dense.shape[1]:
        repeat = q.shape[1] // k_dense.shape[1]
        k_float = k_float.repeat_interleave(repeat, dim=1)
        v_float = v_float.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(q_float, k_float.transpose(2, 3))
    scores = scores * (float(scale) if scale is not None else (q.shape[-1] ** -0.5))
    positions = torch.arange(k_dense.shape[2], device=q.device, dtype=torch.long).view(1, 1, 1, -1)
    invalid = positions >= lengths.view(-1, 1, 1, 1)
    scores = scores.masked_fill(invalid, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask.to(device=q.device, dtype=scores.dtype)
            scores = scores.masked_fill(invalid, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.matmul(weights, v_float).to(dtype=q.dtype)


def test_runtime_paged_attention_decode_matches_dense_reference() -> None:
    device = _require_cuda()
    torch.manual_seed(0)

    batch = 2
    q_heads = 4
    kv_heads = 2
    max_len = 7
    head_dim = 8
    page_size = 4
    scale = 0.75

    q = torch.randn(batch, q_heads, 1, head_dim, device=device, dtype=torch.float32)
    k_dense = torch.randn(batch, kv_heads, max_len, head_dim, device=device, dtype=torch.float32)
    v_dense = torch.randn(batch, kv_heads, max_len, head_dim, device=device, dtype=torch.float32)
    lengths = torch.tensor([7, 5], device=device, dtype=torch.long)
    bool_mask = torch.zeros(batch, q_heads, 1, max_len, device=device, dtype=torch.bool)
    bool_mask[0, :, 0, 1] = True
    bool_mask[1, :, 0, 3] = True

    k_pages, v_pages, block_table = _build_paged_kv(k_dense, v_dense, lengths, page_size)
    out = runtime_paged_attention_decode(
        q,
        k_pages,
        v_pages,
        block_table,
        lengths,
        bool_mask,
        scale=scale,
    )
    ref = _dense_reference(q, k_dense, v_dense, lengths, bool_mask, scale=scale)

    assert out.shape == q.shape
    assert torch.allclose(out, ref, atol=2e-4, rtol=2e-4)


def test_paged_kv_cache_layer_decode_matches_dense_reference() -> None:
    device = _require_cuda()
    torch.manual_seed(1)

    batch = 2
    q_heads = 4
    kv_heads = 2
    seq_len = 5
    head_dim = 8
    scale = 0.5

    cache = PagedKVCache(
        batch=batch,
        n_layers=1,
        n_kv_heads=kv_heads,
        head_dim=head_dim,
        pagesize=4,
        dtype=torch.float32,
        device=device,
    )
    q = torch.randn(batch, q_heads, 1, head_dim, device=device, dtype=torch.float32)
    k_chunk = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v_chunk = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    float_mask = torch.zeros(batch, q_heads, 1, seq_len, device=device, dtype=torch.float32)
    float_mask[0, :, 0, 0] = -1.25
    float_mask[1, :, 0, 4] = -0.5

    out = cache.layer(0).paged_attention_decode(
        q,
        k_chunk,
        v_chunk,
        attn_mask=float_mask,
        scale=scale,
    )
    ref = _dense_reference(
        q,
        k_chunk,
        v_chunk,
        torch.full((batch,), seq_len, device=device, dtype=torch.long),
        float_mask,
        scale=scale,
    )

    assert int(cache.layer(0).length()) == seq_len
    assert torch.equal(cache.layer_lengths(0), torch.full((batch,), seq_len, device=device, dtype=torch.long))
    assert torch.allclose(out, ref, atol=2e-4, rtol=2e-4)


def test_cuda_speculative_accept_matches_expected_tokens() -> None:
    device = _require_cuda()

    target_probs = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    draft_probs = torch.tensor(
        [
            [[0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    draft_token_ids = torch.tensor([[1], [1]], device=device, dtype=torch.long)
    bonus_probs = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    bonus_enabled = torch.tensor([False, True], device=device, dtype=torch.bool)

    emitted, lengths, accepted = runtime_speculative_accept(
        target_probs,
        draft_probs,
        draft_token_ids,
        bonus_probs=bonus_probs,
        bonus_enabled=bonus_enabled,
        method="rejection_sampler",
    )

    assert emitted.device.type == "cuda"
    assert lengths.device.type == "cuda"
    assert accepted.device.type == "cuda"
    assert emitted[:, :2].cpu().tolist() == [[0, -1], [1, 2]]
    assert lengths.cpu().tolist() == [1, 2]
    assert accepted.cpu().tolist() == [0, 1]
