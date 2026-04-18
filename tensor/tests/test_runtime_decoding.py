from __future__ import annotations

import pytest
import torch

from runtime.decoding import beam_search, incremental_beam_search
from runtime.native import has_native_op, runtime_info
import runtime.ops as runtime_ops_mod
from runtime.ops import beam_search_step


def test_beam_search_uses_step_fn() -> None:
    input_ids = torch.tensor([[0]], dtype=torch.long)

    def step_fn(x: torch.Tensor) -> torch.Tensor:
        return x + 10

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.shape[0], x.shape[1], 4), -100.0, dtype=torch.float32)
        logits[:, -1, 1] = 0.0
        use_step = x[:, -1] == 10
        logits[use_step, -1, 2] = 10.0
        return logits

    out = beam_search(
        step_fn=step_fn,
        logits_fn=logits_fn,
        input_ids=input_ids,
        beam_size=1,
        max_new_tokens=1,
        eos_id=3,
        pad_id=0,
    )

    assert out.tolist() == [[0, 2]]



def test_beam_search_keeps_finished_beams_closed() -> None:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    cont_id = 3
    alt_id = 4
    input_ids = torch.tensor([[bos_id]], dtype=torch.long)

    def step_fn(x: torch.Tensor) -> torch.Tensor:
        return x

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.shape[0], x.shape[1], 5), -100.0, dtype=torch.float32)
        last = x[:, -1]

        bos_mask = last == bos_id
        logits[bos_mask, -1, eos_id] = 0.0
        logits[bos_mask, -1, cont_id] = -0.1

        cont_mask = last == cont_id
        logits[cont_mask, -1, alt_id] = -5.0
        logits[cont_mask, -1, eos_id] = -6.0

        eos_mask = last == eos_id
        logits[eos_mask, -1, alt_id] = 20.0
        logits[eos_mask, -1, pad_id] = -20.0
        return logits

    out = beam_search(
        step_fn=step_fn,
        logits_fn=logits_fn,
        input_ids=input_ids,
        beam_size=2,
        max_new_tokens=2,
        eos_id=eos_id,
        pad_id=pad_id,
    )

    assert out.tolist() == [[bos_id, eos_id, pad_id]]



def test_beam_search_accepts_rank2_logits() -> None:
    input_ids = torch.tensor([[0]], dtype=torch.long)

    def step_fn(x: torch.Tensor) -> torch.Tensor:
        return x

    def logits_fn(x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((x.shape[0], 5), -100.0, dtype=torch.float32)
        logits[:, 3] = 0.0
        return logits

    out = beam_search(
        step_fn=step_fn,
        logits_fn=logits_fn,
        input_ids=input_ids,
        beam_size=1,
        max_new_tokens=1,
        eos_id=4,
        pad_id=0,
    )

    assert out.tolist() == [[0, 3]]


def test_append_tokens_supports_row_reindex_before_appending(monkeypatch) -> None:
    monkeypatch.setattr(runtime_ops_mod, "has_native_op", lambda name: False)

    seq = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    next_id = torch.tensor([[9], [8], [7]], dtype=torch.long)
    mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.long)

    next_seq, next_mask = runtime_ops_mod.append_tokens(
        seq,
        next_id,
        mask,
        row_ids=torch.tensor([1, 0, 1], dtype=torch.long),
    )

    assert next_seq.tolist() == [[3, 4, 9], [1, 2, 8], [3, 4, 7]]
    assert next_mask is not None
    assert next_mask.tolist() == [[1, 0, 1], [1, 1, 1], [1, 0, 1]]


def test_incremental_beam_search_advances_with_parent_rows() -> None:
    seen: dict[str, torch.Tensor] = {}
    initial_beams = torch.tensor([[1], [1]], dtype=torch.long)
    initial_logits = torch.full((2, 5), -100.0, dtype=torch.float32)
    initial_logits[0, 2] = 0.0
    initial_logits[0, 3] = -0.1

    def advance(parent_rows: torch.Tensor, beams: torch.Tensor) -> torch.Tensor:
        seen["parent_rows"] = parent_rows.clone()
        seen["beams"] = beams.clone()
        logits = torch.full((2, 5), -100.0, dtype=torch.float32)
        logits[0, 4] = 0.0
        logits[0, 0] = -0.2
        logits[1, 4] = -0.3
        logits[1, 0] = -0.4
        return logits

    out = incremental_beam_search(
        initial_beams,
        initial_logits,
        beam_size=2,
        max_new_tokens=2,
        prompt_length=1,
        eos_id=4,
        pad_id=0,
        advance_fn=advance,
    )

    assert out is not None
    beams, raw_scores, finished, lengths, parent_rows = out
    assert seen["parent_rows"].tolist() == [0, 0]
    assert seen["beams"].tolist() == [[1, 2], [1, 3]]
    assert beams.tolist() == [[1, 2, 4], [1, 3, 4]]
    assert raw_scores.shape == (1, 2)
    assert finished.tolist() == [[True, True]]
    assert lengths.tolist() == [[3, 3]]
    assert parent_rows.tolist() == [0, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_beam_search_step_cuda_backend_available_and_correct() -> None:
    if not has_native_op("beam_search_step"):
        pytest.skip("native beam_search_step not available")

    info = runtime_info()
    assert "beam_search_step" in info.get("cuda_backend_ops", [])

    device = torch.device("cuda")
    beams = torch.tensor([[1], [1]], dtype=torch.long, device=device)
    logits = torch.full((2, 5), -100.0, dtype=torch.float32, device=device)
    logits[0, 2] = 0.0
    logits[0, 3] = -0.1
    logits[1, 4] = -0.2
    logits[1, 2] = -0.3
    raw_scores = torch.tensor([[0.0, -float("inf")]], dtype=torch.float32, device=device)
    finished = torch.zeros((1, 2), dtype=torch.bool, device=device)
    lengths = torch.ones((1, 2), dtype=torch.long, device=device)

    next_beams, best_scores, next_finished, next_lengths, parent_rows = beam_search_step(
        beams,
        logits,
        raw_scores,
        finished,
        lengths,
        beam_size=2,
        eos_id=2,
        pad_id=0,
    )

    assert next_beams.tolist() == [[1, 2], [1, 3]]
    assert next_finished.tolist() == [[True, False]]
    assert next_lengths.tolist() == [[2, 2]]
    assert best_scores.shape == (1, 2)
    assert parent_rows.tolist() == [0, 0]
