from __future__ import annotations

import torch

from runtime.decoding import beam_search


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
