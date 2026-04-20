from __future__ import annotations

import threading
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.cache import allocate_model_kv_cache
from runtime.generation import build_generation_config, speculative_decode_step
from runtime.ops import speculative_accept
import serve.scheduler as scheduler_module
from serve.scheduler import GenerationScheduler, PrefixCache, ScheduledWorkItem, SchedulerConfig


class _DeterministicNextTokenModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 64, *, device: str = "cpu", token_shift: int = 1) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.token_shift = int(token_shift)
        self.forward_calls = 0
        self.cfg = SimpleNamespace(
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_model=8,
            pad_token_id=0,
        )
        self.anchor = torch.nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask=None,
        attention_mask=None,
        cache=None,
        position_ids=None,
        cache_position=None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        del attn_mask, attention_mask, cache, position_ids, cache_position, return_dict
        self.forward_calls += 1
        batch, seq_len = input_ids.shape
        logits = torch.full(
            (batch, seq_len, self.vocab_size),
            -1000.0,
            device=input_ids.device,
            dtype=self.anchor.dtype,
        )
        next_ids = (input_ids.to(torch.long) + self.token_shift) % self.vocab_size
        logits.scatter_(2, next_ids.unsqueeze(-1), 0.0)
        return logits


class _PatternCycleModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16, *, device: str = "cpu") -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.cfg = SimpleNamespace(
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_model=8,
            pad_token_id=0,
        )
        self.anchor = torch.nn.Parameter(torch.zeros(1, device=device, dtype=torch.float32))

    def _next_token(self, prefix: list[int]) -> int:
        if len(prefix) >= 2:
            pair = (int(prefix[-2]), int(prefix[-1]))
            mapping = {
                (1, 2): 3,
                (2, 3): 1,
                (3, 1): 2,
            }
            if pair in mapping:
                return mapping[pair]
        return (int(prefix[-1]) % 3) + 1 if prefix else 1

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask=None,
        attention_mask=None,
        cache=None,
        position_ids=None,
        cache_position=None,
        return_dict: bool = False,
    ) -> torch.Tensor:
        del attn_mask, attention_mask, cache, position_ids, cache_position, return_dict
        batch, seq_len = input_ids.shape
        logits = torch.full(
            (batch, seq_len, self.vocab_size),
            -1000.0,
            device=input_ids.device,
            dtype=self.anchor.dtype,
        )
        ids = input_ids.to(torch.long)
        for batch_idx in range(batch):
            prefix: list[int] = []
            for token_idx in range(seq_len):
                prefix.append(int(ids[batch_idx, token_idx].item()))
                logits[batch_idx, token_idx, self._next_token(prefix)] = 0.0
        return logits


def test_speculative_decode_step_accepts_matching_draft_and_emits_bonus_token() -> None:
    prompt = torch.tensor([[5, 6]], dtype=torch.long)
    attention_mask = torch.ones_like(prompt)
    target_model = _DeterministicNextTokenModel()
    draft_model = _DeterministicNextTokenModel()
    cfg = build_generation_config(max_new_tokens=4, do_sample=False, num_speculative_tokens=2)
    cache = allocate_model_kv_cache(target_model, batch_size=1, pagesize=4, backend="paged")
    ready_logits = target_model(
        attention_mask=attention_mask,
        input_ids=prompt,
        cache=None,
        return_dict=False,
    )[:, -1, :]

    result = speculative_decode_step(
        target_model,
        draft_model,
        prompt,
        attention_mask=attention_mask,
        cache=cache,
        config=cfg,
        ready_logits=ready_logits,
        remaining_new_tokens=4,
        cache_pagesize=4,
        cache_backend="paged",
    )

    assert result is not None
    assert result.seq.tolist() == [[5, 6, 7, 8, 9]]
    assert result.emitted_tokens == 3
    assert result.accepted_tokens == 2
    assert result.drafted_tokens == 2


def test_speculative_decode_step_rejects_mismatched_draft_losslessly() -> None:
    prompt = torch.tensor([[9, 10]], dtype=torch.long)
    attention_mask = torch.ones_like(prompt)
    target_model = _DeterministicNextTokenModel()
    draft_model = _DeterministicNextTokenModel(token_shift=2)
    cfg = build_generation_config(max_new_tokens=3, do_sample=False, num_speculative_tokens=2)
    cache = allocate_model_kv_cache(target_model, batch_size=1, pagesize=4, backend="paged")
    ready_logits = target_model(
        prompt,
        attention_mask=attention_mask,
        cache=None,
        return_dict=False,
    )[:, -1, :]

    result = speculative_decode_step(
        target_model,
        draft_model,
        prompt,
        attention_mask=attention_mask,
        cache=cache,
        config=cfg,
        ready_logits=ready_logits,
        remaining_new_tokens=3,
        cache_pagesize=4,
        cache_backend="paged",
    )

    assert result is not None
    assert result.seq.tolist() == [[9, 10, 11]]
    assert result.emitted_tokens == 1
    assert result.accepted_tokens == 0


def test_speculative_accept_rejection_sampler_emits_lossless_residual_sample() -> None:
    target_probs = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    draft_probs = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32)
    draft_token_ids = torch.tensor([[1]], dtype=torch.long)

    emitted, lengths, accepted = speculative_accept(
        target_probs,
        draft_probs,
        draft_token_ids,
        method="rejection_sampler",
    )

    assert lengths.tolist() == [1]
    assert accepted.tolist() == [0]
    assert emitted[:, :1].tolist() == [[0]]


def test_speculative_accept_typical_acceptance_sampler_rejects_below_threshold() -> None:
    target_probs = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
    draft_probs = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32)
    draft_token_ids = torch.tensor([[1]], dtype=torch.long)

    emitted, lengths, accepted = speculative_accept(
        target_probs,
        draft_probs,
        draft_token_ids,
        method="typical_acceptance_sampler",
        posterior_threshold=0.09,
        posterior_alpha=0.3,
    )

    assert lengths.tolist() == [1]
    assert accepted.tolist() == [0]
    assert emitted[:, :1].tolist() == [[0]]


def test_speculative_decode_step_supports_ngram_prompt_lookup() -> None:
    prompt = torch.tensor([[1, 2, 3, 1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(prompt)
    target_model = _PatternCycleModel()
    cfg = build_generation_config(
        max_new_tokens=4,
        do_sample=False,
        num_speculative_tokens=2,
        speculative_method="ngram",
        prompt_lookup_min=2,
        prompt_lookup_max=3,
    )
    cache = allocate_model_kv_cache(target_model, batch_size=1, pagesize=4, backend="paged")
    ready_logits = target_model(
        prompt,
        attention_mask=attention_mask,
        cache=None,
        return_dict=False,
    )[:, -1, :]

    result = speculative_decode_step(
        target_model,
        None,
        prompt,
        attention_mask=attention_mask,
        cache=cache,
        config=cfg,
        ready_logits=ready_logits,
        prompt_seq=prompt,
        remaining_new_tokens=4,
        cache_pagesize=4,
        cache_backend="paged",
    )

    assert result is not None
    assert result.seq.tolist() == [[1, 2, 3, 1, 2, 3, 1, 2]]
    assert result.accepted_tokens == 2
    assert result.emitted_tokens == 3


def test_speculative_decode_step_supports_suffix_lookup() -> None:
    prompt = torch.tensor([[1, 2, 3, 1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(prompt)
    target_model = _PatternCycleModel()
    cfg = build_generation_config(
        max_new_tokens=4,
        do_sample=False,
        num_speculative_tokens=2,
        speculative_method="suffix",
        suffix_decoding_max_tree_depth=4,
        suffix_decoding_max_spec_factor=2.0,
    )
    cache = allocate_model_kv_cache(target_model, batch_size=1, pagesize=4, backend="paged")
    ready_logits = target_model(
        prompt,
        attention_mask=attention_mask,
        cache=None,
        return_dict=False,
    )[:, -1, :]

    result = speculative_decode_step(
        target_model,
        None,
        prompt,
        attention_mask=attention_mask,
        cache=cache,
        config=cfg,
        ready_logits=ready_logits,
        prompt_seq=prompt,
        remaining_new_tokens=4,
        cache_pagesize=4,
        cache_backend="paged",
    )

    assert result is not None
    assert result.seq.tolist() == [[1, 2, 3, 1, 2, 3, 1, 2]]
    assert result.accepted_tokens == 2
    assert result.emitted_tokens == 3


def test_scheduler_routes_ready_requests_through_speculative_decode() -> None:
    model = _DeterministicNextTokenModel()
    draft_model = _DeterministicNextTokenModel()
    scheduler = GenerationScheduler(
        model=model,
        draft_model=draft_model,
        cache_pagesize=4,
        default_cache_backend="paged",
        config=SchedulerConfig(
            enabled=True,
            max_batch_size=8,
            max_queue_delay_ms=0,
            prefix_cache_size=0,
            prefix_cache_min_tokens=1,
            prefill_chunk_size=None,
            max_num_batched_tokens=8,
        ),
    )
    decode_groups: list[int] = []
    original_process_decode_group = scheduler._process_decode_group

    def tracked_process_decode_group(items):
        decode_groups.append(len(items))
        return original_process_decode_group(items)

    scheduler._process_decode_group = tracked_process_decode_group
    cfg = build_generation_config(max_new_tokens=3, do_sample=False, num_speculative_tokens=2)
    prompt = torch.tensor([[2, 3]], dtype=torch.long)
    attention_mask = torch.ones_like(prompt)

    try:
        output = scheduler.submit(prompt, attention_mask=attention_mask, config=cfg)

        assert output.tolist() == [[2, 3, 4, 5, 6]]
        assert decode_groups == []
    finally:
        scheduler.close()


def test_scheduler_batches_speculative_ready_requests_together() -> None:
    model = _DeterministicNextTokenModel()
    draft_model = _DeterministicNextTokenModel()
    scheduler = GenerationScheduler(
        model=model,
        draft_model=draft_model,
        cache_pagesize=4,
        default_cache_backend="paged",
        config=SchedulerConfig(
            enabled=True,
            max_batch_size=8,
            max_queue_delay_ms=0,
            prefix_cache_size=0,
            prefix_cache_min_tokens=1,
            prefill_chunk_size=None,
            max_num_batched_tokens=8,
        ),
    )
    cfg = build_generation_config(max_new_tokens=3, do_sample=False, num_speculative_tokens=2)
    req_a = scheduler._build_request(
        request_id=1,
        priority=0,
        input_ids=torch.tensor([[2, 3]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        config=cfg,
        cache_backend="paged",
    )
    req_b = scheduler._build_request(
        request_id=2,
        priority=0,
        input_ids=torch.tensor([[4, 5]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        config=cfg,
        cache_backend="paged",
    )
    batch_sizes: list[int] = []
    original_batch = scheduler_module.speculative_decode_batch

    def tracked_batch(*args, **kwargs):
        requests = kwargs.get("requests")
        if requests is None and len(args) >= 3:
            requests = args[2]
        batch_sizes.append(len(requests or []))
        return original_batch(*args, **kwargs)

    scheduler_module.speculative_decode_batch = tracked_batch
    try:
        scheduler._run_request_step(req_a, prefill_tokens=2)
        scheduler._run_request_step(req_b, prefill_tokens=2)

        assert req_a.ready_logits is not None
        assert req_b.ready_logits is not None

        scheduler._process_ready_requests([req_a, req_b])

        assert batch_sizes == [2]
        assert req_a.seq.tolist() == [[2, 3, 4, 5, 6]]
        assert req_b.seq.tolist() == [[4, 5, 6, 7, 8]]
    finally:
        scheduler_module.speculative_decode_batch = original_batch
        scheduler.close()


def test_scheduler_batches_mixed_length_decode_requests() -> None:
    model = _DeterministicNextTokenModel()
    scheduler = GenerationScheduler(
        model=model,
        cache_pagesize=4,
        default_cache_backend="paged",
        config=SchedulerConfig(
            enabled=True,
            max_batch_size=8,
            max_queue_delay_ms=20,
            prefix_cache_size=8,
            prefix_cache_min_tokens=1,
            prefill_chunk_size=None,
        ),
    )
    decode_batches: list[tuple[int, ...]] = []
    original_build_decode_batch = scheduler._build_padded_decode_batch

    def tracked_build_decode_batch(requests):
        decode_batches.append(tuple(int(request.seq.shape[1]) for request in requests))
        return original_build_decode_batch(requests)

    scheduler._build_padded_decode_batch = tracked_build_decode_batch
    cfg = build_generation_config(max_new_tokens=2, do_sample=False, prefill_chunk_size=None)
    req_a = scheduler._build_request(
        request_id=1,
        priority=0,
        input_ids=torch.tensor([[5, 6]], dtype=torch.long),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        config=cfg,
        cache_backend="paged",
    )
    req_b = scheduler._build_request(
        request_id=2,
        priority=0,
        input_ids=torch.tensor([[9, 10, 11, 12]], dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        config=cfg,
        cache_backend="paged",
    )

    try:
        scheduler._run_request_step(req_a, prefill_tokens=2)
        scheduler._run_request_step(req_b, prefill_tokens=4)
        scheduler._process_ready_requests([req_a, req_b])

        scheduler._process_decode_group(
            [
                ScheduledWorkItem(kind="decode", request=req_a, tokens=1),
                ScheduledWorkItem(kind="decode", request=req_b, tokens=1),
            ]
        )

        assert req_a.seq.tolist() == [[5, 6, 7, 8]]
        assert req_b.seq.tolist() == [[9, 10, 11, 12, 13, 14]]
        assert decode_batches == [(3, 5)]
    finally:
        scheduler.close()


def test_scheduler_reuses_exact_prefix_cache() -> None:
    model = _DeterministicNextTokenModel()
    scheduler = GenerationScheduler(
        model=model,
        cache_pagesize=4,
        default_cache_backend="paged",
        config=SchedulerConfig(
            enabled=True,
            max_batch_size=4,
            max_queue_delay_ms=0,
            prefix_cache_size=8,
            prefix_cache_min_tokens=1,
            prefill_chunk_size=None,
        ),
    )
    prefill_groups: list[int] = []
    original_process_prefill_group = scheduler._process_prefill_group

    def tracked_process_prefill_group(requests):
        prefill_groups.append(len(requests))
        return original_process_prefill_group(requests)

    scheduler._process_prefill_group = tracked_process_prefill_group
    cfg = build_generation_config(max_new_tokens=1, do_sample=False, prefill_chunk_size=None)
    prompt = torch.tensor([[3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(prompt)

    try:
        first = scheduler.submit(prompt, attention_mask=attention_mask, config=cfg)
        second = scheduler.submit(prompt, attention_mask=attention_mask, config=cfg)

        assert first.tolist() == [[3, 4, 5, 6]]
        assert second.tolist() == [[3, 4, 5, 6]]
        assert prefill_groups == [1]
    finally:
        scheduler.close()


def test_prefix_cache_supports_partial_matches_for_token_masks() -> None:
    prefix_cache = PrefixCache(max_entries=4, min_tokens=1)
    prefix_cache.store(
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([1, 1, 1], dtype=torch.long),
        "paged",
        object(),
        torch.zeros(1, 8),
    )

    match = prefix_cache.lookup(
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
        "paged",
    )

    assert match is not None
    assert match.exact is False
    assert match.prefix_len == 3


def test_priority_policy_schedules_lower_priority_value_first() -> None:
    model = _DeterministicNextTokenModel()
    scheduler = GenerationScheduler(
        model=model,
        cache_pagesize=4,
        default_cache_backend="paged",
        config=SchedulerConfig(
            enabled=True,
            max_batch_size=1,
            max_queue_delay_ms=0,
            prefix_cache_size=0,
            prefix_cache_min_tokens=1,
            prefill_chunk_size=None,
            max_num_batched_tokens=1,
            scheduling_policy="priority",
        ),
    )
    cfg = build_generation_config(max_new_tokens=1, do_sample=False)
    low = scheduler._build_request(
        request_id=1,
        priority=10,
        input_ids=torch.tensor([[7]], dtype=torch.long),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        config=cfg,
        cache_backend="paged",
    )
    high = scheduler._build_request(
        request_id=2,
        priority=0,
        input_ids=torch.tensor([[9]], dtype=torch.long),
        attention_mask=torch.ones((1, 1), dtype=torch.long),
        config=cfg,
        cache_backend="paged",
    )
    low.created_at = 1.0
    high.created_at = 2.0

    try:
        work_items, consumed = scheduler._schedule_compute_items([low, high], token_budget=1)

        assert consumed == 1
        assert len(work_items) == 1
        assert work_items[0].request.request_id == 2
        assert work_items[0].request.priority == 0
    finally:
        scheduler.close()


def test_scheduler_chunks_long_prefills_concurrently_under_token_budget() -> None:
    model = _DeterministicNextTokenModel()
    scheduler = GenerationScheduler(
        model=model,
        cache_pagesize=4,
        default_cache_backend="paged",
        config=SchedulerConfig(
            enabled=True,
            max_batch_size=8,
            max_queue_delay_ms=0,
            prefix_cache_size=0,
            prefix_cache_min_tokens=1,
            prefill_chunk_size=None,
            max_num_batched_tokens=4,
            max_num_partial_prefills=2,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=2,
        ),
    )
    cfg = build_generation_config(max_new_tokens=1, do_sample=False)
    req_a = scheduler._build_request(
        request_id=1,
        priority=0,
        input_ids=torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        config=cfg,
        cache_backend="paged",
    )
    req_b = scheduler._build_request(
        request_id=2,
        priority=0,
        input_ids=torch.tensor([[6, 7, 8, 9, 10]], dtype=torch.long),
        attention_mask=torch.ones((1, 5), dtype=torch.long),
        config=cfg,
        cache_backend="paged",
    )

    try:
        work_items, consumed = scheduler._schedule_compute_items([req_a, req_b], token_budget=4)

        assert consumed == 4
        assert [(item.kind, item.tokens, item.request.request_id) for item in work_items] == [
            ("prefill", 2, 1),
            ("prefill", 2, 2),
        ]
    finally:
        scheduler.close()


def test_scheduler_iteration_can_mix_ready_prefill_and_decode_work() -> None:
    model = _DeterministicNextTokenModel()
    scheduler = GenerationScheduler(
        model=model,
        cache_pagesize=4,
        default_cache_backend="paged",
        config=SchedulerConfig(
            enabled=True,
            max_batch_size=8,
            max_queue_delay_ms=0,
            prefix_cache_size=8,
            prefix_cache_min_tokens=1,
            prefill_chunk_size=None,
            max_num_batched_tokens=3,
            max_num_partial_prefills=2,
            max_long_partial_prefills=2,
            long_prefill_token_threshold=2,
        ),
    )
    cfg = build_generation_config(max_new_tokens=2, do_sample=False)
    warm_prompt = torch.tensor([[3, 4]], dtype=torch.long)
    warm_mask = torch.ones_like(warm_prompt)

    try:
        assert scheduler.submit(warm_prompt, attention_mask=warm_mask, config=cfg).tolist() == [[3, 4, 5, 6]]
        ready_request = scheduler._build_request(
            request_id=10,
            priority=0,
            input_ids=warm_prompt,
            attention_mask=warm_mask,
            config=cfg,
            cache_backend="paged",
        )
        prefill_request = scheduler._build_request(
            request_id=11,
            priority=0,
            input_ids=torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long),
            attention_mask=torch.ones((1, 5), dtype=torch.long),
            config=cfg,
            cache_backend="paged",
        )

        assert ready_request.ready_logits is not None
        assert prefill_request.prefill_ids is not None and int(prefill_request.prefill_ids.shape[1]) == 5

        scheduler._run_scheduler_iteration([ready_request, prefill_request])

        assert ready_request.generated_tokens == 2
        assert ready_request.done is True
        assert prefill_request.prefill_ids is not None
        assert int(prefill_request.prefill_ids.shape[1]) == 3
    finally:
        scheduler.close()
