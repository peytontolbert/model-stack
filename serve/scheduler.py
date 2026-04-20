from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import torch

from runtime.cache import allocate_model_kv_cache, evict_kv_cache
from runtime.generation import (
    GenerationConfig,
    RuntimeGenerationSession,
    SpeculativeBatchRequest,
    speculative_decode_batch,
    speculative_decode_step,
)
from runtime.kv_cache import clone_kv_cache_rows, concat_kv_caches, split_kv_cache_rows
from runtime.ops import sample_with_policies as runtime_sample_with_policies


@dataclass(frozen=True)
class SchedulerConfig:
    enabled: bool = True
    max_batch_size: int = 8
    max_queue_delay_ms: int = 2
    prefix_cache_size: int = 64
    prefix_cache_min_tokens: int = 16
    prefill_chunk_size: int | None = None
    max_num_batched_tokens: int = 2048
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_token_threshold: int = 0
    scheduling_policy: str = "fcfs"

    @classmethod
    def from_env(cls) -> "SchedulerConfig":
        def _env_flag(name: str, default: bool) -> bool:
            raw = str(__import__("os").environ.get(name, "1" if default else "0")).strip().lower()
            return raw not in {"0", "false", "no", "off", ""}

        def _env_int(name: str, default: int) -> int:
            raw = __import__("os").environ.get(name)
            if raw is None:
                return int(default)
            try:
                return int(raw)
            except Exception:
                return int(default)

        raw_prefill_chunk = __import__("os").environ.get("MODEL_STACK_PREFILL_CHUNK_SIZE")
        prefill_chunk_size = None
        if raw_prefill_chunk is not None:
            try:
                parsed = int(raw_prefill_chunk)
                prefill_chunk_size = parsed if parsed > 0 else None
            except Exception:
                prefill_chunk_size = None

        scheduling_policy = str(
            __import__("os").environ.get("MODEL_STACK_SCHEDULER_POLICY", "fcfs")
        ).strip().lower() or "fcfs"
        if scheduling_policy not in {"fcfs", "priority"}:
            scheduling_policy = "fcfs"

        max_num_partial_prefills = max(_env_int("MODEL_STACK_SCHEDULER_MAX_PARTIAL_PREFILLS", 1), 1)
        max_long_partial_prefills = max(_env_int("MODEL_STACK_SCHEDULER_MAX_LONG_PARTIAL_PREFILLS", 1), 1)
        max_long_partial_prefills = min(max_long_partial_prefills, max_num_partial_prefills)

        return cls(
            enabled=_env_flag("MODEL_STACK_ENABLE_SCHEDULER", True),
            max_batch_size=max(_env_int("MODEL_STACK_SCHEDULER_MAX_BATCH", 8), 1),
            max_queue_delay_ms=max(_env_int("MODEL_STACK_SCHEDULER_MAX_QUEUE_MS", 2), 0),
            prefix_cache_size=max(_env_int("MODEL_STACK_PREFIX_CACHE_SIZE", 64), 0),
            prefix_cache_min_tokens=max(_env_int("MODEL_STACK_PREFIX_CACHE_MIN_TOKENS", 16), 1),
            prefill_chunk_size=prefill_chunk_size,
            max_num_batched_tokens=max(_env_int("MODEL_STACK_SCHEDULER_MAX_BATCHED_TOKENS", 2048), 1),
            max_num_partial_prefills=max_num_partial_prefills,
            max_long_partial_prefills=max_long_partial_prefills,
            long_prefill_token_threshold=max(_env_int("MODEL_STACK_SCHEDULER_LONG_PREFILL_TOKENS", 0), 0),
            scheduling_policy=scheduling_policy,
        )


@dataclass
class PrefixCacheEntry:
    prompt_tokens: tuple[int, ...]
    attention_mask: tuple[int, ...] | None
    cache_backend: str
    cache: object
    next_logits: torch.Tensor
    created_at: float = field(default_factory=time.time)
    hits: int = 0


@dataclass(frozen=True)
class PrefixCacheMatch:
    entry: PrefixCacheEntry
    prefix_len: int
    exact: bool


class PrefixCache:
    def __init__(self, max_entries: int, min_tokens: int) -> None:
        self.max_entries = max(int(max_entries), 0)
        self.min_tokens = max(int(min_tokens), 1)
        self._entries: OrderedDict[
            tuple[str, tuple[int, ...], tuple[int, ...] | None],
            PrefixCacheEntry,
        ] = OrderedDict()

    def _make_key(
        self,
        prompt_tokens: tuple[int, ...],
        attention_mask: tuple[int, ...] | None,
        cache_backend: str,
    ) -> tuple[str, tuple[int, ...], tuple[int, ...] | None]:
        return (str(cache_backend), tuple(prompt_tokens), attention_mask)

    def _mask_prefix_matches(
        self,
        candidate_mask: tuple[int, ...] | None,
        query_mask: tuple[int, ...] | None,
        prefix_len: int,
    ) -> bool:
        if candidate_mask is None or query_mask is None:
            return candidate_mask is None and query_mask is None
        if len(candidate_mask) != prefix_len or len(query_mask) < prefix_len:
            return False
        return tuple(query_mask[:prefix_len]) == tuple(candidate_mask)

    def lookup(
        self,
        prompt: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_backend: str,
    ) -> PrefixCacheMatch | None:
        if self.max_entries <= 0:
            return None
        prompt_tokens = tuple(int(token) for token in prompt.view(-1).detach().to(torch.long).cpu().tolist())
        mask_tokens = None
        if attention_mask is not None:
            mask_tokens = tuple(int(token) for token in attention_mask.view(-1).detach().to(torch.long).cpu().tolist())
        exact_key = self._make_key(prompt_tokens, mask_tokens, cache_backend)
        entry = self._entries.get(exact_key)
        if entry is not None:
            entry.hits += 1
            self._entries.move_to_end(exact_key)
            return PrefixCacheMatch(entry=entry, prefix_len=len(prompt_tokens), exact=True)

        best_key = None
        best_entry = None
        best_prefix_len = 0
        for key, candidate in self._entries.items():
            if key[0] != str(cache_backend):
                continue
            prefix_len = len(candidate.prompt_tokens)
            if prefix_len <= 0 or prefix_len >= len(prompt_tokens):
                continue
            if prompt_tokens[:prefix_len] != candidate.prompt_tokens:
                continue
            if not self._mask_prefix_matches(candidate.attention_mask, mask_tokens, prefix_len):
                continue
            if prefix_len > best_prefix_len:
                best_key = key
                best_entry = candidate
                best_prefix_len = prefix_len
        if best_entry is None or best_key is None:
            return None
        best_entry.hits += 1
        self._entries.move_to_end(best_key)
        return PrefixCacheMatch(entry=best_entry, prefix_len=best_prefix_len, exact=False)

    def store(
        self,
        prompt: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_backend: str,
        cache,
        next_logits: torch.Tensor,
    ) -> None:
        if self.max_entries <= 0 or cache is None or next_logits is None:
            return
        prompt_tokens = tuple(int(token) for token in prompt.view(-1).detach().to(torch.long).cpu().tolist())
        if len(prompt_tokens) < self.min_tokens:
            return
        mask_tokens = None
        if attention_mask is not None:
            mask_tokens = tuple(int(token) for token in attention_mask.view(-1).detach().to(torch.long).cpu().tolist())
        key = self._make_key(prompt_tokens, mask_tokens, cache_backend)
        self._entries[key] = PrefixCacheEntry(
            prompt_tokens=prompt_tokens,
            attention_mask=mask_tokens,
            cache_backend=str(cache_backend),
            cache=cache,
            next_logits=next_logits.detach().clone(),
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


@dataclass
class ScheduledGenerationRequest:
    request_id: int
    priority: int
    prompt_seq: torch.Tensor
    prompt_attention_mask: torch.Tensor | None
    seq: torch.Tensor
    attention_mask: torch.Tensor | None
    config: GenerationConfig
    cache_backend: str
    cache: object | None = None
    ready_logits: torch.Tensor | None = None
    prefill_ids: torch.Tensor | None = None
    prefill_attention_mask: torch.Tensor | None = None
    generated_tokens: int = 0
    created_at: float = field(default_factory=time.time)
    result: torch.Tensor | None = None
    error: BaseException | None = None
    done: bool = False
    event: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class ScheduledWorkItem:
    kind: str
    request: ScheduledGenerationRequest
    tokens: int


class GenerationScheduler:
    def __init__(
        self,
        *,
        model,
        draft_model=None,
        cache_pagesize: int,
        default_cache_backend: str,
        config: SchedulerConfig,
    ) -> None:
        self.model = model
        self.draft_model = draft_model
        self.cache_pagesize = int(cache_pagesize)
        self.default_cache_backend = str(default_cache_backend)
        self.config = config
        self.prefix_cache = PrefixCache(
            max_entries=int(config.prefix_cache_size),
            min_tokens=int(config.prefix_cache_min_tokens),
        )
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._prefix_cache_lock = threading.Lock()
        self._pending: list[ScheduledGenerationRequest] = []
        self._active: list[ScheduledGenerationRequest] = []
        self._shutdown = False
        self._next_request_id = 1
        self._worker = threading.Thread(target=self._worker_main, name="generation-scheduler", daemon=True)
        self._worker.start()

    def submit(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        config: GenerationConfig,
        cache_backend: str | None = None,
        priority: int = 0,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError("GenerationScheduler only supports single-request submissions")
        with self._cv:
            request_id = int(self._next_request_id)
            self._next_request_id += 1
        request = self._build_request(
            request_id=request_id,
            priority=int(priority),
            input_ids=input_ids.contiguous(),
            attention_mask=(attention_mask.contiguous() if attention_mask is not None else None),
            config=config,
            cache_backend=(cache_backend or self.default_cache_backend),
        )
        if int(config.max_new_tokens) <= 0:
            return request.seq.detach().clone()
        with self._cv:
            self._pending.append(request)
            self._cv.notify()
        request.event.wait()
        if request.error is not None:
            raise RuntimeError(str(request.error)) from request.error
        if request.result is None:
            raise RuntimeError("scheduled generation finished without a result")
        return request.result

    def close(self) -> None:
        with self._cv:
            self._shutdown = True
            self._cv.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def _build_request(
        self,
        *,
        request_id: int,
        priority: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        config: GenerationConfig,
        cache_backend: str,
    ) -> ScheduledGenerationRequest:
        request = ScheduledGenerationRequest(
            request_id=int(request_id),
            priority=int(priority),
            prompt_seq=input_ids.clone(),
            prompt_attention_mask=(attention_mask.clone() if attention_mask is not None else None),
            seq=input_ids.clone(),
            attention_mask=(attention_mask.clone() if attention_mask is not None else None),
            config=config,
            cache_backend=str(cache_backend),
            prefill_ids=input_ids.clone(),
            prefill_attention_mask=(attention_mask.clone() if attention_mask is not None else None),
        )
        with self._prefix_cache_lock:
            match = self.prefix_cache.lookup(
                input_ids[0],
                attention_mask[0] if attention_mask is not None else None,
                request.cache_backend,
            )
        if match is None:
            try:
                request.cache = allocate_model_kv_cache(
                    self.model,
                    batch_size=1,
                    pagesize=self.cache_pagesize,
                    backend=request.cache_backend,
                )
            except Exception:
                request.cache = None
            return request
        request.cache = self._clone_single_row_cache(match.entry.cache)
        if match.exact:
            request.ready_logits = match.entry.next_logits.detach().clone()
            request.prefill_ids = None
            request.prefill_attention_mask = None
            return request
        request.prefill_ids = input_ids[:, match.prefix_len:].contiguous()
        if attention_mask is not None:
            request.prefill_attention_mask = attention_mask[:, match.prefix_len:].contiguous()
        else:
            request.prefill_attention_mask = None
        return request

    def _worker_main(self) -> None:
        while True:
            with self._cv:
                while not self._shutdown and not self._pending and not self._active:
                    self._cv.wait()
                if self._shutdown:
                    return
                if self._pending:
                    wait_s = float(self.config.max_queue_delay_ms) / 1000.0
                    if wait_s > 0.0:
                        self._cv.wait(timeout=wait_s)
                    self._active.extend(self._pending)
                    self._pending.clear()
                active_snapshot = [request for request in self._active if not request.done]
            if not active_snapshot:
                continue
            try:
                with torch.inference_mode():
                    self._run_scheduler_iteration(active_snapshot)
            except Exception as exc:
                for request in active_snapshot:
                    self._fail_request(request, exc)
            with self._cv:
                self._active = [request for request in self._active if not request.done]

    def _run_scheduler_iteration(self, requests: list[ScheduledGenerationRequest]) -> None:
        token_budget = int(self.config.max_num_batched_tokens)
        while True:
            ready = [request for request in requests if not request.done and request.ready_logits is not None]
            if ready:
                self._process_ready_requests(ready)
            if token_budget <= 0:
                break
            work_items, consumed_tokens = self._schedule_compute_items(requests, token_budget)
            if not work_items or consumed_tokens <= 0:
                break
            self._dispatch_work_items(work_items)
            token_budget -= int(consumed_tokens)
        trailing_ready = [request for request in requests if not request.done and request.ready_logits is not None]
        if trailing_ready:
            self._process_ready_requests(trailing_ready)

    def _request_sort_key(self, request: ScheduledGenerationRequest) -> tuple[float, ...]:
        if str(self.config.scheduling_policy) == "priority":
            return (float(request.priority), float(request.created_at), float(request.request_id))
        return (float(request.created_at), float(request.request_id))

    def _remaining_prefill_tokens(self, request: ScheduledGenerationRequest) -> int:
        if request.prefill_ids is None:
            return 0
        return int(request.prefill_ids.shape[1])

    def _cache_length(self, cache) -> int:
        if cache is None or not hasattr(cache, "layer"):
            return 0
        try:
            return int(cache.layer(0).length())
        except Exception:
            return 0

    def _clone_single_row_cache(self, cache):
        row_ids = torch.tensor([0], device=cache.device, dtype=torch.long)
        return clone_kv_cache_rows(cache, row_ids)

    def _prefill_chunk_size(self, request: ScheduledGenerationRequest) -> int | None:
        if request.config.prefill_chunk_size is not None:
            return int(request.config.prefill_chunk_size)
        if self.config.prefill_chunk_size is not None:
            return int(self.config.prefill_chunk_size)
        return None

    def _prefers_speculative_request(self, request: ScheduledGenerationRequest) -> bool:
        if int(getattr(request.config, "num_speculative_tokens", 0)) <= 0:
            return False
        if int(getattr(request.config, "beam_size", 1)) > 1:
            return False
        method = str(getattr(request.config, "speculative_method", "")).strip().lower().replace("-", "_")
        if method in {"n_gram", "ngram", "suffix"}:
            return True
        return self.draft_model is not None

    def _scheduled_decode_tokens(self, request: ScheduledGenerationRequest) -> int:
        remaining = max(int(request.config.max_new_tokens) - int(request.generated_tokens), 0)
        if remaining <= 0:
            return 0
        return 1

    def _is_long_prefill(self, request: ScheduledGenerationRequest) -> bool:
        threshold = int(self.config.long_prefill_token_threshold)
        if threshold <= 0:
            return False
        return int(request.prompt_seq.shape[1]) > threshold

    def _scheduled_prefill_tokens(
        self,
        request: ScheduledGenerationRequest,
        token_budget: int,
    ) -> int:
        remaining = self._remaining_prefill_tokens(request)
        if remaining <= 0 or token_budget <= 0:
            return 0
        scheduled = min(int(remaining), int(token_budget))
        if request.cache is None:
            return scheduled if scheduled >= remaining else remaining
        chunk_limit = self._prefill_chunk_size(request)
        if chunk_limit is not None and chunk_limit > 0:
            scheduled = min(scheduled, int(chunk_limit))
        if self._is_long_prefill(request) and int(self.config.long_prefill_token_threshold) > 0:
            scheduled = min(scheduled, int(self.config.long_prefill_token_threshold))
        return max(int(scheduled), 1)

    def _schedule_compute_items(
        self,
        requests: list[ScheduledGenerationRequest],
        token_budget: int,
    ) -> tuple[list[ScheduledWorkItem], int]:
        ordered = sorted(
            [request for request in requests if not request.done and request.ready_logits is None],
            key=self._request_sort_key,
        )
        work_items: list[ScheduledWorkItem] = []
        consumed = 0
        partial_prefills = 0
        long_partial_prefills = 0
        max_batch_size = int(self.config.max_batch_size)

        for request in ordered:
            if len(work_items) >= max_batch_size or consumed >= int(token_budget):
                break
            remaining_budget = int(token_budget) - consumed
            prefill_remaining = self._remaining_prefill_tokens(request)
            if prefill_remaining > 0:
                if request.cache is None:
                    scheduled_tokens = prefill_remaining
                    if scheduled_tokens > remaining_budget and work_items:
                        continue
                    work_items.append(ScheduledWorkItem(kind="prefill", request=request, tokens=int(scheduled_tokens)))
                    consumed += int(scheduled_tokens)
                    partial_prefills += 1
                    if self._is_long_prefill(request):
                        long_partial_prefills += 1
                    continue
                if partial_prefills >= int(self.config.max_num_partial_prefills):
                    continue
                is_long = self._is_long_prefill(request)
                if is_long and long_partial_prefills >= int(self.config.max_long_partial_prefills):
                    continue
                scheduled_tokens = self._scheduled_prefill_tokens(request, remaining_budget)
                if scheduled_tokens <= 0:
                    continue
                work_items.append(ScheduledWorkItem(kind="prefill", request=request, tokens=int(scheduled_tokens)))
                consumed += int(scheduled_tokens)
                partial_prefills += 1
                if is_long:
                    long_partial_prefills += 1
                continue

            if request.cache is None:
                full_cost = max(int(request.seq.shape[1]), 1)
                if full_cost > remaining_budget and work_items:
                    continue
                work_items.append(ScheduledWorkItem(kind="full", request=request, tokens=full_cost))
                consumed += full_cost
                continue

            if remaining_budget <= 0:
                break
            decode_cost = min(self._scheduled_decode_tokens(request), remaining_budget)
            if decode_cost <= 0:
                continue
            work_items.append(ScheduledWorkItem(kind="decode", request=request, tokens=int(decode_cost)))
            consumed += int(decode_cost)

        return work_items, consumed

    def _work_group_signature(self, item: ScheduledWorkItem) -> tuple | None:
        request = item.request
        if item.kind == "prefill":
            if request.cache is None:
                return None
            if request.prefill_attention_mask is not None and request.prefill_attention_mask.ndim != 2:
                return None
            return (
                "prefill",
                str(request.cache_backend),
                int(item.tokens),
                int(self._cache_length(request.cache)),
            )
        if item.kind == "decode":
            if request.cache is None:
                return None
            if self._prefers_speculative_request(request):
                return None
            if request.attention_mask is not None and request.attention_mask.ndim != 2:
                return None
            return ("decode", str(request.cache_backend))
        return None

    def _dispatch_work_items(self, work_items: list[ScheduledWorkItem]) -> None:
        grouped: dict[tuple, list[ScheduledWorkItem]] = {}
        dispatch_order: list[tuple | ScheduledWorkItem] = []
        for item in work_items:
            signature = self._work_group_signature(item)
            if signature is None:
                dispatch_order.append(item)
                continue
            if signature not in grouped:
                grouped[signature] = []
                dispatch_order.append(signature)
            grouped[signature].append(item)

        for entry in dispatch_order:
            if isinstance(entry, ScheduledWorkItem):
                self._run_request_step(
                    entry.request,
                    prefill_tokens=(entry.tokens if entry.kind == "prefill" else None),
                )
                continue
            kind = entry[0]
            if kind == "prefill":
                self._process_prefill_group(grouped[entry])
            elif kind == "decode":
                self._process_decode_group(grouped[entry])

    def _pad_token_id(self, request: ScheduledGenerationRequest) -> int:
        pad_id = getattr(getattr(self.model, "cfg", object()), "pad_token_id", None)
        if pad_id is not None:
            return int(pad_id)
        if request.config.eos_id is not None:
            return int(request.config.eos_id)
        return 0

    def _build_padded_decode_batch(
        self,
        requests: list[ScheduledGenerationRequest],
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if any(request.attention_mask is not None and request.attention_mask.ndim != 2 for request in requests):
            return None, None
        max_len = max(int(request.seq.shape[1]) for request in requests)
        device = requests[0].seq.device
        dtype = requests[0].seq.dtype
        batch_seq = torch.full(
            (len(requests), max_len),
            self._pad_token_id(requests[0]),
            dtype=dtype,
            device=device,
        )
        batch_attention_mask = torch.zeros(
            (len(requests), max_len),
            dtype=torch.long,
            device=device,
        )
        for row_idx, request in enumerate(requests):
            row_len = int(request.seq.shape[1])
            batch_seq[row_idx, :row_len] = request.seq[0]
            if request.attention_mask is not None:
                batch_attention_mask[row_idx, :row_len] = request.attention_mask[0].to(device=device, dtype=torch.long)
            else:
                batch_attention_mask[row_idx, :row_len] = 1
        return batch_seq.contiguous(), batch_attention_mask.contiguous()

    def _trim_prefill_state(
        self,
        request: ScheduledGenerationRequest,
        consumed_tokens: int,
        final_logits: torch.Tensor | None,
    ) -> None:
        if request.prefill_ids is None:
            return
        remaining = int(request.prefill_ids.shape[1]) - int(consumed_tokens)
        if remaining > 0:
            request.prefill_ids = request.prefill_ids[:, int(consumed_tokens):].contiguous()
            if request.prefill_attention_mask is not None:
                request.prefill_attention_mask = request.prefill_attention_mask[:, int(consumed_tokens):].contiguous()
            return
        request.prefill_ids = None
        request.prefill_attention_mask = None
        if final_logits is None:
            raise RuntimeError("prefill completed without final logits")
        request.ready_logits = final_logits.detach().contiguous()
        self._store_prefix_cache(request, request.cache, request.ready_logits)

    def _process_ready_requests(self, requests: list[ScheduledGenerationRequest]) -> None:
        ready = [request for request in requests if request.ready_logits is not None and not request.done]
        if not ready:
            return
        speculative_ready = [request for request in ready if self._prefers_speculative_request(request)]
        regular_ready = [request for request in ready if not self._prefers_speculative_request(request)]
        if speculative_ready:
            try:
                speculative_results = speculative_decode_batch(
                    self.model,
                    self.draft_model,
                    requests=[
                        SpeculativeBatchRequest(
                            seq=request.seq,
                            attention_mask=request.attention_mask,
                            cache=request.cache,
                            config=request.config,
                            ready_logits=request.ready_logits,
                            prompt_seq=request.prompt_seq,
                            cache_backend=request.cache_backend,
                            remaining_new_tokens=int(request.config.max_new_tokens) - int(request.generated_tokens),
                        )
                        for request in speculative_ready
                    ],
                    cache_pagesize=self.cache_pagesize,
                    cache_backend=self.default_cache_backend,
                )
                fallback_ready: list[ScheduledGenerationRequest] = []
                for request, speculative in zip(speculative_ready, speculative_results):
                    if speculative is None or int(speculative.emitted_tokens) <= 0:
                        fallback_ready.append(request)
                        continue
                    self._apply_speculative_result(request, speculative)
                regular_ready.extend(fallback_ready)
            except Exception as exc:
                for request in speculative_ready:
                    try:
                        self._run_request_step(request)
                    except Exception:
                        self._fail_request(request, exc)
        if regular_ready:
            self._process_regular_ready_requests(regular_ready)

    def _process_regular_ready_requests(self, requests: list[ScheduledGenerationRequest]) -> None:
        ready = [request for request in requests if request.ready_logits is not None and not request.done]
        if not ready:
            return
        try:
            logits = torch.cat([request.ready_logits for request in ready], dim=0).contiguous()
            for request in ready:
                request.ready_logits = None
            self._sample_and_advance(ready, logits)
        except Exception as exc:
            for request in ready:
                self._fail_request(request, exc)

    def _process_prefill_group(self, items: list[ScheduledWorkItem]) -> None:
        requests = [item.request for item in items]
        scheduled_tokens = int(items[0].tokens) if items else 0
        try:
            if scheduled_tokens <= 0:
                return
            if any(item.tokens != scheduled_tokens for item in items):
                raise RuntimeError("batched prefill group requires a uniform scheduled token count")
            if any(request.prefill_attention_mask is not None and request.prefill_attention_mask.ndim != 2 for request in requests):
                raise RuntimeError("batched prefill only supports rank-2 token attention masks")

            prefill_ids = torch.cat(
                [request.prefill_ids[:, :scheduled_tokens].contiguous() for request in requests],
                dim=0,
            ).contiguous()
            attention_masks = []
            for request in requests:
                if request.prefill_attention_mask is not None:
                    attention_masks.append(request.prefill_attention_mask[:, :scheduled_tokens].contiguous())
                else:
                    attention_masks.append(
                        torch.ones(
                            (1, scheduled_tokens),
                            dtype=torch.long,
                            device=request.prefill_ids.device,
                        )
                    )
            batch_attention_mask = torch.cat(attention_masks, dim=0).contiguous() if attention_masks else None
            batch_cache = concat_kv_caches([request.cache for request in requests])
            session = RuntimeGenerationSession.from_model(
                self.model,
                prefill_ids,
                attention_mask=batch_attention_mask,
                cache=batch_cache,
                cache_pagesize=self.cache_pagesize,
                cache_backend=requests[0].cache_backend,
                trace=False,
            )
            next_logits = session.prefill_next_logits(chunk_size=None)
            cache_rows = split_kv_cache_rows(session.cache, [1 for _ in requests])
            logits_rows = next_logits.split(1, dim=0) if next_logits is not None else [None for _ in requests]
            for request, cache_row, logits_row in zip(requests, cache_rows, logits_rows):
                request.cache = cache_row
                self._trim_prefill_state(request, scheduled_tokens, logits_row)
        except Exception:
            for item in items:
                self._run_request_step(item.request, prefill_tokens=item.tokens)

    def _process_decode_group(self, items: list[ScheduledWorkItem]) -> None:
        requests = [item.request for item in items]
        try:
            batch_seq, batch_attention_mask = self._build_padded_decode_batch(requests)
            if batch_seq is None or batch_attention_mask is None:
                raise RuntimeError("batched decode only supports rank-2 token attention masks")
            batch_cache = concat_kv_caches([request.cache for request in requests])
            session = RuntimeGenerationSession(
                model=self.model,
                seq=batch_seq,
                attention_mask=batch_attention_mask,
                cache=batch_cache,
                trace=False,
            )
            next_logits = session.decode_next_logits()
            if next_logits is None:
                raise RuntimeError("batched decode returned no logits")
            cache_rows = split_kv_cache_rows(session.cache, [1 for _ in requests])
            for request, cache_row in zip(requests, cache_rows):
                request.cache = cache_row
            self._sample_and_advance(requests, next_logits)
        except Exception:
            for item in items:
                self._run_request_step(item.request)

    def _apply_speculative_result(
        self,
        request: ScheduledGenerationRequest,
        result,
    ) -> None:
        request.ready_logits = result.ready_logits
        request.seq = result.seq
        request.attention_mask = result.attention_mask
        request.cache = result.cache
        request.generated_tokens += int(result.emitted_tokens)
        if request.config.sliding_window is not None and request.cache is not None:
            try:
                evict_kv_cache(request.cache, int(request.config.sliding_window), policy="sliding-window")
            except Exception:
                pass
        eos_hit = False
        if request.config.eos_id is not None and int(result.emitted_tokens) > 0:
            eos_tail = request.seq[:, -int(result.emitted_tokens):]
            eos_hit = bool((eos_tail == int(request.config.eos_id)).any().item())
        limit_hit = request.generated_tokens >= int(request.config.max_new_tokens)
        if eos_hit or limit_hit:
            self._finish_request(request, request.seq)

    def _sample_and_advance(self, requests: list[ScheduledGenerationRequest], logits: torch.Tensor) -> None:
        for request, logit_row in zip(requests, logits.split(1, dim=0)):
            cfg = request.config
            next_id = runtime_sample_with_policies(
                logit_row.contiguous(),
                request.seq,
                do_sample=bool(cfg.do_sample),
                temperature=float(cfg.temperature),
                top_k=(int(cfg.top_k) if cfg.top_k is not None else None),
                top_p=(float(cfg.top_p) if cfg.top_p is not None else None),
                no_repeat_ngram=int(cfg.no_repeat_ngram),
                repetition_penalty=float(cfg.repetition_penalty),
                presence_penalty=float(cfg.presence_penalty),
                frequency_penalty=float(cfg.frequency_penalty),
            )
            if next_id.ndim == 1:
                next_id = next_id.unsqueeze(-1)
            request.seq = torch.cat([request.seq, next_id.contiguous()], dim=1)
            if request.attention_mask is not None:
                ones = torch.ones((1, next_id.shape[1]), dtype=request.attention_mask.dtype, device=request.attention_mask.device)
                request.attention_mask = torch.cat([request.attention_mask, ones], dim=1)
            request.generated_tokens += int(next_id.shape[1])
            if request.config.sliding_window is not None and request.cache is not None:
                try:
                    evict_kv_cache(request.cache, int(request.config.sliding_window), policy="sliding-window")
                except Exception:
                    pass
            eos_hit = request.config.eos_id is not None and bool((next_id == int(request.config.eos_id)).all().item())
            limit_hit = request.generated_tokens >= int(request.config.max_new_tokens)
            if eos_hit or limit_hit:
                self._finish_request(request, request.seq)

    def _store_prefix_cache(
        self,
        request: ScheduledGenerationRequest,
        cache,
        next_logits: torch.Tensor,
    ) -> None:
        if cache is None:
            return
        try:
            cached = self._clone_single_row_cache(cache)
            with self._prefix_cache_lock:
                self.prefix_cache.store(
                    request.prompt_seq[0],
                    request.prompt_attention_mask[0] if request.prompt_attention_mask is not None else None,
                    request.cache_backend,
                    cached,
                    next_logits,
                )
        except Exception:
            pass

    def _run_request_step(
        self,
        request: ScheduledGenerationRequest,
        *,
        prefill_tokens: int | None = None,
    ) -> None:
        if request.done:
            return
        try:
            if request.prefill_ids is not None:
                remaining = int(request.prefill_ids.shape[1])
                scheduled_tokens = remaining if prefill_tokens is None else min(max(int(prefill_tokens), 1), remaining)
                chunk_ids = request.prefill_ids[:, :scheduled_tokens].contiguous()
                chunk_mask = None
                if request.prefill_attention_mask is not None:
                    chunk_mask = request.prefill_attention_mask[:, :scheduled_tokens].contiguous()
                session = RuntimeGenerationSession.from_model(
                    self.model,
                    chunk_ids,
                    attention_mask=chunk_mask,
                    cache=request.cache,
                    cache_pagesize=self.cache_pagesize,
                    cache_backend=request.cache_backend,
                    trace=False,
                )
                logits = session.prefill_next_logits(chunk_size=None)
                request.cache = session.cache
                final_logits = logits if scheduled_tokens >= remaining else None
                self._trim_prefill_state(request, scheduled_tokens, final_logits)
                return

            if self._prefers_speculative_request(request):
                speculative = speculative_decode_step(
                    self.model,
                    self.draft_model,
                    request.seq,
                    attention_mask=request.attention_mask,
                    cache=request.cache,
                    config=request.config,
                    ready_logits=request.ready_logits,
                    prompt_seq=request.prompt_seq,
                    remaining_new_tokens=int(request.config.max_new_tokens) - int(request.generated_tokens),
                    cache_pagesize=self.cache_pagesize,
                    cache_backend=request.cache_backend,
                )
                if speculative is not None and int(speculative.emitted_tokens) > 0:
                    self._apply_speculative_result(request, speculative)
                    return

            if request.ready_logits is not None:
                self._process_regular_ready_requests([request])
                return

            if request.cache is not None:
                session = RuntimeGenerationSession(
                    model=self.model,
                    seq=request.seq,
                    attention_mask=request.attention_mask,
                    cache=request.cache,
                    trace=False,
                )
                logits = session.decode_next_logits()
                if logits is None:
                    raise RuntimeError("decode step returned no logits")
                request.cache = session.cache
                self._sample_and_advance([request], logits)
                return

            session = RuntimeGenerationSession(
                model=self.model,
                seq=request.seq,
                attention_mask=request.attention_mask,
                cache=None,
                trace=False,
            )
            logits = session.full_next_logits()
            self._sample_and_advance([request], logits)
        except Exception as exc:
            self._fail_request(request, exc)

    def _finish_request(self, request: ScheduledGenerationRequest, result: torch.Tensor) -> None:
        request.result = result.detach().clone()
        request.done = True
        request.event.set()

    def _fail_request(self, request: ScheduledGenerationRequest, exc: BaseException) -> None:
        request.error = exc
        request.done = True
        request.event.set()
