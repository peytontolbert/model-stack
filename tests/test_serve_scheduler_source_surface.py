from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_runtime_generation_exposes_chunked_prefill_and_cache_concat_helpers() -> None:
    generation_source = _read("runtime/generation.py")
    kv_cache_source = _read("runtime/kv_cache.py")

    assert "prefill_chunk_size: Optional[int] = None" in generation_source
    assert "num_speculative_tokens: int = 0" in generation_source
    assert "speculative_method: str | None = None" in generation_source
    assert 'rejection_sample_method: str = "strict"' in generation_source
    assert "prompt_lookup_min: int = 2" in generation_source
    assert "prompt_lookup_max: int = 4" in generation_source
    assert "suffix_decoding_max_tree_depth: int = 32" in generation_source
    assert "suffix_decoding_max_spec_factor: float = 2.0" in generation_source
    assert "suffix_decoding_min_token_prob: float = 0.0" in generation_source
    assert "typical_acceptance_sampler_posterior_threshold: float = 0.09" in generation_source
    assert "typical_acceptance_sampler_posterior_alpha: float = 0.3" in generation_source
    assert "class SpeculativeDecodeResult:" in generation_source
    assert "class SpeculativeProposal:" in generation_source
    assert "class SpeculativeBatchRequest:" in generation_source
    assert "def _select_last_token_logits(" in generation_source
    assert "def _slice_prefill_attention_mask(" in generation_source
    assert "def _supports_speculative_decode(" in generation_source
    assert "def speculative_decode_batch(" in generation_source
    assert "def speculative_decode_step(" in generation_source
    assert "def _token_mask_lengths(" in generation_source
    assert "def _token_mask_has_padding(" in generation_source
    assert "def _resolve_decode_step_inputs(" in generation_source
    assert "def _build_target_speculative_probs(" in generation_source
    assert "def _finalize_speculative_result(" in generation_source
    assert "def prefill_next_logits(self, *, chunk_size: int | None = None)" in generation_source
    assert "session.prefill_next_logits(" in generation_source
    assert 'chunk_size=getattr(cfg, "prefill_chunk_size", None)' in generation_source
    assert "if not _token_mask_has_padding(seq, attention_mask):" in generation_source
    assert "decode_tokens, pos_ids, cache_pos = _resolve_decode_step_inputs(self.seq, self.attention_mask)" in generation_source
    assert "def speculative_accept(" in _read("runtime/ops.py")
    assert "speculative_accept_forward" in _read("runtime/csrc/model_stack_native.cpp")
    assert "def concat_kv_caches(caches: list[object | None]):" in kv_cache_source
    assert "def split_kv_cache_rows(cache, row_sizes: list[int]):" in kv_cache_source
    assert "def truncate_kv_cache_prefix(cache, max_tokens: int):" in kv_cache_source
    assert '"concat_kv_caches"' in kv_cache_source
    assert '"split_kv_cache_rows"' in kv_cache_source
    assert '"truncate_kv_cache_prefix"' in kv_cache_source


def test_serve_runtime_routes_single_requests_through_scheduler_and_prefix_cache() -> None:
    api_source = _read("serve/api.py")
    runtime_source = _read("serve/runtime.py")
    scheduler_source = _read("serve/scheduler.py")
    readme_source = _read("serve/README.md")

    assert "class SchedulerConfig:" in scheduler_source
    assert "class PrefixCache:" in scheduler_source
    assert "class GenerationScheduler:" in scheduler_source
    assert "max_num_batched_tokens: int = 2048" in scheduler_source
    assert "max_num_partial_prefills: int = 1" in scheduler_source
    assert "max_long_partial_prefills: int = 1" in scheduler_source
    assert "long_prefill_token_threshold: int = 0" in scheduler_source
    assert 'scheduling_policy: str = "fcfs"' in scheduler_source
    assert "def lookup(" in scheduler_source
    assert "def store(" in scheduler_source
    assert "def _schedule_compute_items(" in scheduler_source
    assert "def _dispatch_work_items(" in scheduler_source
    assert 'kind="prefill"' in scheduler_source
    assert 'kind="decode"' in scheduler_source
    assert "speculative_decode_step" in scheduler_source
    assert "speculative_decode_batch" in scheduler_source
    assert "SpeculativeBatchRequest" in scheduler_source
    assert "def _prefers_speculative_request(" in scheduler_source
    assert "def _scheduled_decode_tokens(" in scheduler_source
    assert "def _apply_speculative_result(" in scheduler_source
    assert "concat_kv_caches" in scheduler_source
    assert "split_kv_cache_rows" in scheduler_source
    assert "self.prefix_cache.lookup(" in scheduler_source
    assert "self._process_prefill_group(" in scheduler_source
    assert "self._process_decode_group(" in scheduler_source
    assert "num_speculative_tokens: Optional[int] = None" in api_source
    assert 'priority: int = 0' in api_source
    assert "def _build_padded_decode_batch(" in scheduler_source
    assert "runtime_sample_with_policies(" in scheduler_source
    assert "request.seq," in scheduler_source
    assert "self._sample_and_advance(" in scheduler_source
    assert "self.scheduler_config = SchedulerConfig.from_env()" in runtime_source
    assert "MODEL_STACK_DRAFT_MODEL_DIR" in runtime_source
    assert "MODEL_STACK_NUM_SPECULATIVE_TOKENS" in runtime_source
    assert "MODEL_STACK_SPECULATIVE_METHOD" in runtime_source
    assert "MODEL_STACK_SPEC_DECODING_ACCEPTANCE_METHOD" in runtime_source
    assert '"speculative": {' in runtime_source
    assert "def scheduler(self) -> GenerationScheduler | None:" in runtime_source
    assert "self.scheduler.submit(" in runtime_source
    assert 'priority=int(priority)' in runtime_source
    assert "prefill_chunk_size: Optional[int] = None" in api_source
    assert "num_speculative_tokens=req.num_speculative_tokens" in api_source
    assert "speculative_method: Optional[str] = None" in api_source
    assert "rejection_sample_method: Optional[str] = None" in api_source
    assert "prompt_lookup_min: int = 2" in api_source
    assert "suffix_decoding_max_tree_depth: int = 32" in api_source
    assert "typical_acceptance_sampler_posterior_threshold: float = 0.09" in api_source
    assert "request scheduler" in readme_source
    assert "serve/scheduler.py owns the request queue" in readme_source
