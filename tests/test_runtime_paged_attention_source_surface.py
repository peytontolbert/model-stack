from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


def test_python_runtime_routes_decode_through_paged_attention_op() -> None:
    ops_source = _read("runtime/ops.py")
    cache_source = _read("runtime/cache.py")
    kv_cache_source = _read("runtime/kv_cache.py")
    attn_source = _read("runtime/attention_modules.py")
    native_py_source = _read("runtime/native.py")
    runtime_init_source = _read("runtime/__init__.py")

    assert '"paged_attention_decode",' in native_py_source
    assert '"paged_attention_decode": "runtime.kv_cache"' in runtime_init_source
    assert "def supports_paged_attention_decode(self) -> bool:" in cache_source
    assert "def paged_attention_decode(" in cache_source
    assert "runtime_paged_attention_decode(" in cache_source
    assert "def paged_attention_decode(" in kv_cache_source
    assert "cache.append_batch(layer_idx, k_chunk, v_chunk, block_ids=block_ids)" in kv_cache_source
    assert "runtime_paged_attention_decode(" in kv_cache_source
    assert "def paged_attention_decode(" in ops_source
    assert "module.paged_attention_decode_forward(" in ops_source
    assert "invalid = positions >= lengths_long.view(-1, 1, 1, 1)" in ops_source
    assert "return attention(q, k, v, attn_mask=merged_mask, is_causal=False, scale=scale)" in ops_source
    assert 'getattr(cache, "supports_paged_attention_decode", lambda: False)()' in attn_source
    assert "cache.paged_attention_decode(" in attn_source
    assert "exception in paged decode attention path; falling back" in attn_source


def test_native_runtime_exposes_paged_attention_decode_kernel_surface() -> None:
    native_source = _read("runtime/csrc/model_stack_native.cpp")
    cuda_source = _read("runtime/csrc/backend/cuda_attention.cu")

    assert '{"paged_attention_decode", true}' in native_source
    assert '"paged_attention_decode", "attention_decode"' in native_source
    assert "torch::Tensor PagedAttentionDecodeForward(" in native_source
    assert "torch::Tensor CudaPagedAttentionDecodeForward(" in native_source
    assert "bool HasCudaPagedAttentionDecodeKernel();" in native_source
    assert "bool HasCudaPagedAttentionDecodeKernel() {" in native_source
    assert "NativeCachePagedAttentionDecodeLayer(" in native_source
    assert "return PagedAttentionDecodeForward(q, k_pages, v_pages, block_table, lengths, normalized_mask, scale);" in native_source
    assert 'm.def("paged_attention_decode_forward", &PagedAttentionDecodeForward' in native_source
    assert "paged_decode_attention_q1_forward_kernel" in cuda_source
    assert "paged_decode_attention_q1_hdim_sm90_forward_kernel" in cuda_source
    assert "MODEL_STACK_PAGED_DECODE_THREADS" in cuda_source
    assert "SelectPagedDecodeThreads(mask_seq, q_contig.size(3))" in cuda_source
    assert "if (head_dim <= 128)" in cuda_source
    assert "denom = denom * expf(row_max - score) + 1.0f" in cuda_source
    assert "TryLaunchPagedDecodeAttentionQ1Sm90SpecializedForHeadDim" in cuda_source
    assert "mask_seq < 512" in cuda_source
    assert "auto out = torch::empty_like(q_contig);" in cuda_source
    assert "LaunchPagedDecodeAttentionQ1" in cuda_source
    assert "bool HasCudaPagedAttentionDecodeKernel()" in cuda_source
    assert "torch::Tensor CudaPagedAttentionDecodeForward(" in cuda_source
    assert "block_table_contig" in cuda_source
    assert '"model_stack_cuda_paged_attention_decode_forward"' in cuda_source


def test_native_runtime_exposes_cuda_speculative_accept_kernel_surface() -> None:
    native_source = _read("runtime/csrc/model_stack_native.cpp")
    cuda_sampling_source = _read("runtime/csrc/backend/cuda_sampling.cu")
    ops_source = _read("runtime/ops.py")

    assert "torch::Tensor CudaSampleWithPoliciesForward(" in native_source
    assert "CudaSpeculativeAcceptForward(" in native_source
    assert 'm.def("speculative_accept_forward", &SpeculativeAcceptForward' in native_source
    assert "speculative_accept_forward_kernel" in cuda_sampling_source
    assert "CudaSpeculativeAcceptForward(" in cuda_sampling_source
    assert "SpeculativeSampleSeed(" in cuda_sampling_source
    assert "BlockArgmaxProbIndex(" in cuda_sampling_source
    assert "BlockEntropy(" in cuda_sampling_source
    assert "policy.row_reduce_threads" in cuda_sampling_source
    assert "SampleResidualProbIndex(" in cuda_sampling_source
    assert "module.speculative_accept_forward(" in ops_source
