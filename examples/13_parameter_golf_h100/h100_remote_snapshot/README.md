## H100 Remote Snapshot

This directory preserves runtime files that differed on `/root/transformer_10_h100`
when checked from the H100. They are kept as a recovery snapshot, not merged into
the active tree.

The snapshot mostly contains an older SM80 attention prefill experiment. It
conflicts with the current local BitNet decode and paged-SDPA bridge work, so it
should be inspected before reuse instead of copied over wholesale.

Saved files:

- `runtime/ops.py`
- `runtime/csrc/backend/attention/cuda_attention_prefill.cuh`
- `runtime/csrc/backend/attention/cuda_attention_sm80_inference_prefill.cu`
- `runtime/csrc/policy/attention_policy.h`
- `tests/bench_sm80_attention_scaling.py`
- `tests/test_runtime_attention_sm80_cuda_parity.py`
- `tests/test_runtime_attention_source_surface.py`
- `tests/test_runtime_ops_dispatch.py`
- `tests/test_runtime_quant_source_surface.py`

The `root_copies/` directory preserves two stray top-level files from
`/root/transformer_10_h100` that differed from the active local files:

- `root_copies/bench_parameter_golf_bitnet_export.py`
- `root_copies/quantization.py`
