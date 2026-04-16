# llama3b-asm — Production-Grade CPU Inference for Llama-3B with Assembly Kernels + Transformers Parity Tests

`llama3b-asm` is a **production-grade CPU inference engine** for **Llama-3B-class** decoder-only models, built around:
- a **memory-mappable packed weight format** for fast startup and predictable memory behavior,
- a **token-by-token decode path** with **KV cache** optimized for single-batch interactive inference,
- **hand-tuned assembly / SIMD micro-kernels** (AVX2 baseline; optional AVX-512/VNNI),
- a **test harness that compares generation against HuggingFace Transformers** for semantic parity.

The engine is designed to be **verifiably correct** (via parity and regression tests) and **fast** (via reduced overhead + packed weights + vectorized kernels).

---

## Why this exists

Torch/Transformers are excellent, but for production inference you often want:
- fewer layers of abstraction,
- predictable memory layout and mmap startup,
- specialized kernels for the “decode 1 token at a time” workload,
- strict regression tests that prevent silent drift.

This project focuses on that niche: **CPU decode performance** with **deterministic correctness guarantees**.

---

## Architecture Overview

```mermaid
flowchart LR
  A[Tokenizer] --> B[Token IDs]
  B --> C[Decode Loop: 1 token/step]
  C --> D[Forward: Embedding + N Layers + Final Head]
  D --> E[Logits]
  E --> F[Sampler]
  F --> C

  subgraph Hot Kernels (ASM/SIMD)
    K1[MatVec / GEMV]
    K2[RMSNorm]
    K3[RoPE]
    K4[Softmax]
  end

  D --> K1
  D --> K2
  D --> K3
  D --> K4
```
## Forward Pass (Llama-family)
Each layer implements:

RMSNorm → QKV projections → RoPE → attention with KV cache → output projection + residual

RMSNorm → SwiGLU MLP (W1/W3, SiLU ⊙, W2) + residual

## Parity Contract (what “matches Transformers” means)
Exact bitwise parity with Torch is usually unrealistic once you introduce:

different reduction orders (floating-point associativity),

fused ops,

approximate exp/silu,

FP16 / quantization.

So this project defines production parity levels:

- **P0 — Token parity (primary goal)**: for the same prompt + tokenizer + decoding settings, the engine produces the same token IDs as Transformers for N steps.
- **P1 — Logit parity within tolerance (debug/verification goal)**: logits match within a strict tolerance when both sides are run in FP32 on CPU (e.g. atol=1e-5, rtol=1e-5).

In production, you should expect P0 to hold even when you switch to FP16 weights or quantization, while P1 may not.

## Features
✅ Packed model format (.pack) with tensor index + aligned blobs; supports mmap() zero-copy weights.

✅ Single-token decode loop with layer-wise KV cache.

✅ Assembly/SIMD micro-kernels:

AVX2 baseline (x86-64)

optional AVX-512/VNNI (if available)

✅ Transformers parity harness:

greedy decode token parity

logit parity mode (FP32)

✅ Deterministic regression tests with golden prompts/tokens.

✅ Benchmark runner (tokens/sec, latency, memory).

## Supported Platforms
CPU / OS
Linux x86-64 (recommended; primary target)

macOS x86-64 possible (toolchain differences)

Windows possible but not a primary target (ABI/assembler friction)

## Instruction Sets
AVX2: required for best performance; fallback reference kernels can be built for portability (slower).

AVX-512/VNNI: optional fast path.

## Status in this repo
This document currently describes a **target design**. The actual `llama3b-asm/` CMake project and `.S` kernels referenced below are **not implemented in this repository yet**.

If/when we implement it here, it should integrate with the existing `kernel/` registry (so attention/rope/norm can be swapped without churn) and the existing `serve/` decode loop APIs.

## Proposed (future) repository layout
```text
llama3b-asm/
  README.md
  CMakeLists.txt
  include/
    config.h
    tensor.h
    runtime.h
    llama.h
    kernels.h
  src/
    main.cpp
    loader_pack.cpp
    llama_forward.cpp
    kv_cache.cpp
    sampler.cpp
    tokenizer_bridge.cpp
    ops_ref.cpp
  asm/
    x86/
      matvec_f32_avx2.S
      rmsnorm_avx2.S
      rope_avx2.S
      softmax_avx2.S
      # optional:
      matvec_f16_avx2.S
      matvec_int8_vnni.S
  tools/
    export_hf_to_pack.py
    parity_harness.py
    benchmark.py
  tests/
    prompts.jsonl
    golden_tokens.jsonl
    pytest.ini
```

## Getting Started
### 1) Build
### Requirements
CMake ≥ 3.20

clang or gcc (x86-64)

Python 3.10+ (for tooling/tests)

pip install -r tools/requirements.txt (Transformers, safetensors, torch CPU)

### Build (Release)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### CPU feature selection
At runtime, the engine selects the best available kernel set:

AVX-512/VNNI → AVX2 → reference C

You can force a specific path for testing:

```bash
./build/llama3b-asm --force-kernels=ref
./build/llama3b-asm --force-kernels=avx2
```

## Model Export: HuggingFace → .pack
The engine loads a single .pack file.

### 2) Export weights

```bash
python tools/export_hf_to_pack.py \
  --hf_model /path/to/llama-3b \
  --out models/llama3b.pack \
  --dtype fp32 \
  --layout row_major \
  --pad_to 64
```
Parity recommendation: start with --dtype fp32 to validate P1 logit parity.

Once validated, you can export FP16 weights to improve bandwidth:

```bash
python tools/export_hf_to_pack.py \
  --hf_model /path/to/llama-3b \
  --out models/llama3b-f16.pack \
  --dtype fp16 \
  --layout row_major \
  --pad_to 64
```
## Tokenization
For production correctness, tokenizer compatibility is non-negotiable.

This project supports two modes:

Tokenizer Bridge (recommended initially)
Uses HuggingFace tokenizer from Python in the test harness and optionally at runtime via a bridge.

Native Tokenizer (optional)
A native tokenizer implementation matching the model’s tokenizer spec.

Production guidance: keep tokenizer behavior pinned and regression-tested. A tokenizer mismatch will look like “model bug” but is actually preprocessing drift.

## Running Inference
### Basic run (greedy)

```bash
./build/llama3b-asm \
  --model models/llama3b.pack \
  --prompt "Hello world" \
  --steps 64 \
  --temperature 0
```

### Sampling

```bash
./build/llama3b-asm \
  --model models/llama3b.pack \
  --prompt "Write a short story about a robot." \
  --steps 128 \
  --temperature 0.8 \
  --top_p 0.95 \
  --top_k 40
```

## Testing Against Transformers (Parity Harness)
The parity harness runs Transformers generation and compares results against the engine.

Deterministic settings (recommended)
Use greedy decoding for stable token parity:

temperature=0

top_k=0

top_p=1.0

fixed max steps

CPU only

### 1) Install tooling dependencies

```bash
pip install -r tools/requirements.txt
```

### 2) Token parity test (P0)

```bash
python tools/parity_harness.py \
  --hf_model /path/to/llama-3b \
  --engine ./build/llama3b-asm \
  --pack models/llama3b.pack \
  --prompts tests/prompts.jsonl \
  --steps 128 \
  --mode token \
  --temperature 0
```
Expected outcome:

identical token IDs for each prompt for N steps

diffs reported with the first mismatch index and surrounding context

### 3) Logit parity test (P1, FP32)
Run only when your engine is using FP32 weights and reference math:

```bash
python tools/parity_harness.py \
  --hf_model /path/to/llama-3b \
  --engine ./build/llama3b-asm \
  --pack models/llama3b.pack \
  --prompts tests/prompts.jsonl \
  --steps 32 \
  --mode logits \
  --atol 1e-5 \
  --rtol 1e-5
```

## Test Suite (CI-friendly)
### Pytest regression tests

```bash
pytest -q
```
Recommended test categories:

test_pack_format.py: validates tensor table integrity and alignment

test_kv_cache.py: verifies KV cache indexing and persistence

test_rope.py: verifies RoPE matches reference within tolerance

test_layer0_smoke.py: runs 1-layer forward against a reference

test_token_parity.py: end-to-end generation parity (greedy)

Golden tests
The repository includes:

tests/prompts.jsonl: canonical prompts

tests/golden_tokens.jsonl: expected outputs under pinned settings

To update goldens (explicit + reviewed only):

```bash
python tools/parity_harness.py --write-golden ...
```

## Benchmarking
### Tokens/sec + latency

```bash
python tools/benchmark.py \
  --engine ./build/llama3b-asm \
  --pack models/llama3b-f16.pack \
  --prompt_len 128 \
  --gen_len 256 \
  --runs 10
```
Metrics emitted:

first-token latency (ms)

steady-state tokens/sec

RSS and KV cache footprint

kernel path used (avx2/avx512/ref)

## Production Hardening Checklist
### Correctness and drift prevention
 Pin exact HF model revision (commit hash) used to export .pack

 Pin tokenizer artifacts (tokenizer.json etc.)

 Run P0 token parity in CI for golden prompts

 Run P1 logit parity in nightly (FP32 reference mode)

### Performance and stability
 Use mmap for .pack weights in production

 Align weights and scratch buffers (64B)

 Threading policy documented (per head vs per output channel)

 NUMA-aware KV cache allocation (for multi-socket)

### Safety
 Assembly kernels covered by fuzzable unit tests (bounds checks at the C boundary)

 Sanitizer builds for dev (ASAN/UBSAN) using reference kernels

 Clear fallback path when CPU features are absent

## Troubleshooting
### Token mismatch vs Transformers
Common causes:

tokenizer mismatch (most common)

RoPE theta/scaling mismatch

GQA head mapping bug (n_heads vs n_kv_heads)

RMSNorm epsilon mismatch

sampling parameter mismatch

weight transpose/layout error during export

Suggested debug path:

switch engine to --force-kernels=ref

run --mode logits P1 test in FP32

bisect layer-by-layer with optional debug dumps

### Performance worse than Torch
Likely reasons:

using reference kernels (no AVX2 path)

not using packed layout / mmap

small batch sizes with heavyweight allocations per step

attention loop not optimized (dot products not vectorized)

running with debug flags / asserts

## Configuration Notes
### KV Cache memory sizing
KV cache grows with:

layers × max_seq × n_kv_heads × head_dim × sizeof(dtype) × 2

Plan your max context accordingly and expose it as a runtime flag:

```bash
./build/llama3b-asm --max_seq 4096 ...
```
### Threading model
Recommended default for decode:

parallelize matvec output channels (projection) and/or attention heads

avoid oversubscription; respect OMP_NUM_THREADS or a project-specific flag

## Security & Reliability
This project executes low-level code paths (SIMD/assembly). Production deployment should:

run under standard hardening flags in release builds where possible,

use clear ABI boundaries,

validate .pack metadata before mapping tensor pointers,

fail closed on corrupted model files.

## License
This repository is Apache-2.0; keep the engine code under Apache-2.0 unless you have a specific reason to change it. Also ensure your model usage complies with the model’s license terms.

## Roadmap (typical)
 FP32 parity-complete baseline (P0 + P1)

 AVX2 kernels for matvec/norm/rope/softmax (maintain P0)

 FP16 weights with FP32 accum (maintain P0; measure drift)

 INT8/INT4 quantized projections (P0-focused; eval quality)

 Kernel fusion + cache blocking for attention

 Full native tokenizer (optional; parity-tested)