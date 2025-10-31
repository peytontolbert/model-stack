### Description

This repository is a modular, production‑oriented model stack that cleanly separates the concerns required to train, serve, and evaluate large language models. It defines stable contracts (versioned configs, tensor/shape specs, KV‑cache API, kernel registry, checkpoints, and streaming data) and organizes implementations into focused domains (attention, blocks, data, training, serving, evaluation, kernels, distribution, export, and observability), with supporting areas for compression, autotuning, interpretability, governance, RAG, and packaging. The split enables swappable implementations, targeted performance work, and reproducible artifacts without churn across the rest of the system.

Current constellation (by plane)
Foundations (math, shapes, kernels)

attn — MHA/GQA/MQA, masks, KV-cache interface/paging shims

autotune — HP search, schedule factories, kernel picking

blocks — norms/MLP/residual wiring, block policies (pre/post-norm)

compress — quant/prune/LoRA deltas, KV compaction

corpus — corpus curation, dedup/PII/licensing, manifests

data — tokenizers, iterable datasets, sharding/mmap/collation

dist — DDP/FSDP/ZeRO strategies, offload, launchers

eval — benchmarks, metrics, perf/latency harness

examples — e2e demos & smoke tests

experiments — research scripts, configs, and ablations

export — TorchScript/ONNX/TensorRT, PTQ/FP8, model-card w/ hashes

governance — lineage, SBOM, license packs, reproducibility receipts

interpret — logit lens, causal tracing, SAE/feature dictionaries

kernel — Triton/CUDA/Flash-Attn adapters + registry

model — Encoder/Decoder/LM-Head assemblers, checkpoint I/O

pack — SDK/CLIs/templates, config validators, env bootstrap

rag — retrieval bridges (indexes, chunkers, caches, tool spec)

registry — artifact registry (promotions, retention, signatures)

rl — SFT/DPO/ORPO/RLHF trainers, reward models, preference data I/O

safety — policy-as-code, guardrails, red-team suites

serve — decode loops (greedy/beam/speculative), paged caches, server

specs — configs, tensor/shape/dtype contracts, schema/versioning

tensor — stateless numerics (init, masking, RoPE/ALiBi, residual math)

train — trainers, optim/schedules, AMP, gradient ckpt

viz — scalars, activation probes, profiler traces (TB/W&B/CSV)



Model graph (attention → blocks → model)


Data, training, serving, eval

Scale, artifactization, observability

Safety, data quality, compression, autotune, interpretability, governance, DX


-----------------------
Core Contracts (stable across repos)

Config: versioned @dataclass schemas in specs. JSON/YAML load/dump; schema migration utilities.

TensorSpec: shape/dtype/layout descriptors to prevent drift ((B,T,D), (H,KV,T,Dh), etc.).

Cache API: opaque KVCache handle (append/read/evict/paging) with no globals.

Kernel Registry: kernel exposes KernelRegistry looked up by symbolic names in attn.

Checkpoint: model writes/reads a layout-aware state dict + ModelCard metadata.

StreamingData: data yields Batch objects with masks/positions normalized to specs.


Assembly Paths
Inference Path (Runtime)
sequenceDiagram
  participant D as data
  participant M as model
  participant A as attn
  participant K as kernel
  participant S as serve

  D->>M: Batch(input_ids, attn_mask)
  M->>A: qkv + mask per block
  A->>K: resolve("flash") → fused kernel
  K-->>A: attn_out
  A-->>M: attn_out
  M-->>S: logits
  S-->>S: decode loop (KV cache)

Training Path (Orchestration)
flowchart LR
  G[data Dataloader] --> T[train Trainer]
  F[model TransformerLM] --> T
  T -->|loss/ckpt| E[eval Metrics]
  T -->|state_dict| F

Why this split scales

Swappable implementations without churn: attention rewrites (e.g., Flash-Decoding, paged KV) don’t perturb blocks/model/training.

Performance experiments are sandboxed: kernel can evolve (Triton/CUDA) with a stable registry shim.

Interop is explicit: one source of truth for shapes/config (specs), one for checkpoints (model).

Operational clarity: training (train) vs. serving (serve) evolve independently (very different constraints).

Versioning & Compatibility Matrix

specs MAJOR bumps ripple outward (requires minor bumps elsewhere).

kernel may add implementations without breaking attn as long as names/semantics hold.

model checkpoint ModelCard.version gates loader behavior; provide migration utilities (v1→v2).

serve guarantees ABI for generate() (sampler interface) across minor versions.

CI/CD Touchpoints (per-repo)

Unit: shapes, dtype flows, numerical invariants (grad checks in tensor, KV append/read in attn).

Property: determinism under fixed seeds, loss non-increase with no-op migrations.

Perf: microbenchmarks pinned in kernel + macro e2e in eval.

Artifacts: wheels w/ CUDA extras for kernel (pip install kernel[cuda12x]).

Example: Building a small LM from the 10 repos
from specs.config import ModelConfig
from model.lm import TransformerLM
from data.loader import build_dataloader
from train.trainer import Trainer
from serve.generate import generate
from eval.metrics import perplexity

cfg = ModelConfig(d_model=512, n_heads=8, n_layers=8, d_ff=2048, vocab_size=32000, attn_impl="flash")

model = TransformerLM(cfg).cuda()
dl = build_dataloader("/corpus/shards", batch_size=8, seq_len=1024)
trainer = Trainer(model, cfg, dl)

for step, batch in enumerate(dl):
    loss = trainer.step(batch)
    if step % 100 == 0:
        print("ppl:", perplexity(loss))

# Inference
out = generate(model, batch.input_ids[:1].cuda(), max_new_tokens=64)

Extensions (future repos if you need them)

dist (distributed/FSdp wrappers), export (ONNX/TensorRT), viz (activation probes). But the 10 above are sufficient for a clean, production-oriented split.

Quickstart Snippets
1) Scale training across 8 GPUs (FSDP + bf16)
torchrun --nproc_per_node=8 -m train.run \
  --config cfgs/llm_small.yaml \
  --dist.strategy FSDP --dist.precision bf16 --dist.grad_ckpt true \
  --viz.backend csv --viz.profile_every_n_steps 500

2) Export ONNX + int8 PTQ for server
from export.exporter import export
from specs.export import ExportConfig
out = export(model, ExportConfig(target="onnx", opset=19, quantize="int8", dynamic_axes=True, outdir="artifacts/"), vocab_size=32000, d_model=cfg.d_model)
print("artifact:", out)  # artifacts/model.onnx + modelcard.json

3) Instrument a serve loop
from viz.session import VizSession
viz = VizSession(cfg.viz)
tokens = generate_instrumented(model, input_ids, viz=viz, max_new_tokens=128, step=global_step)

Why these three repos help

dist lets you chase bigger context / deeper stacks without refactoring your trainer—just a strategy switch.

export gives repeatable, verifiable deployables with hashes and model cards—critical for supply-chain trust.

viz makes bottlenecks visible (math, memory, or serving), so optimization becomes targeted, not guesswork.

Governance (artifacts + CI gate)

Where to wire it in:

- Train end: write a reproducibility receipt and lineage graph next to the latest checkpoint.
- Export step: after producing deployable artifacts, generate SBOM + checksums/signatures and a model card.
- CI gate: verify expected artifacts exist before releasing or serving.
- Serve packaging: include the artifacts directory in the image/tarball for traceability.

Minimal commands:

```bash
# SBOM from current env (SPDX 2.3)
python -m governance sbom --out artifacts/SBOM.spdx.json --name $MODEL_NAME

# Checksums (and optional Ed25519 signatures if you have a 32-byte key)
python -m governance sign --files artifacts/model.onnx artifacts/tokenizer.json \
  --out artifacts/CHECKSUMS.sha256 --key path/to/ed25519.key

# Reproducibility receipt (system, env, packages, git)
python -m governance receipt --artifacts artifacts/model.onnx \
  --out artifacts/RECEIPT.json --metadata train/metadata.json

# Lineage graph from training metadata (writes .dot and optional .png if graphviz available)
python -m governance lineage --metadata train/metadata.json --out artifacts/LINEAGE.dot

# Model card (Markdown + JSON sidecar)
python -m governance card artifacts/model.onnx --metadata train/metadata.json \
  --sbom artifacts/SBOM.spdx.json --out artifacts/MODEL_CARD.md

# CI verify (fails if any governance artifacts are missing)
python -m governance verify artifacts/model.onnx
```

Typical integration points:

- train/trainer.py: on checkpoint save, emit RECEIPT.json and LINEAGE.dot
- export/exporter.py: after write, emit SBOM.spdx.json, CHECKSUMS.sha256, MODEL_CARD.md
- .github/workflows/release.yml: run `python -m governance verify` before publishing assets
- serve/Dockerfile or pack step: copy artifacts/ into image and surface MODEL_CARD.md

flowchart LR
  subgraph Data Plane
    C[corpus] --> D[data]
  end

  subgraph Model Plane
    A[specs] --> F[model]
    B[kernel] --> E[attn]
    E --> G[blocks] --> F
    F --> H[train] --> I[eval]
    F --> J[serve]
  end

  subgraph Ops & Safety
    K[safety] --> J
    K --> H
    L[compress] --> F
    L --> M[export]
    N[autotune] --> H
    O[interpret] --> I
    P[governance] --> M
    Q[pack] --> H
    Q --> J
    R[examples] --> Q
  end

  A --> K
  A --> N
  A --> P

Examples

1) Minimal end-to-end sample (train a few steps, then generate)
python example.py

2) Build a tiny toy shard, then run the minimal sample
python -m corpus.cli build \
  --input ./raw_corpus \
  --outdir ./corpus/shards \
  --tokenizer gpt2 \
  --shard-size-tokens 1048576 \
  --redact-pii --dedup
python example.py

3) Blocks-only model assembly reference
See blocks/examples/example_lm.py for constructing a stack from `LlamaBlock`/`GPTBlock`.

For the roadmap of larger, reproducible examples, see examples/README.md.