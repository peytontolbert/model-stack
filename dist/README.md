# Distributed Package

`dist/` contains the repository-owned distributed training and checkpointing
helpers. It is intentionally a thin layer over PyTorch distributed primitives:
the package normalizes launch environment, wraps models with common strategies,
builds distributed dataloaders, and saves/loads rank-local safetensor shards.

This package is not the same as Python package distribution. It is the runtime
distributed execution layer.

## File Map

| File | Responsibility |
| --- | --- |
| `launch.py` | SLURM-to-torchrun environment mapping and process-group initialization. |
| `utils.py` | rank/world/local-rank helpers, seeding, device setup, barriers, broadcast helpers. |
| `engine.py` | `DistributedEngine` facade around initialization, model wrapping, dataloaders, precision, tensor/pipeline parallel hooks, and teardown. |
| `dataloader.py` | `DistributedSampler`-backed dataloader construction. |
| `checkpoint.py` | rank-sharded safetensor save/load helpers. |
| `cli.py` | small CLI for launch command generation, smoke-test init, and environment inspection. |
| `parallel/tensor_parallel.py` | tensor-parallel model rewrite hooks. |
| `parallel/pipeline.py` | layer partitioning into pipeline stages. |
| `strategy/ddp.py` | DDP wrapper helper. |
| `strategy/fsdp.py` | FSDP wrapper and transformer-block auto-wrap policy. |
| `strategy/deepspeed.py` | DeepSpeed wrapper helper. |

## Launch Model

`initialize_distributed()` performs three steps:

1. Map SLURM variables to torchrun-style `RANK`, `WORLD_SIZE`, and
   `LOCAL_RANK` when needed.
2. Set the CUDA device from local rank before process-group initialization.
3. Initialize `torch.distributed` and optionally seed all ranks.

Use:

```bash
python -m dist.cli print-cmd --nnodes 1 --nproc-per-node 8
```

to print a baseline `torchrun` command. Use:

```bash
python -m dist.cli env
```

to inspect the rank environment visible to the current process.

## DistributedEngine

`DistributedEngine` is a convenience facade around `specs.dist.DistConfig`.
It owns:

- process-group initialization
- DDP/FSDP/DeepSpeed model wrapping
- distributed dataloader construction
- tensor-parallel and pipeline partition hooks
- bf16/fp16 autocast context
- barrier and teardown helpers

The engine should stay policy-light. Strategy-specific logic belongs under
`dist/strategy/`, and topology-specific partitioning belongs under
`dist/parallel/`.

## Checkpoint Semantics

`save_sharded(model, outdir)` writes one safetensor shard per rank:

```text
model-rank00000.safetensors
model-rank00001.safetensors
...
model.index.json
```

`load_any(model, indir)` tries, in order:

1. `model.safetensors`
2. the local rank shard
3. the first indexed shard as a single-rank fallback

The fallback is useful for inspection and smoke tests, but it is not a full
merge of all rank shards.

## Operational Rules

- Device selection must happen before CUDA tensors are materialized on a rank.
- Rank helpers must work before and after process-group initialization.
- Barriers and broadcasts should degrade safely in single-process mode.
- Checkpoint paths should be rank-stable and deterministic.
- Strategy wrappers should be optional-import friendly.
- New launch assumptions should be reflected in this README and in
  `specs.dist.DistConfig`.

