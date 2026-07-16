# Hunyuan Avatar FSDP Adapter

This adapter is the `transformer_10` integration point for HunyuanVideo-Avatar
on two 24 GB GPUs. It keeps the upstream transformer CPU-resident until FSDP
or FSDP2 owns placement, avoiding the upstream launcher behavior that constructs
a full CUDA replica on each rank.

## Environment

Use `py311build` for this lane. Verified core stack:

| Package | Version / Status |
| --- | --- |
| Python | 3.11.14 |
| Torch | 2.13.0+cu130 |
| Torch CUDA | available, CUDA 13.0 |
| diffusers | present |
| transformers | present |
| accelerate | present |
| safetensors | present |
| einops / imageio / loguru | present |

Run Hunyuan commands with `PYTHONNOUSERSITE=1 PYTHONPATH=.` from
`/data/transformer_10`. The upstream checkout is expected at
`HUNYUAN_AVATAR_ROOT=/data/clone/hunyuanvideo-avatar`; model assets are under
`/arxiv/models/HunyuanVideo-Avatar` and `/data/models/ckpts/hunyuan-video-t2v-720p`.

## Current Status

| Path | Status | Evidence |
| --- | --- | --- |
| BF16 FSDP1 placement | works after dtype normalization | `logs/hunyuan_avatar_fsdp_bf16_probe2.log`: both ranks report `cpu_bytes=25982986368`, `gpu_allocated=13030046208`. |
| BF16 FSDP1 shard materialization | complete | `checkpoints/hunyuan_avatar_bf16_fsdp2/` has rank 00/01 shards and manifest from `/arxiv/models/HunyuanVideo-Avatar/.../mp_rank_00_model_states.pt`. |
| Native FP8 FSDP2 placement | works | `logs/hunyuan_avatar_fp8_fsdp2_probe.log`: both ranks report `cpu_bytes=13863706880`, `gpu_allocated=7022257664`, `gpu_reserved=8074035200`. |
| Native FP8 FSDP2 shard materialization | complete | `checkpoints/hunyuan_avatar_fp8_fsdp2/` has rank 00/01 shards and manifest from `/data/models/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt`. |
| Native FP8 FSDP2 generation | reaches sampler, then fails | `logs/hunyuan_avatar_fp8_fsdp2_generate.log`: fails in LLaVA text/image prompt encoding with `Image features and image tokens do not match: tokens: 1, features 576`. |

Earlier BF16 FSDP1 probing failed before dtype normalization because FSDP1 cannot
flatten mixed BF16/FP32 parameters in the same wrapped unit. That failure is kept
in `logs/hunyuan_avatar_fsdp_bf16_probe.log`; use `normalize_floating_point_dtype(...)`
before FSDP1 wrapping for the BF16 path.

## Artifacts

| Artifact | Size / Shape |
| --- | --- |
| `checkpoints/hunyuan_avatar_bf16_fsdp2/avatar_transformer.rank00.pt` | 12,992,501,079 bytes |
| `checkpoints/hunyuan_avatar_bf16_fsdp2/avatar_transformer.rank01.pt` | 12,992,501,911 bytes |
| `checkpoints/hunyuan_avatar_fp8_fsdp2/avatar_transformer.rank00.pt` | 6,932,474,121 bytes |
| `checkpoints/hunyuan_avatar_fp8_fsdp2/avatar_transformer.rank01.pt` | 6,932,474,121 bytes |

Both manifests declare `world_size=2`.

## Commands

BF16 placement probe:

```bash
conda run -n py311build env PYTHONNOUSERSITE=1 PYTHONPATH=. HUNYUAN_AVATAR_ROOT=/data/clone/hunyuanvideo-avatar torchrun --standalone --nproc_per_node=2 scripts/probe_hunyuan_avatar_fsdp.py --ckpt /arxiv/models/HunyuanVideo-Avatar/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt
```

BF16 shard materialization:

```bash
conda run -n py311build env PYTHONNOUSERSITE=1 PYTHONPATH=. HUNYUAN_AVATAR_ROOT=/data/clone/hunyuanvideo-avatar torchrun --standalone --nproc_per_node=2 scripts/materialize_hunyuan_avatar_fsdp_shards.py --output-dir /data/transformer_10/checkpoints/hunyuan_avatar_bf16_fsdp2 --ckpt /arxiv/models/HunyuanVideo-Avatar/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt
```

FP8 FSDP2 placement probe:

```bash
conda run -n py311build env PYTHONNOUSERSITE=1 PYTHONPATH=. HUNYUAN_AVATAR_ROOT=/data/clone/hunyuanvideo-avatar torchrun --standalone --nproc_per_node=2 scripts/probe_hunyuan_avatar_fp8_fsdp2.py --use-fp8 --ckpt /data/models/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt
```

FP8 FSDP2 generation attempt:

```bash
conda run -n py311build env PYTHONNOUSERSITE=1 PYTHONPATH=. HUNYUAN_AVATAR_ROOT=/data/clone/hunyuanvideo-avatar MODEL_BASE=/data/models torchrun --standalone --nproc_per_node=2 scripts/sample_hunyuan_avatar_fp8_fsdp2.py --shard-dir /data/transformer_10/checkpoints/hunyuan_avatar_fp8_fsdp2 --use-fp8 --ckpt /data/models/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt --cpu-offload --input input/peyton_avatar_test.csv --save-path results/peyton_seated_fp8_fsdp2 --infer-steps 20 --flow-shift-eval-video 5.0 --seed 1024
```

## Next Fix

The next blocker is not FSDP placement. It is prompt/image formatting for the
LLaVA encoder in the upstream Avatar pipeline. The failing call reports one
image token but 576 image features, so the next pass should inspect the
`prompt_template_video`, `TextEncoder.text2tokens(...)`, and the CSV/image path
format to ensure the prompt includes the expected image token expansion for the
provided anchor image.
