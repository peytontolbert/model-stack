# Model Stack Conda Environment Matrix

Personal runtime notes for integrating locally downloaded models into model-stack.
Use this as the first stop before loading a catalog entry, because several model
families need different PyTorch, CUDA, Diffusers, Transformers, or project-local
dependency versions.

Last verified: 2026-07-15.

## Rule Of Thumb

Use `ai` first for `diffusers_cuda_bridge` work. It is the currently verified
environment for FLUX.2-dev and the ChronoEdit LoRA adapter path, with CUDA
visible and recent enough Diffusers/Transformers support.

Use `trellis` only when a model or tool was built around the Trellis stack or
CUDA 12.4-era dependencies. Do not try to merge the TRELLIS and Hunyuan3D
runtimes into one Python process; model-stack should bridge them through
env-isolated workers and a common mesh artifact API.

Use `py311build` for build/runtime experiments that need the CUDA 13 PyTorch
stack, but do not assume Diffusers parity with `ai` until the target model is
smoked there.

Avoid base Python for model-stack bridge validation; its Torch import surface has
not been reliable for this work.

## Environment Versions

| Env | Python | Torch / CUDA | Diffusers | Transformers | Accelerate | PEFT | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ai` | 3.11.11 | 2.10.0+cu128 / CUDA 12.8 | 0.39.0.dev0 | 4.57.6 | 1.5.2 | 0.17.1 | Primary Diffusers bridge env. Verified CUDA available. |
| `py311build` | 3.11.14 | 2.13.0+cu130 / CUDA 13.0 | 0.33.0 | 4.50.0 | 1.14.0 | 0.19.1 | Build/cuda13 candidate. Re-smoke each Diffusers model before relying on it. |
| `trellis` | 3.10.16 | 2.6.0+cu124 / CUDA 12.4 | 0.36.0 | 4.57.6 | 1.10.1 | 0.19.1 | Trellis-era compatibility lane with `spconv-cu121`, `kaolin`, `nvdiffrast`, Open3D, `numpy==1.26.4`; `/data/clone/third_party/TRELLIS.2` imports and provides `Trellis2ImageTo3DPipeline`; load-only and 512/1-step GLB export pass. |
| `nemo_speech` | 3.12.13 | 2.13.0+cu132 / CUDA 13.2 | missing | 4.57.6 | 1.14.0 | 0.19.1 | Dedicated NeMo ASR env with `nemo_toolkit` 2.7.3; local `.nemo` archives status-smoke cleanly. |

`nemo_toolkit` was not installed in these three environments when checked. The
`nemo_asr_bridge` lane uses the dedicated `nemo_speech` environment. The env exists at `/home/peyton/miniconda3/envs/nemo_speech` with Python 3.12.13, Torch 2.13.0+cu132, and `nemo_toolkit` 2.7.3. Keep `PYTHONNOUSERSITE=1` when validating it so user-site packages cannot mask missing env packages.

## Verification Reports

The lightweight first-wave catalog verifier is `scripts/verify_model_stack_models.py`. It writes the current human-readable report to [`docs/model-stack-model-verification.md`](model-stack-model-verification.md) and machine-readable env outputs under `reports/`.
For package-level version blockers, use [`docs/model-dependency-diagnostics.md`](model-dependency-diagnostics.md) and `scripts/model_dependency_diagnostics.py`.

Current `ai` verification summary after resolving `/arxiv/models`: 10 Diffusers/video snapshots pass lightweight component validation, 2 adapter snapshots are present, 37 Transformers snapshots are local candidates needing full load smokes, 22 PEFT adapters need explicit base models, 6 NeMo ASR entries use the dedicated `nemo_speech` env, 27 local entries need custom bridge/env work, and 1 Diffusers snapshot is incomplete.

Run all lightweight env checks with:

```bash
conda run -n ai env PYTHONPATH=. python scripts/verify_model_stack_models.py --env-name ai --markdown-out docs/model-stack-model-verification.md --json-out reports/model-stack-model-verification.ai.json
conda run -n py311build env PYTHONPATH=. python scripts/verify_model_stack_models.py --env-name py311build --json-out reports/model-stack-model-verification.py311build.json
conda run -n trellis env PYTHONPATH=. python scripts/verify_model_stack_models.py --env-name trellis --json-out reports/model-stack-model-verification.trellis.json
```

## Transformers Causal LM Bridge

Use `ai` first for local Transformers causal LM snapshots. The bridge smoke harness records config/tokenizer/load/generate status plus RSS and GPU memory snapshots:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ai env PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/smoke_transformers_causal_lm_bridge.py /arxiv/models/MobileLLM-125M --model-id MobileLLM-125M --device cuda:0 --dtype bfloat16 --generate --max-new-tokens 4 --no-use-cache --json-out reports/causal-lm-smokes/MobileLLM-125M.cuda0.ai.json
```

Classic `MobileLLM-*` snapshots use repo-local `configuration_mobilellm.py` / `modeling_mobilellm.py` and need `trust_remote_code=True`. In `ai` with Transformers 4.57.6, `AutoTokenizer` / `LlamaTokenizerFast` return `False` for these snapshots, while slow `LlamaTokenizer` works. Their generation cache path also calls the removed `DynamicCache.get_max_length` API, so use `use_cache=False` until the local modeling code is patched or an older Transformers env is chosen.

`MobileLLM-R1.5-*` snapshots use built-in `llama4_text`; `MobileLLM-R1.5-140M`, `MobileLLM-R1.5-360M`, and `MobileLLM-R1.5-950M` validate cleanly with `AutoTokenizer` and `Llama4ForCausalLM` in `ai`. `MobileLLM-Pro*` snapshots use remote-code `MobileLLMP1ForCausalLM` and validate in `ai`, but the int4 accelerator snapshot currently falls back to a normal bf16 load and ignores fake-quant scale tensors.

Verified CUDA smoke results on `CUDA_VISIBLE_DEVICES=0`:

| Model | Env | Loader | Params | Load | 4-token generate | GPU memory after generate | Caveat | Report |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `MobileLLM-125M` | `ai` | `MobileLLMForCausalLM` + slow `LlamaTokenizer` | 124,635,456 | 3.99s | 2.53s | 596 MB | `--no-use-cache`; `AutoTokenizer` returns `False` | `reports/causal-lm-smokes/MobileLLM-125M.cuda0.ai.json` |
| `MobileLLM-125M-layer-share` | `ai` | `MobileLLMForCausalLM` + slow `LlamaTokenizer` | 124,635,456 | 8.12s | 0.68s | 609 MB | `--no-use-cache`; `AutoTokenizer` returns `False` | `reports/causal-lm-smokes/MobileLLM-125M-layer-share.cuda0.ai.json` |
| `MobileLLM-1.5B` | `ai` | `LlamaForCausalLM` + slow `LlamaTokenizer` | 1,562,388,800 | 79.42s | 1.05s | 3471 MB | `--no-use-cache`; `AutoTokenizer` returns `False`; single `.bin` load is slow | `reports/causal-lm-smokes/MobileLLM-1.5B.cuda0.ai.json` |
| `MobileLLM-1B` | `ai` | `MobileLLMForCausalLM` + slow `LlamaTokenizer` | 1,005,461,760 | 58.47s | 0.77s | 2527 MB | `--no-use-cache`; single `.bin` load is slow | `reports/causal-lm-smokes/MobileLLM-1B.cuda0.ai.json` |
| `MobileLLM-350M` | `ai` | `MobileLLMForCausalLM` + slow `LlamaTokenizer` | 345,355,200 | 19.28s | 0.54s | 1234 MB | `--no-use-cache`; `AutoTokenizer` returns `False` | `reports/causal-lm-smokes/MobileLLM-350M.cuda0.ai.json` |
| `MobileLLM-350M-layer-share` | `ai` | `MobileLLMForCausalLM` + slow `LlamaTokenizer` | 345,355,200 | 18.01s | 0.60s | 1043 MB | `--no-use-cache`; `AutoTokenizer` returns `False` | `reports/causal-lm-smokes/MobileLLM-350M-layer-share.cuda0.ai.json` |
| `MobileLLM-600M` | `ai` | `MobileLLMForCausalLM` + slow `LlamaTokenizer` | 603,188,352 | 36.93s | 0.76s | 1698 MB | `--no-use-cache`; single `.bin` load is slow | `reports/causal-lm-smokes/MobileLLM-600M.cuda0.ai.json` |
| `MobileLLM-R1.5-140M` | `ai` | `Llama4ForCausalLM` + `AutoTokenizer` | 140,248,512 | 9.00s | 0.60s | 614 MB | no classic MobileLLM tokenizer fallback needed | `reports/causal-lm-smokes/MobileLLM-R1.5-140M.cuda0.ai.json` |
| `MobileLLM-Pro` | `ai` | `MobileLLMP1ForCausalLM` + `AutoTokenizer` | 1,084,453,120 | 83.61s | 0.64s | 2533 MB | remote code OK; no compatibility patches needed | `reports/causal-lm-smokes/MobileLLM-Pro.cuda0.ai.json` |
| `MobileLLM-Pro-base` | `ai` | `MobileLLMP1ForCausalLM` + `AutoTokenizer` | 1,084,453,120 | 55.15s | 0.61s | 2533 MB | remote code OK; no compatibility patches needed | `reports/causal-lm-smokes/MobileLLM-Pro-base.cuda0.ai.json` |
| `MobileLLM-Pro-base-int4-accelerator` | `ai` | `MobileLLMP1ForCausalLM` + `AutoTokenizer` | 1,084,453,120 | 52.72s | 0.57s | 2533 MB | bf16 fallback only; fake-quant scale tensors ignored by Transformers loader | `reports/causal-lm-smokes/MobileLLM-Pro-base-int4-accelerator.cuda0.ai.json` |
| `MobileLLM-R1.5-950M` | `ai` | `Llama4ForCausalLM` + `AutoTokenizer` | 949,685,760 | 48.15s | 0.63s | 2183 MB | no compatibility patches needed | `reports/causal-lm-smokes/MobileLLM-R1.5-950M.cuda0.ai.json` |
| `MobileLLM-R1.5-360M` | `ai` | `Llama4ForCausalLM` + `AutoTokenizer` | 359,431,168 | 23.50s | 0.53s | 1067 MB | no compatibility patches needed | `reports/causal-lm-smokes/MobileLLM-R1.5-360M.cuda0.ai.json` |

## NeMo ASR Env Target

The dedicated env has been created. Recreate it with this recipe if needed instead of mutating `ai`, `py311build`, `trellis`, or `base`:

```bash
conda create -n nemo_speech python=3.12 -y
conda activate nemo_speech
pip install 'nemo-toolkit[asr,tts,cu13]' --extra-index-url https://download.pytorch.org/whl/cu132
```

If CUDA 13 wheels are not compatible on the machine, use the CUDA 12 wheel path:

```bash
pip install 'nemo-toolkit[asr,tts,cu12]' --extra-index-url https://download.pytorch.org/whl/cu126
```

Status-only local archive smoke, safe to run before restoring weights:

```bash
conda run -n nemo_speech env PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/smoke_nemo_asr_bridge.py parakeet-rnnt-0.6b
```

Optional restore smoke once the env imports cleanly:

```bash
conda run -n nemo_speech env PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/smoke_nemo_asr_bridge.py parakeet-rnnt-0.6b --map-location cpu --transcribe-audio /tmp/model_stack_hello_probe_16k.wav
```

Small archive restore/transcribe probe, useful before testing the larger 0.6B/1.1B files:

```bash
conda run -n nemo_speech env PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/smoke_nemo_asr_bridge.py --archive-path /arxiv/models/parakeet-tdt_ctc-110m/parakeet-tdt_ctc-110m.nemo --map-location cpu --transcribe-audio /tmp/model_stack_hello_probe_16k.wav
```

Resolved caveat: NeMo 2.7.3 pins `transformers~=4.57.0`, while local Parakeet configs/model cards advertise a Transformers 5.x line. The `.nemo` archive path was validated with `parakeet-tdt_ctc-110m` and `parakeet-rnnt-0.6b`: `ASRModel.restore_from(...)` restored `EncDecHybridRNNTCTCBPEModel` / `EncDecRNNTBPEModel`, and `model.transcribe(...)` completed under `nemo_speech`. Treat the Transformers 5.x field as external metadata noise for `.nemo` archive loading unless a specific archive emits a Transformers API error.

### NeMo ASR Warm Inference Benchmarks

Benchmark script: `scripts/benchmark_nemo_asr_inference.py`. It restores once, moves the model to CUDA, runs warmup, then measures repeated `model.transcribe(...)` calls. These numbers use `/tmp/model_stack_hello_probe_16k.wav`, a 1-second 16 kHz probe, on `CUDA_VISIBLE_DEVICES=0`. Cold restore time is cache-sensitive and should not be treated as request latency.

| Model | Restore | Move to CUDA | Warm transcribe median | Memory note | Report |
| --- | ---: | ---: | ---: | --- | --- |
| `parakeet-tdt_ctc-110m` | 5.49s cached / 29.87s earlier cold-ish run | 0.38s | 58.6ms | process GPU memory 906 MB after repeats; RSS 2.1 GB | `reports/nemo-asr-benchmarks/parakeet-tdt_ctc-110m.cuda0.pidmem.json` |
| `parakeet-rnnt-0.6b` | 331.07s | 1.49s | 73.8ms | whole-GPU snapshot rose by about 10.7 GB on GPU 0; rerun with PID memory logging for exact ownership | `reports/nemo-asr-benchmarks/parakeet-rnnt-0.6b.cuda0.json` |

Inference architecture rule: do not restore `.nemo` per request. Restore the archive once during service startup, keep the model warm on GPU, and route many transcribe calls through the resident model.

## World Model Validation

World-model layout validation is tracked in [`docs/world-model-validation.md`](world-model-validation.md) and `reports/world-model-validation.ai.json`. Run it with:

```bash
conda run -n ai env PYTHONPATH=. python scripts/validate_world_models.py --json-out reports/world-model-validation.ai.json --markdown-out docs/world-model-validation.md
```

Current buckets:

| Family | Status | Env / Next Step |
| --- | --- | --- |
| `Cosmos-Embed1-224p` | `verified_transformers_remote_code` | `ai`; BF16 text + video embeddings verified with `AutoProcessor` + `AutoModel`, projection shape `[1, 256]`. |
| `Cosmos-Embed1-448p` | `verified_transformers_remote_code` | `ai`; FP32 text + video embeddings verified, projection shape `[1, 768]`; BF16 video path fails on a layer-norm dtype mismatch. |
| `Cosmos-Embed1-448p-anomaly-detection` | `verified_transformers_remote_code` | `py311build`; FP32 text embedding verified. `ai` is incompatible because Transformers 4.57 lacks `transformers.modeling_utils.apply_chunking_to_forward`. |
| `Cosmos3-Nano*` | `verified_cosmos3_upstream_diffusers_adapter_plus_lightx2v_lazy_generation` for base Nano; Policy-DROID still candidate | `ai`; upstream Diffusers exists as `Cosmos3OmniPipeline`, while local snapshots advertise stale `Cosmos3OmniDiffusersPipeline`. Use `runtime.cosmos3_omni_diffusers_pipeline.patch_diffusers_cosmos3()` plus `prepare_cosmos3_upstream_diffusers_snapshot()` for real Diffusers call sites; bounded construction with transformer skipped passes in 6.07s. For generation today, keep using the LightX2V lazy path through `scripts/lightx2v_cosmos3_lazy_infer.py`; base Nano bounded 256px/1-step `t2i` completes in 75.5s. |
| `Cosmos-Predict2.5-14B`, `Cosmos-Transfer2.5-2B` | `candidate_cosmos25_repo_checkpoint` | `cosmos25_py310`; local BF16 `.pt` artifacts and official runtime checkouts validate. `cosmos25_py310` is installed with the official cu128 stack; Predict and Transfer runtime import probes pass. Bridge launch plans now use official examples with local `--checkpoint-path` overrides; full generation is still pending and needs offload/multi-GPU validation. |
| `Hunyuan3D-2mv` | `verified_hy3dgen_bridge` | `ai`; official `hy3dgen` loads `/arxiv/models/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/model.fp16.safetensors` and exports GLB through `three_d_gen_bridge`. LightX2V remains only an asset/import probe for this schema. |
| `Hunyuan3D-Omni` | `needs_hunyuan3d_omni_bridge` | `hunyuan3d_omni_or_custom_bridge`; local Omni assets are present, but it is not the same LightX2V Hunyuan3D 2.x shape-DiT path. |
| `AnyFlow-Wan2.1-T2V-1.3B-Diffusers` | `verified_cached_prompt_embeds_cuda_bridge` | `ai`; no-text `AnyFlowPipeline` load and tiny latent prompt-embeds inference pass on CUDA BF16 with `device_map='cuda'`. Use cached prompt embeddings; keep UMT5 off the hot path. |
| `AnyFlow-FAR-Wan2.1-1.3B-Diffusers` | `verified_cached_prompt_embeds_cuda_bridge` | `ai`; no-text `AnyFlowFARPipeline` load and tiny latent prompt-embeds inference pass on CUDA BF16. Use cached prompt embeddings; keep UMT5 off the hot path. |
| `zai-org--CogVideoX-2b` | `verified_no_generation_pipeline_load` | `ai`; missing T5 text encoder shard was restored. `CogVideoXPipeline` no-generation load passes in BF16 on CUDA; minimal 9-frame/2-step generation smoke passes. |
| `acvlab--ABot-World-0-5B-LF` | `verified_generator_cuda_bridge` | `abot_world` works for generator CUDA BF16 via `runtime.abot_world_bridge`; Torch 2.13.0+cu132 and Diffusers 0.31.0 need narrow API patches, not an env replacement. Generator load: 42.95s / 10.3GB; one action-conditioned forward: 1.07s / 18.1GB with SDPA fallback. T5 `.pth` prompt encoder is now lazy with prompt-cache support; full pipeline construction validates at 17.18s / 10.34GB without loading T5, and cached `set_prompts` keeps `text_encoder_loaded=false`. Optional `flash_attn`, `sageattention`, and `sageattn3` are still missing for fastest attention. |
| `robbyant/lingbot-video-dense-1.3b` | `needs_lingbot_video_runtime` | Snapshot is complete, but current envs lack `LingBotVideoPipeline` and the `lingbot_video` package; local snapshot has `scheduling_flow_unipc.py` only, not pipeline/transformer code. |
| `pe-av-small`, `pe-av-base-16-frame` | `needs_pe_av_transformers_bridge` | Need a dedicated Transformers-main or `perception_models` env; current `ai`, `py311build`, and `trellis` lack `PeAudioVideoModel` / `PeAudioVideoProcessor`. |
| `EGM-4B`, `EGM-4B-SFT` | `candidate_transformers_image_text_to_text` | `ai`; config/processor/meta construction verified as `Qwen3VLForConditionalGeneration` with `AutoModelForImageTextToText`. Full CUDA placement is not verified; use explicit `device_map`/`max_memory`/offload instead of plain `.to(cuda)`. |
| LightX2V/Wan int8 split | `needs_wan_lightx2v_loader` | Partial block replacement checkpoint only; needs base Wan2.2 I2V plus custom block assembly. |
| `GEN3C-Cosmos-7B` | `needs_gen3c_cosmos_predict1_runtime` | `gen3c_cosmos_predict1_or_custom_bridge`; local `Cosmos_GEN3C` config and 27GB PyTorch zip checkpoint metadata validate. Needs the nv-tlabs/Gen3C runtime on top of Cosmos-Predict1; do not use generic Diffusers/Transformers or unsafe `torch.load` in status probes. |
| `HunyuanWorld-Voyager` | `needs_hunyuanworld_custom_bridge` | Local layout exists; top config JSON has trailing comma and the model needs HunyuanWorld-specific loader. |
| `robbyant/lingbot-world-*` | `candidate_lingbot_world_lightx2v_bridge` | `ai`; model-stack routes through `runtime.lingbot_world_bridge` and the local LightX2V checkout. v2 causal-fast uses `model_cls=lingbot_world_fast`, local `transformers/` shards, and shared UMT5; base-act uses the older two-stage `lingbot_world` high/low-noise layout. Bounded generation is still pending. |
| `repository_library/world-planner-adapter` | `adapter_needs_base_model` | PEFT adapter present; identify explicit planner/world base model before load. |


Cosmos Embed1 smoke command pattern:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n ai env PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/smoke_cosmos_embed1.py /arxiv/models/Cosmos-Embed1-224p --device cuda:0 --dtype bfloat16 --video --json-out reports/world-model-smokes/Cosmos-Embed1-224p.text-video.cuda2.json
CUDA_VISIBLE_DEVICES=2 conda run -n ai env PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/smoke_cosmos_embed1.py /arxiv/models/Cosmos-Embed1-448p --device cuda:0 --dtype float32 --video --json-out reports/world-model-smokes/Cosmos-Embed1-448p.text-video-fp32.cuda2.json
CUDA_VISIBLE_DEVICES=2 conda run -n py311build env PYTHONNOUSERSITE=1 PYTHONPATH=. python scripts/smoke_cosmos_embed1.py /arxiv/models/Cosmos-Embed1-448p-anomaly-detection --device cuda:0 --dtype float32 --json-out reports/world-model-smokes/Cosmos-Embed1-448p-anomaly-detection.text-fp32.py311build.cuda2.json
```

AnyFlow cold-load note: the text-prompt path for `AnyFlow-Wan2.1-T2V-1.3B-Diffusers` reached the 5-shard `UMT5EncoderModel` load and timed out at 480 seconds after shard 2/5. Both AnyFlow 1.3B snapshots are now verified through cached-prompt bridge paths in `ai`: skip `text_encoder`/`tokenizer` and pass `prompt_embeds` at inference. Non-FAR uses `device_map='cuda'`; FAR uses normal CUDA placement plus its return-tuple compatibility patch.


### Official Cosmos 2.5 env target

The cloned Cosmos 2.5 repos expect `uv sync --extra=cu128` or `uv sync --extra=cu130`; the shared `cosmos-oss` package pins `transformers==4.51.3` and CUDA extras for Torch 2.7.0/cu128 or Torch 2.9.1/cu130 with `flash-attn`, `natten`, and `transformer-engine`. Dedicated env `cosmos25_py310` is installed with Python 3.10.20, Torch 2.7.0+cu128, Transformers 4.51.3, Diffusers 0.35.2, `flash-attn==2.7.3+cu128.torch27`, `natten==0.21.0+cu128.torch27`, and Transformer Engine 2.2. Keep this separate from `ai`, because `ai` is carrying newer Transformers/Diffusers bridge work. Model-stack launch plans pass local checkpoint files explicitly, so the official registry does not need to download defaults. The discarded Python 3.11 env name `cosmos25` was removed; official Cosmos FlashAttention wheels for these extras do not support CPython 3.11.


### 3D Generation Bridge Split: TRELLIS vs Hunyuan3D

Local TRELLIS is `microsoft/TRELLIS.2-4B` in `/data/huggingface/hub`, not `/arxiv/models`. Its `pipeline.json` names `Trellis2ImageTo3DPipeline` and references safetensors modules for sparse structure, shape SLAT, and texture SLAT. The existing `trellis` env has the geometry stack TRELLIS needs: Python 3.10.16, Torch 2.6.0+cu124, Diffusers 0.36.0, Transformers 4.57.6, `spconv-cu121==2.3.8`, `kaolin==0.17.0`, `nvdiffrast==0.4.0`, Open3D 0.19, and NumPy 1.26.4. Official runtime source is cloned at `/data/clone/third_party/TRELLIS.2` and imports in `trellis`; model-stack routes TRELLIS.2 through `trellis2.pipelines.Trellis2ImageTo3DPipeline` and exports GLB through the installed `o_voxel` extension.

Hunyuan3D is a separate lane. Execution now uses official `hy3dgen` in `ai` with Python 3.11.11, Torch 2.10.0+cu128, Diffusers 0.39 dev, Transformers 4.57.6, Safetensors 0.8.0rc1, and NumPy 2.2.6. The bridge bypasses the incompatible LightX2V `x_embedder.weight` loader for configs targeting `hy3dgen.shapegen.models.Hunyuan3DDiT`; Hunyuan3D-2mv load-only and GLB export are verified.

Bridge decision: do not try to resolve this by installing one shared env. Use `runtime.three_d_gen_bridge` as the model-stack API boundary, launch backend-specific workers pinned to `trellis` or `ai`, and normalize outputs to a mesh artifact bundle (`glb` preferred, with `obj`/`ply`/textures as needed). The executable worker path is `scripts/three_d_gen_worker.py`; validated reports are `reports/world-model-smokes/hunyuan3d.hy3dgen.load_only.ai.json`, `reports/world-model-smokes/hunyuan3d.hy3dgen.smoke_generate.ai.json`, `reports/world-model-smokes/trellis2.official_runtime.load_only.trellis.json`, and `reports/world-model-smokes/trellis2.official_runtime.tiny_generate.trellis.json`. Conflict report: `reports/world-model-smokes/trellis2-vs-hunyuan3d.bridge_conflicts.json`.

## Current Model Compatibility Notes

| Model / Catalog ID | Bridge Lane | Preferred Env | Status | Command / Note |
| --- | --- | --- | --- | --- |
| `black-forest-labs--FLUX.2-dev` | `diffusers_cuda_bridge` | `ai` | Component schemas validate, but generic pipeline-level `device_map=balanced` is flawed: the whole ~60GiB BF16 transformer is assigned to CPU. Use custom Flux2 bridge with component-level transformer submodule placement and cached prompt embeddings; skip text encoder on hot path. | `reports/world-model-smokes/black-forest-labs--FLUX.2-dev.pipeline_flaw.ai.json` |
| `ChronoEdit-14B-Diffusers-Paint-Brush-Lora` | `diffusers_cuda_bridge` adapter asset | `ai` | Known good as a LoRA/adapter snapshot. It resolves to the `nvidia` HF cache alias and has `paintbrush_lora_diffusers.safetensors`. | `conda run -n ai env PYTHONPATH=. python scripts/smoke_diffusers_bridge.py ChronoEdit-14B-Diffusers-Paint-Brush-Lora --adapter-status` |
| `black-forest-labs--FLUX.2-klein-9B` | `diffusers_cuda_bridge` | `ai` | Full placement and 256x256 1-step latent generation verified with `device_map=balanced` and explicit `max_memory`; warmed load 5.46s, generation 5.03s. | `reports/world-model-smokes/black-forest-labs--FLUX.2-klein-9B.flux2_full_256x256_1step.latent.ai.json` |
| `AnyFlow-Wan2.1-T2V-1.3B-Diffusers` | `diffusers_cuda_bridge` / `world_model_bridge` | `ai` verified | No-text pipeline load and tiny latent 1-step prompt-embeds inference pass with `device_map='cuda'`; load 7.83s, generation 0.62s, about 3.12GB allocated. | `conda run -n ai env PYTHONPATH=. python scripts/smoke_diffusers_bridge.py AnyFlow-Wan2.1-T2V-1.3B-Diffusers --model-root /arxiv/models --load-pipeline --device cuda:0 --dtype bfloat16 --device-map cuda --skip-component text_encoder --skip-component tokenizer` |
| `AnyFlow-FAR-Wan2.1-1.3B-Diffusers` | `diffusers_cuda_bridge` / `world_model_bridge` | `ai` verified | No-text pipeline load passes in 6.90s / 3.10GB allocated; tiny latent 1-step prompt-embeds inference passes in 2.55s generation. Bridge skips `text_encoder` and `tokenizer` and applies `compat:anyflow_far_return_tuple_padding`. | `conda run -n ai env PYTHONPATH=. python scripts/smoke_diffusers_bridge.py AnyFlow-FAR-Wan2.1-1.3B-Diffusers --model-root /arxiv/models --load-pipeline --device cuda:0 --dtype bfloat16 --skip-component text_encoder --skip-component tokenizer` |
| `nvidia/ChronoEdit-14B-Diffusers` | `diffusers_cuda_bridge` / `world_model_bridge` | `ai` candidate | Wan I2V Diffusers snapshot is complete after linking the config-compatible LingBot Wan VAE; VAE and CLIP image encoder component smokes pass on CUDA BF16; transformer/full-pipeline cold-load smokes are pending because the current managed-placement path is too slow interactively. | `conda run -n ai env PYTHONPATH=. python scripts/smoke_diffusers_bridge.py nvidia/ChronoEdit-14B-Diffusers --component vae --device cuda:0 --dtype bfloat16` |
| `Wan-AI--Wan2.2-Animate-14B` | Wan Animate custom bridge | `py311build_or_custom_wan_env` | Local checkpoint is complete; use cached T5 prompt embeddings and cached VAE control latents, with int8/offload controls for constrained VRAM. | `reports/world-model-smokes/Wan-AI--Wan2.2-Animate-14B.status.ai.json` |
| `HunyuanVideo-Avatar` | `hunyuan_avatar_bridge` | `py311build` | Custom bridge verifies local checkpoint assets, upstream `hymm_sp` imports, MODEL_BASE path mapping, BF16 shards, and FP8 FSDP2 shards; applies `compat:transformers_utils_flax_weights_name`. Full generation still needs the LLaVA image-token alignment fix. | `reports/world-model-smokes/HunyuanVideo-Avatar.bridge.py311build.json` |
| `HunyuanVideo-I2V` | Hunyuan custom bridge | `py311build_or_hunyuanvideo_env` | Config is Python-style/non-strict JSON; needs Hunyuan runtime/config adapter. | `reports/world-model-smokes/HunyuanVideo-I2V.status.ai.json` |
| `Hunyuan3D-2mv` | `three_d_gen_bridge` / official `hy3dgen` | `ai` verified | Official `hy3dgen` loads local fp16 weights and exports GLB; bridge adapts single-image requests to the mv front-view schema and reports empty meshes gracefully for too-degenerate settings. | `reports/world-model-smokes/hunyuan3d.hy3dgen.load_only.ai.json`, `reports/world-model-smokes/hunyuan3d.hy3dgen.smoke_generate.ai.json` |
| `Hunyuan3D-Omni` | Hunyuan3D Omni custom bridge | `hunyuan3d_omni_or_custom_bridge` | Assets are present but need an Omni-specific runtime/control bridge, not the LightX2V 2.x shape-DiT route. | `reports/world-model-smokes/Hunyuan3D-Omni.status.ai.json` |
| `microsoft/TRELLIS.2-4B` | `three_d_gen_bridge` / official `trellis2` | `trellis` verified | Official TRELLIS.2 runtime loads local 4B weights and exports GLB; worker routes HF cache to `/data/huggingface` and uses installed `o_voxel`. 512/1-step smoke took 151.65s inside the worker. | `reports/world-model-smokes/trellis2.official_runtime.load_only.trellis.json`, `reports/world-model-smokes/trellis2.official_runtime.tiny_generate.trellis.json` |
| `HunyuanVideo-1.5` | HunyuanVideo 1.5 bridge | `hunyuanvideo_or_custom_bridge` | Local snapshot lacks transformer weights; do not attempt generic load until completed. | `reports/world-model-smokes/HunyuanVideo-1.5.status.ai.json` |
| `DAM-3B`, `DAM-3B-Video`, `DAM-3B-Self-Contained` | DAM lazy submodule bridge | `ai` | Component bridge works: vendored runtime imports, vision_tower loads, mm_projector/context_provider load with parent `LlavaLlamaConfig` and exact key matches. Full lazy LLM assembly/placement remains pending. | `reports/world-model-smokes/DAM-3B.component_bridge.ai.json` |
| `PixelDiT-1300M-1024px` | PixelDiT custom bridge | `pixeldit_or_custom_bridge` | Runtime source gap: config and PyTorch zip checkpoint are present, but no local PixelDiT runtime source/module is installed. Safe zip metadata report records 395 entries / 391 tensor storage entries. | `reports/world-model-smokes/PixelDiT-1300M-1024px.runtime_gap.ai.json` |
| `Wan-AI--Wan2.2-S2V-14B` | Wan S2V custom bridge | `wan_s2v_or_custom_wan_env` | Local path is custom Wan layout and incomplete: diffusion index expects shards 00001-00004, but only 00004 is materialized; cache contains incomplete downloads. Bundled Wav2Vec2 submodel passes CUDA forward in `ai`. | `reports/world-model-smokes/Wan-AI--Wan2.2-S2V-14B.layout_and_subcomponents.ai.json` |
| `MOSS-SoundEffect-v2.0` | audio diffusion custom runtime | `moss-soundeffect-v2_or_custom_bridge` | Local snapshot is complete, but current `ai`, `py311build`, and `trellis` do not have `moss_soundeffect_v2`, and installed Diffusers does not expose `MossSoundEffectPipeline`. Needs OpenMOSS/MOSS-TTS `moss_soundeffect_v2` runtime or a model-stack bridge around it. | report `reports/world-model-smokes/MOSS-SoundEffect-v2.0.runtime_gap.ai.json` |


| `RMBG-2.0` | `encoder_classifier_bridge` / image segmentation | `trellis` verified | `ai` lacks `kornia`; `trellis` has `kornia` and `timm`. Remote-code `AutoModelForImageSegmentation` loads as `BiRefNet`; 256x256 CUDA FP32 forward passes in 0.93s after cold load. | report `reports/encoder-classifier-smokes/RMBG-2.0.image_segmentation.cuda0.trellis.json` |
| `nvidia/instruction-data-guard` | `instruction_data_guard_bridge` | `ai` verified | Standalone 4-layer MLP classifier over 4096-d text embeddings. CUDA FP32 forward passes; upstream Aegis embedding model is separate. | report `reports/encoder-classifier-smokes/instruction-data-guard.embedding_mlp.cuda0.ai.json` |


| `MobileLLM-Pro-base-int4-cpu` | `transformers_causal_lm_bridge` | `ai` verified | Direct `device_map="cuda:0"` CUDA BF16 load/generate passes; checkpoint contains BF16 tensors and no quantization keys, so it is not an optimized int4 CPU runtime in this path. | `reports/causal-lm-smokes/MobileLLM-Pro-base-int4-cpu.device_map_cuda0.ai.json` |
| `microsoft--FastContext-1.0-4B-SFT` | `transformers_causal_lm_bridge` | `ai` blocked | Config/tokenizer OK as Qwen3/Qwen2TokenizerFast, but direct device-map checkpoint load was stopped after >4 minutes. Needs long-running/offload placement probe. | `reports/causal-lm-smokes/microsoft--FastContext-1.0-4B-SFT.device_map_cuda1.timeout.ai.json` |


| `EGM-4B` | `world_model_bridge` / Qwen3-VL | `ai` blocked | Config and processor load; correct loader is `AutoModelForImageTextToText`, but direct device-map checkpoint load timed out after >4 minutes. Active index uses 2 shards while extra 4-shard files exist. | `reports/world-model-smokes/EGM-4B.load_device_map_cuda1.timeout.ai.json` |
| `Audio2Face-3D-v2.3-Mark`, `Audio2Face-3D-v3.0` | `audio2face_onnx_bridge` | `ai` candidate | ONNX Runtime can load `network.onnx` for both snapshots; full SDK-compatible preprocessing/postprocessing bridge still needed. | `reports/world-model-smokes/Audio2Face-3D-*.onnxruntime.ai.json` |


| `sam-audio-small`, `sam-audio-base`, `sam-audio-large`, `sam-audio-small-tv`, `sam-audio-large-tv` | `sam_audio_bridge` | `sam_audio_or_custom_bridge` | Local custom `.pt` checkpoints and configs are present, but current envs lack `sam_audio.SAMAudio` / `SAMAudioProcessor`; Transformers also lacks SAMAudio and PE-AV classes. | `reports/world-model-smokes/sam-audio-*.runtime_gap.ai.json` |

## Bridge-Specific Defaults

| Bridge | Start Env | Why |
| --- | --- | --- |
| `diffusers_cuda_bridge` | `ai` | Verified with CUDA, Diffusers 0.39 dev, FLUX.2-dev, and ChronoEdit LoRA adapter status. |
| `video_diffusion_bridge` | `ai`, then model-specific env if import errors appear | Start with latest working Diffusers/Transformers stack, then pin per model. |
| `transformers_causal_lm_bridge` | `ai` or `py311build` | Use `ai` for runtime parity first; try `py311build` when CUDA 13 behavior or build integration matters. |
| `peft_adapter_bridge` | Match the base model env | Adapter compatibility follows the base model and Transformers/PEFT versions more than the adapter files. |
| `encoder_classifier_bridge` | `ai` | Transformers 4.57.6 is available and CUDA Torch works. Repository-library BERT encoders/classifiers pass CUDA FP32 tokenize/load/forward smoke. Four classifier snapshots need `transformers_classifier_head_num_labels_from_checkpoint` because checkpoint head shapes disagree with stale `config.num_labels`. |
| `nemo_asr_bridge` | `nemo_speech` | Local `.nemo` archives are present under `/arxiv/models`; existing envs are missing `nemo_toolkit`, so use the dedicated Python 3.12 + Torch 2.13.0 cu132 NeMo env. |
| `hunyuan_avatar_bridge` / `hunyuan_avatar_fsdp` | `py311build` | HunyuanVideo-Avatar bridge is wired into model-stack. Python 3.11.14 + Torch 2.13.0+cu130; BF16 and FP8 rank-local shards exist under `checkpoints/`; runtime imports pass after `compat:transformers_utils_flax_weights_name`; current generation blocker remains LLaVA image-token alignment. |
| `world_model_bridge` | `ai` for Cosmos Embed1 224p/448p, Hunyuan3D 2.x shape, Cosmos3 LightX2V, and Diffusers video candidates; `py311build` for 448p anomaly; `cosmos25_py310` for Cosmos 2.5 | Use `runtime.world_model_bridge.world_model_status(...)` first. Runnable/candidate today: Cosmos Embed1, Hunyuan3D 2.x shape assets, Cosmos3 LightX2V, Cosmos 2.5 repo checkpoints, plus video candidates with component-level validation such as ChronoEdit. GEN3C routes to a Cosmos-Predict1 runtime-needed status; Hunyuan3D-Omni and HunyuanWorld return explicit bridge-needed blockers, while LingBot World now routes to a LightX2V bridge candidate instead of a generic load failure. |

## Smoke Commands

Status-only Diffusers pipeline check:

```bash
conda run -n ai env PYTHONPATH=. python scripts/smoke_diffusers_bridge.py black-forest-labs--FLUX.2-dev
```

Component check, useful before a full load:

```bash
conda run -n ai env PYTHONPATH=. python scripts/smoke_diffusers_bridge.py black-forest-labs--FLUX.2-dev --component vae --device cuda:1 --dtype bfloat16
```

Adapter-only check:

```bash
conda run -n ai env PYTHONPATH=. python scripts/smoke_diffusers_bridge.py ChronoEdit-14B-Diffusers-Paint-Brush-Lora --adapter-status
```

Large pipeline construction without generation:

```bash
conda run -n ai env PYTHONPATH=. python scripts/smoke_diffusers_bridge.py black-forest-labs--FLUX.2-dev --load-pipeline --device-map balanced --max-memory 0=20GiB --max-memory 1=20GiB --max-memory 2=16GiB --max-memory cpu=110GiB
```

## Update Checklist For New Models

1. Record the catalog ID, bridge lane, and local cache path.
2. Run status-only smoke in `ai` first unless the model is known to require a different stack.
3. If status fails due layout, document whether it is adapter-only, custom Diffusers, Transformers, NeMo, or manual.
4. If imports fail, retry in `trellis` or `py311build` only after writing down the failure.
5. Add the verified env, exact smoke command, and result to the compatibility table above.
