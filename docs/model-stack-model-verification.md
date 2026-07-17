# Model Stack Model Verification

Lightweight verification of first-wave catalog entries. This checks env packages, local path/cache resolution, and bridge-compatible snapshot layout. It does not load every full model by default.

Last verified: 2026-07-14.
Verifier env: `ai`.

## Environment

| Package | Version / Status |
| --- | --- |
| `accelerate` | `1.5.2` |
| `diffusers` | `0.39.0.dev0` |
| `huggingface_hub` | `0.36.2` |
| `nemo_toolkit` | `missing` |
| `peft` | `0.17.1` |
| `safetensors` | `0.8.0rc1` |
| `torch` | `2.10.0` |
| `torch_cuda` | `True` |
| `torch_cuda_version` | `12.8` |
| `torch_import` | `2.10.0+cu128` |
| `transformers` | `4.57.6` |

## Summary

| Status | Count |
| --- | ---: |
| `candidate_transformers_snapshot` | 37 |
| `needs_custom_bridge_or_env` | 27 |
| `adapter_needs_base_model` | 22 |
| `works_snapshot_status` | 10 |
| `needs_nemo_speech_env` | 6 |
| `works_adapter_status` | 2 |
| `verified_sapiens2_pose_load_bridge` | 1 |
| `incomplete_diffusers_snapshot` | 1 |

| Lane | Count |
| --- | ---: |
| `video_diffusion_bridge` | 34 |
| `encoder_classifier_bridge` | 22 |
| `peft_adapter_bridge` | 22 |
| `transformers_causal_lm_bridge` | 15 |
| `diffusers_cuda_bridge` | 6 |
| `nemo_asr_bridge` | 6 |
| `sapiens2_pose_bridge` | 1 |

## diffusers_cuda_bridge

| Model | Status | Preferred Env | Local | Detail |
| --- | --- | --- | --- | --- |
| `ChronoEdit-14B-Diffusers-Paint-Brush-Lora` | `works_adapter_status` | `ai` | yes | weights=paintbrush_lora_diffusers.safetensors |
| `MOSS-SoundEffect-v2.0` | `needs_moss_soundeffect_v2_runtime` | `moss-soundeffect-v2_or_custom_bridge` | yes | snapshot complete, but `moss_soundeffect_v2.MossSoundEffectPipeline` runtime is absent from `ai`, `py311build`, and `trellis`; installed Diffusers has no `MossSoundEffectPipeline` |
| `Wan-AI--Wan2.2-S2V-14B` | `incomplete_wan_s2v_custom_layout` | `wan_s2v_or_custom_wan_env` | yes | custom Wan S2V layout; main diffusion checkpoint missing shards 00001-00003 of 00004; bundled Wav2Vec2 submodel passes CUDA forward in `ai` |
| `black-forest-labs--FLUX.2-dev` | `needs_flux2_component_level_transformer_placement` | `ai` | yes | `Flux2Pipeline`; component schemas validate, but pipeline-level `device_map=balanced` places the whole ~60GiB BF16 transformer on CPU; hot path needs cached prompt embeds plus component-level transformer submodule map |
| `black-forest-labs--FLUX.2-klein-9B` | `verified_flux2_klein_latent_generation` | `ai` | yes | `Flux2KleinPipeline`; full placement and 256x256 1-step latent generation pass with `device_map=balanced`; warmed load 5.46s, generation 5.03s |
| `nvidia/ChronoEdit-14B-Diffusers` | `works_component_smoke` | `ai` | yes | class=WanImageToVideoPipeline; components=image_encoder,image_processor,scheduler,text_encoder,tokenizer,transformer,vae; VAE and image_encoder load on CUDA BF16; transformer managed cold-load reached 1/14 shards in 163.73s and remains pending |

## encoder_classifier_bridge

| Model | Status | Preferred Env | Local | Detail |
| --- | --- | --- | --- | --- |
| `RMBG-2.0` | `works_cuda_forward_smoke` | `trellis` | yes | `AutoModelForImageSegmentation` remote-code load/forward OK; `ai` blocked by missing `kornia`; report `reports/encoder-classifier-smokes/RMBG-2.0.image_segmentation.cuda0.trellis.json` |
| `nvidia/instruction-data-guard` | `works_embedding_mlp_cuda_smoke` | `ai` | yes | standalone 4-layer MLP over 4096-d Aegis embeddings; CUDA FP32 load/forward OK; report `reports/encoder-classifier-smokes/instruction-data-guard.embedding_mlp.cuda0.ai.json` |
| `repository_library/abstract-repo-planning` | `works_cuda_forward_smoke_with_config_patch` | `ai` | yes | CUDA FP32 tokenize/load/forward OK; patched config.num_labels from classifier.weight |
| `repository_library/author-embedding` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/bug-localization` | `works_cuda_forward_smoke_with_config_patch` | `ai` | yes | CUDA FP32 tokenize/load/forward OK; patched config.num_labels from classifier.weight |
| `repository_library/candidate-row-reranker` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/citation-prediction` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/cluster-router` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/cross-encoder-reranker` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/cross-modal-retrieval` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/file-embedding` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/graph-neighborhood-router` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/heap-scheduler` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/jepa-repo-state` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/metadata-category-classifier` | `works_cuda_forward_smoke_with_config_patch` | `ai` | yes | CUDA FP32 tokenize/load/forward OK; patched config.num_labels from classifier.weight |
| `repository_library/paper-link-prediction` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/repo-embedding` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/repo-similarity` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/repo-state-grounding` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/self-play-reward-model` | `works_cuda_forward_smoke_with_config_patch` | `ai` | yes | CUDA FP32 tokenize/load/forward OK; patched config.num_labels from classifier.weight |
| `repository_library/span-infill-gate` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |
| `repository_library/verifier-accept-policy` | `works_cuda_forward_smoke` | `ai` | yes | CUDA FP32 tokenize/load/forward OK |

### Encoder/Classifier CUDA Smoke Reports

| Model | Loader | Params | Forward | GPU Memory After Forward | Report |
| --- | --- | ---: | ---: | ---: | --- |
| `repository_library/abstract-repo-planning` | `BertForSequenceClassification` | 110,705,920 | 0.230s | 791 MB | `reports/encoder-classifier-smokes/abstract-repo-planning.cuda0.ai.json` |
| `repository_library/author-embedding` | `BertModel` | 22,713,216 | 0.263s | 423 MB | `reports/encoder-classifier-smokes/author-embedding.cuda0.ai.json` |
| `repository_library/bug-localization` | `BertForSequenceClassification` | 109,920,002 | 0.261s | 791 MB | `reports/encoder-classifier-smokes/bug-localization.cuda0.ai.json` |
| `repository_library/candidate-row-reranker` | `BertForSequenceClassification` | 22,714,756 | 0.248s | 423 MB | `reports/encoder-classifier-smokes/candidate-row-reranker.cuda0.ai.json` |
| `repository_library/citation-prediction` | `BertForSequenceClassification` | 109,920,002 | 0.251s | 791 MB | `reports/encoder-classifier-smokes/citation-prediction.cuda0.ai.json` |
| `repository_library/cluster-router` | `BertForSequenceClassification` | 22,715,526 | 0.270s | 423 MB | `reports/encoder-classifier-smokes/cluster-router.cuda0.ai.json` |
| `repository_library/cross-encoder-reranker` | `BertForSequenceClassification` | 22,713,986 | 0.250s | 423 MB | `reports/encoder-classifier-smokes/cross-encoder-reranker.cuda0.ai.json` |
| `repository_library/cross-modal-retrieval` | `BertModel` | 22,713,216 | 0.248s | 423 MB | `reports/encoder-classifier-smokes/cross-modal-retrieval.cuda0.ai.json` |
| `repository_library/file-embedding` | `BertModel` | 22,713,216 | 0.250s | 423 MB | `reports/encoder-classifier-smokes/file-embedding.cuda0.ai.json` |
| `repository_library/graph-neighborhood-router` | `BertForSequenceClassification` | 22,715,526 | 0.261s | 423 MB | `reports/encoder-classifier-smokes/graph-neighborhood-router.cuda0.ai.json` |
| `repository_library/heap-scheduler` | `BertForSequenceClassification` | 22,715,141 | 0.251s | 423 MB | `reports/encoder-classifier-smokes/heap-scheduler.cuda0.ai.json` |
| `repository_library/jepa-repo-state` | `BertModel` | 22,713,216 | 0.294s | 423 MB | `reports/encoder-classifier-smokes/jepa-repo-state.cuda0.ai.json` |
| `repository_library/metadata-category-classifier` | `BertForSequenceClassification` | 110,705,920 | 0.267s | 791 MB | `reports/encoder-classifier-smokes/metadata-category-classifier.cuda0.ai.json` |
| `repository_library/paper-link-prediction` | `BertForSequenceClassification` | 109,920,002 | 0.287s | 791 MB | `reports/encoder-classifier-smokes/paper-link-prediction.cuda0.ai.json` |
| `repository_library/repo-embedding` | `BertModel` | 22,713,216 | 0.292s | 423 MB | `reports/encoder-classifier-smokes/repo-embedding.cuda0.ai.json` |
| `repository_library/repo-similarity` | `BertModel` | 22,713,216 | 0.271s | 688 MB | `reports/encoder-classifier-smokes/repo-similarity.cuda0.ai.json` |
| `repository_library/repo-state-grounding` | `BertForSequenceClassification` | 22,714,756 | 0.360s | 423 MB | `reports/encoder-classifier-smokes/repo-state-grounding.cuda0.ai.json` |
| `repository_library/self-play-reward-model` | `BertForSequenceClassification` | 109,920,002 | 0.250s | 805 MB | `reports/encoder-classifier-smokes/self-play-reward-model.cuda0.ai.json` |
| `repository_library/span-infill-gate` | `BertForSequenceClassification` | 22,714,371 | 0.283s | 423 MB | `reports/encoder-classifier-smokes/span-infill-gate.cuda0.ai.json` |
| `repository_library/verifier-accept-policy` | `BertForSequenceClassification` | 22,714,756 | 0.274s | 423 MB | `reports/encoder-classifier-smokes/verifier-accept-policy.cuda0.ai.json` |

Patch note: `abstract-repo-planning`, `bug-localization`, `metadata-category-classifier`, and `self-play-reward-model` require `transformers_classifier_head_num_labels_from_checkpoint`; their checkpoint classifier head shape is authoritative over stale `config.num_labels`.

## sapiens2_pose_bridge

| Model | Status | Preferred Env | Local | Detail |
| --- | --- | --- | --- | --- |
| `facebook/sapiens2-pose-1b` | `verified_sapiens2_pose_load_bridge` | `ai` | yes | Custom model-stack bridge loads the original `backbone.*` / `decode_head.*` safetensors schema directly, avoiding the unavailable Transformers 5.10 dev Sapiens2 classes. BF16 load-only smoke on `cuda:0` passed in 164.64s with about 2.9GB allocated after load; full 1024x768 forward remains a separate latency/VRAM benchmark. |

## nemo_asr_bridge

| Model | Status | Preferred Env | Local | Detail |
| --- | --- | --- | --- | --- |
| `nemotron-3.5-asr-streaming-0.6b` | `needs_nemo_speech_env` | `nemo_speech` | yes | nemo_toolkit missing; use/create nemo_speech |
| `nemotron-speech-streaming-en-0.6b` | `needs_nemo_speech_env` | `nemo_speech` | yes | nemo_toolkit missing; use/create nemo_speech |
| `nvidia/nemotron-speech-streaming-en-0.6b` | `needs_nemo_speech_env` | `nemo_speech` | yes | nemo_toolkit missing; use/create nemo_speech |
| `parakeet-ctc-1.1b` | `needs_nemo_speech_env` | `nemo_speech` | yes | nemo_toolkit missing; use/create nemo_speech |
| `parakeet-rnnt-0.6b` | `needs_nemo_speech_env` | `nemo_speech` | yes | nemo_toolkit missing; use/create nemo_speech |
| `parakeet-rnnt-1.1b` | `needs_nemo_speech_env` | `nemo_speech` | yes | nemo_toolkit missing; use/create nemo_speech |

## peft_adapter_bridge

| Model | Status | Preferred Env | Local | Detail |
| --- | --- | --- | --- | --- |
| `Wan-AI--Wan2.2-Animate-14B/relighting_lora` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/abstract-code-relevance` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/abstract-code-relevance-pairs` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/abstract-keywords` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/abstract-method-summary` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/adapter-fusion` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/equation-reasoning` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/figure-table-interpretation` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/full-paper-lm` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/metadata-embedding` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/paper-conditioned-adapter` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/paper-fulltext-embedding` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/paper-qa` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/paper-sentence-embedding` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/paper-to-code` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/pdf-tokenization` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/query-rewriter` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/repo-conditioned-adapter` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/repo-paper-alignment` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/section-to-algorithm` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/unified-knowledge-model` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |
| `repository_library/world-planner-adapter` | `adapter_needs_base_model` | `match_base_model_env` | yes | adapter files present; verify with explicit base model env |

## transformers_causal_lm_bridge

| Model | Status | Preferred Env | Local | Detail |
| --- | --- | --- | --- | --- |
| `MobileLLM-125M` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK; slow `LlamaTokenizer` fallback and `use_cache=False` required because classic MobileLLM remote code expects old Transformers cache API |
| `MobileLLM-125M-layer-share` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK; same slow `LlamaTokenizer` fallback and `use_cache=False` caveat as classic MobileLLM |
| `MobileLLM-1.5B` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK; slow `LlamaTokenizer` fallback and `use_cache=False` required; single `.bin` load took 79.42s |
| `MobileLLM-1B` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK; same classic MobileLLM caveats; single `.bin` load took 58.47s |
| `MobileLLM-350M` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK; same slow `LlamaTokenizer` fallback and `use_cache=False` caveat as 125M |
| `MobileLLM-350M-layer-share` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK; same slow `LlamaTokenizer` fallback and `use_cache=False` caveat as classic MobileLLM |
| `MobileLLM-600M` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK; same classic MobileLLM caveats; single `.bin` load took 36.93s |
| `MobileLLM-Pro` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK through remote-code `MobileLLMP1ForCausalLM`; no compatibility patches needed |
| `MobileLLM-Pro-base` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK through remote-code `MobileLLMP1ForCausalLM`; no compatibility patches needed |
| `MobileLLM-Pro-base-int4-accelerator` | `works_cuda_generate_smoke_bf16_fallback` | `ai` | yes | CUDA BF16 load/generate OK, but Transformers ignored `weight_fake_quantizer.scale` tensors; not validated as optimized int4 accelerator runtime |
| `MobileLLM-Pro-base-int4-cpu` | `works_cuda_generate_smoke_bf16_direct_device_map` | `ai` | yes | CUDA BF16 load/generate OK with `device_map="cuda:0"`; checkpoint contains BF16 tensors and no quantization keys, so not validated as optimized int4 CPU runtime |
| `MobileLLM-R1.5-140M` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK with built-in `Llama4ForCausalLM` and `AutoTokenizer`; no classic tokenizer fallback needed |
| `MobileLLM-R1.5-360M` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK with built-in `Llama4ForCausalLM` and `AutoTokenizer`; no compatibility patches needed |
| `MobileLLM-R1.5-950M` | `works_cuda_generate_smoke` | `ai` | yes | CUDA BF16 load/generate OK with built-in `Llama4ForCausalLM` and `AutoTokenizer`; no compatibility patches needed |
| `cwm` | `needs_custom_bridge_or_new_transformers` | `ai` | yes | `AutoConfig` fails: Transformers 4.57.6 does not recognize `model_type: cwm`; local snapshot has no `auto_map` or `modeling_cwm.py`; missing shards 00005 and 00010-00014 of 00014; no auto_map/modeling_cwm.py; Transformers 4.57.6 does not recognize model_type cwm |
| `microsoft--FastContext-1.0-4B-SFT` | `config_tokenizer_ok_load_timeout` | `ai` | yes | `Qwen3Config` and `Qwen2TokenizerFast` load; direct `device_map="cuda:0"` checkpoint load on GPU1 was stopped after >4 minutes before generation; needs long-running/offload placement probe |

### Causal LM CUDA Smoke Reports

| Model | Params | Load | 4-token Generate | GPU Memory After Generate | Report |
| --- | ---: | ---: | ---: | ---: | --- |
| `MobileLLM-125M` | 124,635,456 | 3.99s | 2.53s | 596 MB | `reports/causal-lm-smokes/MobileLLM-125M.cuda0.ai.json` |
| `MobileLLM-125M-layer-share` | 124,635,456 | 8.12s | 0.68s | 609 MB | `reports/causal-lm-smokes/MobileLLM-125M-layer-share.cuda0.ai.json` |
| `MobileLLM-1.5B` | 1,562,388,800 | 79.42s | 1.05s | 3471 MB | `reports/causal-lm-smokes/MobileLLM-1.5B.cuda0.ai.json` |
| `MobileLLM-1B` | 1,005,461,760 | 58.47s | 0.77s | 2527 MB | `reports/causal-lm-smokes/MobileLLM-1B.cuda0.ai.json` |
| `MobileLLM-350M-layer-share` | 345,355,200 | 18.01s | 0.60s | 1043 MB | `reports/causal-lm-smokes/MobileLLM-350M-layer-share.cuda0.ai.json` |
| `MobileLLM-Pro` | 1,084,453,120 | 83.61s | 0.64s | 2533 MB | `reports/causal-lm-smokes/MobileLLM-Pro.cuda0.ai.json` |
| `MobileLLM-Pro-base` | 1,084,453,120 | 55.15s | 0.61s | 2533 MB | `reports/causal-lm-smokes/MobileLLM-Pro-base.cuda0.ai.json` |
| `MobileLLM-Pro-base-int4-accelerator` | 1,084,453,120 | 52.72s | 0.57s | 2533 MB | `reports/causal-lm-smokes/MobileLLM-Pro-base-int4-accelerator.cuda0.ai.json` |
| `MobileLLM-Pro-base-int4-cpu` | 1,084,453,120 | 5.68s with `device_map="cuda:0"` | 1.53s / 1 token | 2177 MB | `reports/causal-lm-smokes/MobileLLM-Pro-base-int4-cpu.device_map_cuda0.ai.json` |
| `MobileLLM-R1.5-950M` | 949,685,760 | 48.15s | 0.63s | 2183 MB | `reports/causal-lm-smokes/MobileLLM-R1.5-950M.cuda0.ai.json` |
| `MobileLLM-R1.5-360M` | 359,431,168 | 23.50s | 0.53s | 1067 MB | `reports/causal-lm-smokes/MobileLLM-R1.5-360M.cuda0.ai.json` |
| `MobileLLM-350M` | 345,355,200 | 19.28s | 0.54s | 1234 MB | `reports/causal-lm-smokes/MobileLLM-350M.cuda0.ai.json` |
| `MobileLLM-600M` | 603,188,352 | 36.93s | 0.76s | 1698 MB | `reports/causal-lm-smokes/MobileLLM-600M.cuda0.ai.json` |
| `MobileLLM-R1.5-140M` | 140,248,512 | 9.00s | 0.60s | 614 MB | `reports/causal-lm-smokes/MobileLLM-R1.5-140M.cuda0.ai.json` |

Classic `MobileLLM-*` failures to watch for: `AutoTokenizer` returning `False`, and `DynamicCache` missing `get_max_length` when cache is enabled. The current bridge smoke handles the tokenizer fallback and uses `--no-use-cache` for these models. `cwm` is a larger architecture gap: current Transformers does not recognize `model_type: cwm`, and the local folder lacks remote-code files/`auto_map`, so model-stack needs a dedicated CWM bridge or a Transformers build that includes CWM before runtime validation can proceed.

## video_diffusion_bridge

| Model | Status | Preferred Env | Local | Detail |
| --- | --- | --- | --- | --- |
| `AnyFlow-FAR-Wan2.1-1.3B-Diffusers` | `works_snapshot_status` | `ai` | yes | class=AnyFlowFARPipeline; components=scheduler,text_encoder,tokenizer,transformer,vae |
| `AnyFlow-Wan2.1-T2V-1.3B-Diffusers` | `works_snapshot_status` | `ai` | yes | class=AnyFlowPipeline; components=scheduler,text_encoder,tokenizer,transformer,vae |
| `Audio2Face-3D-v2.3-Mark` | `verified_onnx_load_needs_audio2face_bridge` | `ai` | yes | `network.onnx` loads with ONNX Runtime CPU provider; needs Audio2Face preprocessing/postprocessing bridge |
| `Audio2Face-3D-v3.0` | `verified_onnx_load_needs_audio2face_bridge` | `ai` | yes | `network.onnx` loads with ONNX Runtime CPU provider; needs Audio2Face preprocessing/recurrent latent/postprocessing bridge |
| `Cosmos3-Nano` | `candidate_cosmos3_lightx2v_bridge` | `ai` | yes | local snapshot complete; LightX2V Cosmos3 imports pass and all transformer shards validate; full generation pending |
| `Cosmos3-Nano-Policy-DROID` | `candidate_cosmos3_lightx2v_bridge` | `ai` | yes | local snapshot complete; LightX2V Cosmos3 imports pass and all transformer shards validate; full generation pending |
| `nvidia/Cosmos3-Nano` | `candidate_cosmos3_lightx2v_bridge` | `ai` | yes | local snapshot complete; LightX2V Cosmos3 imports pass and all transformer shards validate; full generation pending |
| `nvidia/Cosmos-Predict2.5-14B` | `candidate_cosmos25_repo_checkpoint` | `cosmos25_py310` | yes | BF16 repo-format `.pt` artifacts and runtime source validate; `cosmos25_py310` runtime import probe passes |
| `nvidia/Cosmos-Transfer2.5-2B` | `candidate_cosmos25_repo_checkpoint` | `cosmos25_py310` | yes | BF16 repo-format `.pt` artifacts and runtime source validate; `cosmos25_py310` runtime import probe passes |
| `Cosmos-Embed1-448p-anomaly-detection` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `EGM-4B` | `meta_construct_ok_checkpoint_load_pending` | `ai` | yes | Qwen3-VL config/processor/meta model construction OK; correct loader is `AutoModelForImageTextToText`; direct device-map checkpoint load timed out before generation; active 2-shard index plus extra unused 4-shard files |
| `EGM-4B-SFT` | `meta_construct_ok_checkpoint_load_pending` | `ai` | yes | Qwen3-VL config/processor/meta model construction OK; clean active 2-shard safetensors layout; full checkpoint placement/generation pending |
| `GEN3C-Cosmos-7B` | `needs_gen3c_cosmos_predict1_runtime` | `gen3c_cosmos_predict1_or_custom_bridge` | yes | local `Cosmos_GEN3C` config and 27GB PyTorch zip checkpoint metadata validate; needs nv-tlabs/Gen3C runtime on top of Cosmos-Predict1 before generation |
| `HunyuanVideo-Avatar` | `verified_hunyuan_avatar_custom_bridge_assets` | `py311build` | yes | `runtime.hunyuan_avatar_bridge` validates assets/imports/path mapping and BF16/FP8 FSDP shard dirs; launch plan generated; full generation still blocked by LLaVA image-token alignment |
| `Hunyuan3D-2.1` | `candidate_hunyuan3d_lightx2v_bridge` | `ai` | yes | LightX2V Hunyuan3D shape imports pass; `hunyuan3d-dit-v2-1` validates; full mesh generation pending |
| `Hunyuan3D-2mini` | `candidate_hunyuan3d_lightx2v_bridge` | `ai` | yes | LightX2V Hunyuan3D shape imports pass; mini, mini-fast, and mini-turbo variants validate |
| `Hunyuan3D-2mv` | `verified_hy3dgen_bridge` | `ai` | yes | official `hy3dgen` loads local mv fp16 weights and exports GLB through model-stack; LightX2V x_embedder route is bypassed |
| `Hunyuan3D-Omni` | `needs_hunyuan3d_omni_bridge` | `hunyuan3d_omni_or_custom_bridge` | yes | local Omni assets are present, but need an Omni-specific runtime/control bridge |
| `tencent/Hunyuan3D-2` | `candidate_hunyuan3d_lightx2v_bridge` | `ai` | yes | LightX2V Hunyuan3D shape imports pass; v2-0, v2-0-fast, and v2-0-turbo variants validate; paint/delight are separate bridge targets |
| `microsoft/TRELLIS.2-4B` | `verified_trellis2_official_runtime_bridge` | `trellis` | yes | official TRELLIS.2 runtime loads local 4B weights and exports GLB through model-stack with `o_voxel`; 512/1-step smoke passes |
| `Wan-AI--Wan2.2-Animate-14B` | `candidate_wan_animate_custom_bridge` | `py311build_or_custom_wan_env` | yes | custom Wan Animate 14B checkpoint complete: 4 diffusion shards, VAE, CLIP, T5, relighting LoRA present; needs model-stack Wan Animate bridge with cached T5 prompt embeds/VAE control latents and int8/offload controls |
| `DAM-3B` | `needs_dam_lazy_submodule_bridge` | `ai` | yes | config imports with trust_remote_code; remote constructor eagerly loads LLM shards via `build_llm_and_tokenizer -> AutoModelForCausalLM.from_pretrained`, so bridge should lazy-load/place submodules with explicit device_map/dtype/offload |
| `DAM-3B-Self-Contained` | `needs_dam_lazy_submodule_bridge` | `ai` | yes | self-contained `llava_llama.py` runtime present; same eager LLM load patch point as DAM-3B |
| `DAM-3B-Video` | `needs_dam_lazy_submodule_bridge` | `ai` | yes | same DAM layout with video context provider; needs lazy submodule bridge instead of blind AutoModel load |
| `PixelDiT-1300M-1024px` | `needs_pixeldit_custom_bridge` | `pixeldit_or_custom_bridge` | yes | model_type=`pixeldit`, no `auto_map`; Transformers 4.57 does not recognize it, so needs PixelDiT runtime bridge |
| `Wan-AI--Wan2.2-Animate-14B/xlm-roberta-large` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `Wan-AI--Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english` | `works_cuda_forward_smoke` | `ai` | yes | `AutoFeatureExtractor` + `AutoModelForCTC` synthetic 1s audio forward OK; report `reports/world-model-smokes/Wan-AI--Wan2.2-S2V-14B.wav2vec2_submodel.cuda0.ai.json` |
| `llama-nemotron-rerank-1b-v2` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers` | `works_snapshot_status` | `ai` | yes | class=AnyFlowFARPipeline; components=scheduler,text_encoder,tokenizer,transformer,vae |
| `nvidia/AnyFlow-Wan2.1-T2V-1.3B-Diffusers` | `works_snapshot_status` | `ai` | yes | class=AnyFlowPipeline; components=scheduler,text_encoder,tokenizer,transformer,vae |
| `pe-av-base` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `pe-av-base-16-frame` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `pe-av-large` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `pe-av-small` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `processor` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `robbyant/lingbot-video-dense-1.3b` | `needs_lingbot_video_runtime` | `lingbot_video_or_custom_bridge` | yes | snapshot complete, but `ai` lacks `LingBotVideoPipeline` and `lingbot_video.transformer_lingbot_video` |
| `robbyant/lingbot-video-dense-1.3b/processor` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `robbyant/lingbot-video-dense-1.3b/text_encoder` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `robbyant/lingbot-video-moe-30b-a3b` | `needs_lingbot_video_runtime` | `lingbot_video_or_custom_bridge` | yes | snapshot complete, but `ai` lacks `LingBotVideoPipeline` and `lingbot_video.transformer_lingbot_video` |
| `robbyant/lingbot-video-moe-30b-a3b/processor` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `robbyant/lingbot-video-moe-30b-a3b/text_encoder` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `robbyant/lingbot-video-rewriter-lora` | `works_adapter_status` | `ai` | yes | weights=adapter_model.safetensors |
| `robbyant/lingbot-world-base-act-preview` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `robbyant/lingbot-world-v2-14b-causal-fast` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `sam-audio-base` | `needs_sam_audio_runtime` | `sam_audio_or_custom_bridge` | yes | local `checkpoint.pt` and config present, but `sam_audio.SAMAudio` / `SAMAudioProcessor` runtime is absent from current envs |
| `sam-audio-large` | `needs_sam_audio_runtime` | `sam_audio_or_custom_bridge` | yes | local `checkpoint.pt` and config present, but `sam_audio.SAMAudio` / `SAMAudioProcessor` runtime is absent from current envs |
| `sam-audio-large-tv` | `needs_sam_audio_runtime` | `sam_audio_or_custom_bridge` | yes | local `checkpoint.pt` and config present, but `sam_audio.SAMAudio` / `SAMAudioProcessor` runtime is absent from current envs |
| `sam-audio-small` | `needs_sam_audio_runtime` | `sam_audio_or_custom_bridge` | yes | local `checkpoint.pt` and config present, but `sam_audio.SAMAudio` / `SAMAudioProcessor` runtime is absent from current envs |
| `sam-audio-small-tv` | `needs_sam_audio_runtime` | `sam_audio_or_custom_bridge` | yes | local `checkpoint.pt` and config present, but `sam_audio.SAMAudio` / `SAMAudioProcessor` runtime is absent from current envs |
| `text_encoder` | `needs_custom_bridge_or_env` | `ai` | yes | local path exists but no Diffusers model_index.json; verify model-specific repo/env |
| `zai-org--CogVideoX-2b` | `verified_no_generation_pipeline_load` | `ai` | yes | missing T5 shard restored; transformer, VAE, text encoder, full no-generation pipeline, and minimal 9-frame/2-step generation smoke pass |

FastContext status: `microsoft--FastContext-1.0-4B-SFT` config/tokenizer are valid standard Qwen3 assets, but direct `device_map="cuda:0"` load of the 7.6GB two-shard checkpoint did not complete within a bounded >4 minute interactive window. Report: `reports/causal-lm-smokes/microsoft--FastContext-1.0-4B-SFT.device_map_cuda1.timeout.ai.json`.
