# Model Stack Compatibility Patches

Model-stack should prefer narrow API drift patches over creating a new conda env
when the installed runtime is otherwise capable of running the model.

The contract is:

1. Probe what the local model expects.
2. Probe what the current env provides.
3. Apply a named bridge patch when the mismatch is narrow and behavior can be
   preserved.
4. Require a model-specific env only when the mismatch is structural.

Use:

```bash
PYTHONPATH=. python scripts/model_compatibility_report.py /arxiv/models/MobileLLM-125M --model-id MobileLLM-125M
```

## Current Patch Registry

| Patch ID | Model Family | Mismatch | Patch | Status |
| --- | --- | --- | --- | --- |
| `transformers_mobilellm_legacy_cache` | classic `MobileLLM-*` | Local `modeling_mobilellm.py` calls `transformers.cache_utils.DynamicCache.get_max_length`, which is absent in Transformers 4.57.6. | Run generation/forward with `use_cache=False` until the model code is patched to the new cache API. | verified on classic `MobileLLM-125M` through `MobileLLM-1.5B` |
| `transformers_mobilellm_slow_tokenizer` | classic `MobileLLM-*` | `AutoTokenizer` / fast LLaMA tokenizer can return `False` for the legacy tokenizer config. | Reject non-callable tokenizer returns and fall back to slow `transformers.LlamaTokenizer`. | verified on classic `MobileLLM-125M` through `MobileLLM-1.5B` |
| `transformers_classifier_head_num_labels_from_checkpoint` | BERT sequence classifiers | `config.num_labels` disagrees with `classifier.weight` first dimension, causing Linear weight/bias size mismatch during `AutoModelForSequenceClassification.from_pretrained`. | Read classifier head shape from `model.safetensors` and override `config.num_labels`, `id2label`, and `label2id` before loading. | verified on `abstract-repo-planning`, `bug-localization`, `metadata-category-classifier`, `self-play-reward-model` |
| `transformers_apply_chunking_to_forward_compat` | `Cosmos-Embed1-448p-anomaly-detection` | Remote code imports `transformers.modeling_utils.apply_chunking_to_forward`, which is not exported by Transformers 4.57.6. | Candidate: inject a local shim before loading remote code, then validate in `ai`. | candidate; model currently works in `py311build` |
| `diffusers_linear_activation_fallback` | ABot-World upstream runtime | `diffusers.models.activations.LinearActivation` is imported by ABot, but installed Diffusers 0.31.0 does not export it. | Provide a local `Linear -> activation` fallback for ABot's `linear-silu` feed-forward path. | verified on `acvlab--ABot-World-0-5B-LF` |
| `abot_generator_direct_cuda_device_map_bf16` | ABot-World causal Wan generator | Default upstream load spends minutes in CPU parameter materialization/casting for the 9.8GB safetensors file. | Pass `torch_dtype=bfloat16`, `low_cpu_mem_usage=True`, `use_safetensors=True`, and `device_map={"": "cuda:0"}` at generator load. | verified on `acvlab--ABot-World-0-5B-LF` |
| `safetensors_device_map_string_cuda` | ABot-World / Accelerate safetensors dispatch | Safetensors rejects `torch.device("cuda:0")` as an invalid device in this env. | Use string device map values such as `"cuda:0"`. | verified on `acvlab--ABot-World-0-5B-LF` |
| `abot_flash_attention_sdpa_fallback` | ABot-World causal cross-attention | Causal cross-attention calls `flash_attention` directly and asserts FlashAttention2 even though SDPA is available. | Fall back to `torch.nn.functional.scaled_dot_product_attention` when FlashAttention packages are absent. | verified on `acvlab--ABot-World-0-5B-LF` |
| `abot_lazy_t5_prompt_encoder` | ABot-World text encoder | `WanTextEncoder` eagerly loaded the 10.6GB `models_t5_umt5-xxl-enc-bf16.pth` during pipeline construction, causing CPU paging stalls. | Construct tokenizer at startup, keep T5 unloaded until an uncached prompt encode is requested. | verified on `acvlab--ABot-World-0-5B-LF` |
| `abot_prompt_embedding_cache` | ABot-World text encoder | Live runtime should not repeatedly load/run T5 for fixed prompts. | Cache `prompt_embeds` by prompt/tokenizer/checkpoint hash; `ABOT_WORLD_REQUIRE_PROMPT_CACHE=1` enforces cache-only prompt setup. | synthetic cache-hit verified on `acvlab--ABot-World-0-5B-LF` |
| `diffusers_anyflow_far_return_tuple_padding` | AnyFlow-FAR Diffusers pipeline | Installed `AnyFlowFARPipeline.__call__` unpacks `noise_pred, _`, but `AnyFlowFARTransformer3DModel._forward_train(..., return_dict=False)` returns a one-item tuple when `use_kv_cache=False`. | `runtime.diffusers_bridge.prepare_diffusers_pipeline` wraps only `AnyFlowFARPipeline.transformer.forward` and pads `(sample,)` to `(sample, None)` for `return_dict=False`. | verified on `AnyFlow-FAR-Wan2.1-1.3B-Diffusers` tiny latent prompt-embeds smoke |
| `nemo_prefer_archive_over_external_transformers_metadata` | Parakeet / NeMo ASR archives | External model metadata advertises Transformers 5.x, while NeMo 2.7.3 uses Transformers 4.57.x. | Prefer local `*.nemo` archive restore with `ASRModel.restore_from`; ignore external Transformers metadata unless restore raises a Transformers API error. | verified on `parakeet-tdt_ctc-110m` and `parakeet-rnnt-0.6b` |

## Status Meanings

| Status | Meaning |
| --- | --- |
| `patch_available` | The current env lacks the expected API, and model-stack has a narrow runtime patch. |
| `patch_candidate` | The drift is narrow enough to patch, but the patch still needs validation in the target env. |
| `not_needed` | The env already provides the expected API. |
| `needs_env_package` | A required coarse package is missing; this is not an API drift patch. |

## Generated Reports

Current reports:

| Model | Env | Report |
| --- | --- | --- |
| `MobileLLM-125M` | `ai` | `reports/compatibility/MobileLLM-125M.ai.json` |
| `Cosmos-Embed1-448p-anomaly-detection` | `ai` | `reports/compatibility/Cosmos-Embed1-448p-anomaly-detection.ai.json` |
| `parakeet-rnnt-0.6b` | `nemo_speech` | `reports/compatibility/parakeet-rnnt-0.6b.nemo_speech.json` |

The causal LM smoke harness also embeds compatibility decisions directly in its
JSON output. Example:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n ai env PYTHONNOUSERSITE=1 python scripts/smoke_transformers_causal_lm_bridge.py /arxiv/models/MobileLLM-125M --model-id MobileLLM-125M --device cuda:0 --dtype bfloat16 --generate --max-new-tokens 1 --no-use-cache --json-out reports/causal-lm-smokes/MobileLLM-125M.compat.cuda0.ai.json
```

That report records both `compatibility.patches` and
`compatibility_patches_applied`.

| `compat:transformers_utils_flax_weights_name` | `HunyuanVideo-Avatar` | `py311build` Diffusers imports `transformers.utils.FLAX_WEIGHTS_NAME`, but the installed Transformers no longer exports it there. The bridge injects `FLAX_WEIGHTS_NAME = 'flax_model.msgpack'` before importing `hymm_sp.sample_inference_audio`. |
