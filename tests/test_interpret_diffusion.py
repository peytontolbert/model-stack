from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("torch.nn")
import torch.nn as nn

from interpret import (
    DiffusionModelAdapter,
    DiffusionTracer,
    MultimodalInputs,
    MultimodalModelAdapter,
    MultimodalTracer,
    classifier_logit_score,
    collect_diffusion_attention_maps,
    diffusion_attention_phase_summary,
    diffusion_module_patch_sweep,
    mse_distance,
    patch_denoiser_latents,
    patch_diffusion_module_outputs,
    prompt_counterfactual_delta,
    prompt_token_occlusion_importance,
    summarize_diffusion_steps,
    summarize_diffusion_trace_dataset,
    summarize_prompt_token_attribution,
    tensor_mean_score,
    tensor_region_score,
    token_region_attribution,
    trace_diffusion_generation,
    trace_multimodal_forward,
    trace_prompt_dataset,
)
from interpret.diffusion.metrics import cosine_distance, extract_tensor_output


class _CrossAttention(nn.Module):
    is_cross_attention = True

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None):
        tokens = 4 if encoder_hidden_states is None else int(encoder_hidden_states.shape[1])
        probs = torch.softmax(torch.arange(x.shape[1] * tokens, dtype=x.dtype, device=x.device).reshape(1, 1, x.shape[1], tokens), dim=-1)
        return x + probs.mean(dim=-1).transpose(1, 2), probs


class _Denoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn2 = _CrossAttention()

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor | None = None):
        x = sample.flatten(2).transpose(1, 2)
        x, _probs = self.attn2(x, encoder_hidden_states=encoder_hidden_states)
        return types.SimpleNamespace(sample=x.transpose(1, 2).reshape_as(sample) + timestep.float().view(1, 1, 1, 1) * 0.01)


class _FakeTokenizer:
    def tokenize(self, prompt: str):
        return prompt.split()


class _Pipeline:
    def __init__(self) -> None:
        self.unet = _Denoiser()
        self.tokenizer = _FakeTokenizer()

    def __call__(self, prompt: str, num_inference_steps: int = 2, latents: torch.Tensor | None = None, **_kwargs):
        sample = torch.zeros(1, 1, 2, 2) if latents is None else latents.clone()
        token_count = max(1, len(prompt.split()))
        enc = torch.ones(1, token_count, 3) * float(token_count)
        for step in range(num_inference_steps):
            sample = self.unet(sample, torch.tensor(step), encoder_hidden_states=enc).sample
        return types.SimpleNamespace(images=sample + float(token_count))


def test_diffusion_adapter_discovers_components_and_attention() -> None:
    pipe = _Pipeline()
    adapter = DiffusionModelAdapter(pipe)
    assert adapter.denoiser_module() is pipe.unet
    assert adapter.tokenize_prompt("red cube") == ["red", "cube"]
    targets = adapter.cross_attention_targets()
    assert [target.name for target in targets] == ["denoiser.attn2"]


def test_diffusion_tracer_captures_steps_latents_and_attention() -> None:
    pipe = _Pipeline()
    tracer = DiffusionTracer(pipe)
    output, cache, records = tracer.trace_generation("red cube", num_inference_steps=2)

    assert output.images.shape == (1, 1, 2, 2)
    assert len(records) == 2
    assert records[0].timestep == 0
    assert cache.get("denoiser.step_0.latent_in") is not None
    assert cache.get("denoiser.step_1.latent_out") is not None
    probs = cache.get("denoiser.attn2.step_0.attn_probs")
    assert probs is not None
    assert probs.shape[-2:] == (4, 2)
    assert summarize_diffusion_steps(records)["steps"] == 2
    phase = diffusion_attention_phase_summary(cache)
    assert phase["steps"] == [0, 1]
    assert len(collect_diffusion_attention_maps(cache)) == 2

    output2, cache2, records2 = trace_diffusion_generation(pipe, "blue square", num_inference_steps=1)
    assert output2.images.shape == (1, 1, 2, 2)
    assert cache2.get("denoiser.step_0.latent_in") is not None
    assert len(records2) == 1


def test_prompt_attribution_and_metrics() -> None:
    pipe = _Pipeline()
    rows = prompt_token_occlusion_importance(pipe, "red striped cube", generation_kwargs={"num_inference_steps": 1})
    assert [row.token for row in rows] == ["red", "striped", "cube"]
    assert all(row.delta >= 0 for row in rows)
    assert summarize_prompt_token_attribution(rows, topk=2)[0]["token"] in {"red", "striped", "cube"}

    delta = prompt_counterfactual_delta(pipe, "red cube", "blue cube", generation_kwargs={"num_inference_steps": 1})
    assert delta.ndim == 0
    assert mse_distance({"images": torch.zeros(1, 1)}, {"images": torch.ones(1, 1)}).item() == 1.0
    assert cosine_distance(torch.ones(2), torch.ones(2)).abs().item() < 1e-6
    assert torch.equal(extract_tensor_output({"sample": torch.ones(1)}), torch.ones(1))


def test_token_region_attribution_aggregates_attention_maps() -> None:
    attn = torch.zeros(1, 2, 4, 3)
    attn[..., 0] = 0.2
    attn[..., 1] = 0.5
    attn[..., 2] = 0.3
    heatmaps = token_region_attribution(attn, ["red", "cube"], spatial_shape=(2, 2))
    assert set(heatmaps) == {"red", "cube"}
    assert heatmaps["red"].shape == (2, 2)
    assert torch.allclose(heatmaps["cube"], torch.ones(2, 2))


def test_diffusion_interventions_recover_clean_generation() -> None:
    pipe = _Pipeline()
    latent_result = patch_denoiser_latents(
        pipe,
        clean_prompt="red cube",
        corrupted_prompt="red striped cube",
        patch_steps=[0],
        num_inference_steps=1,
    )
    assert latent_result.patched_keys == ("denoiser.step_0.latent_in",)
    assert isinstance(latent_result.recovery_fraction, float)

    module_result = patch_diffusion_module_outputs(
        pipe,
        clean_prompt="red cube",
        corrupted_prompt="red striped cube",
        module_names=["denoiser.attn2"],
        patch_steps=[0],
        num_inference_steps=1,
    )
    assert module_result.patched_keys == ("denoiser.attn2.step_0.out",)
    assert isinstance(module_result.recovery_fraction, float)

    sweep = diffusion_module_patch_sweep(
        pipe,
        clean_prompt="red cube",
        corrupted_prompt="red striped cube",
        module_names=["denoiser.attn2"],
        patch_steps=[0],
        num_inference_steps=1,
    )
    assert sweep["recovery"].shape == (1, 1)


def test_diffusion_dataset_and_objectives() -> None:
    pipe = _Pipeline()
    dataset = trace_prompt_dataset(pipe, ["red cube", "blue square"], num_inference_steps=1, score_fn=tensor_mean_score)
    summary = summarize_diffusion_trace_dataset(dataset)
    assert summary["examples"] == 2
    assert summary["scored_examples"] == 2
    assert dataset.scores().shape == (2,)

    score = tensor_region_score(torch.ones(1, 1, 2, 2))(types.SimpleNamespace(images=torch.ones(1, 1, 2, 2)))
    assert score.item() == 1.0
    classifier_score = classifier_logit_score(lambda image: torch.tensor([[0.5, 1.5]]), 1)
    assert classifier_score(types.SimpleNamespace(images=torch.ones(1, 1, 2, 2))).item() == 1.5


class _TinyVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_tower = nn.Linear(4, 3)
        self.mm_projector = nn.Linear(3, 2)
        self.language_model = nn.Linear(2, 5)

    def forward(self, input_ids: torch.Tensor | None = None, pixel_values: torch.Tensor | None = None, **_kwargs):
        vision = self.vision_tower(pixel_values.flatten(1))
        projected = self.mm_projector(vision)
        return self.language_model(projected)


def test_multimodal_adapter_and_tracer_capture_components() -> None:
    model = _TinyVLM()
    adapter = MultimodalModelAdapter(model)
    assert [component.role for component in adapter.components()] == ["vision", "projector", "language"]
    inputs = MultimodalInputs(input_ids=torch.ones(1, 2, dtype=torch.long), pixel_values=torch.ones(1, 1, 2, 2))
    tracer = MultimodalTracer(model)
    output, cache = tracer.trace_forward(inputs)
    assert output.shape == (1, 5)
    assert cache.get("vision_encoder") is not None
    assert cache.get("projector") is not None
    assert cache.get("language_model") is not None

    output2, cache2 = trace_multimodal_forward(model, inputs)
    assert output2.shape == (1, 5)
    assert cache2.get("vision_encoder") is not None
