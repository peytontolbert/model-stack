from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


PREDICT_RUNTIME_CANDIDATES = (
    Path("/data/clone/third_party/cosmos-predict2.5"),
    Path("/data/clone/cosmos-predict2.5"),
    Path("/data/repositories/cosmos-predict2.5"),
    Path("/home/peyton/src/cosmos-predict2.5"),
)
TRANSFER_RUNTIME_CANDIDATES = (
    Path("/data/clone/third_party/cosmos-transfer2.5"),
    Path("/data/clone/cosmos-transfer2.5"),
    Path("/data/repositories/cosmos-transfer2.5"),
    Path("/home/peyton/src/cosmos-transfer2.5"),
)


@dataclass(frozen=True)
class Cosmos25LaunchPlan:
    env: dict[str, str]
    command: tuple[str, ...]
    cwd: str
    mode: str
    checkpoint_path: str
    output_dir: str

@dataclass(frozen=True)
class Cosmos25Status:
    model_id: str
    model_path: str
    family: str
    status: str
    runnable: bool
    preferred_env: str
    loader: str
    recommended_dtype: str
    supports_text: bool
    supports_image: bool
    supports_video: bool
    detail: str
    blockers: tuple[str, ...] = ()
    present_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    runtime_candidates: tuple[str, ...] = ()


def cosmos25_status(model_path: str | Path, *, model_id: str | None = None) -> Cosmos25Status:
    path = Path(model_path)
    resolved_id = model_id or path.name
    lower_id = resolved_id.lower()
    if "cosmos-predict2.5" in lower_id:
        return _cosmos25_predict_status(path, resolved_id)
    if "cosmos-transfer2.5" in lower_id:
        return _cosmos25_transfer_status(path, resolved_id)
    return Cosmos25Status(
        model_id=resolved_id,
        model_path=str(path),
        family="cosmos25",
        status="not_cosmos25",
        runnable=False,
        preferred_env="manual",
        loader="none",
        recommended_dtype="bfloat16",
        supports_text=False,
        supports_image=False,
        supports_video=False,
        detail="Model id does not match a Cosmos 2.5 family.",
        blockers=("not a Cosmos 2.5 model id",),
    )


def status_to_json(status: Cosmos25Status) -> str:
    return json.dumps(asdict(status), indent=2, sort_keys=True)


def _cosmos25_predict_status(path: Path, model_id: str) -> Cosmos25Status:
    expected = {
        "README.md": path / "README.md",
        "base/pre-trained ema_bf16 checkpoint": path / "base" / "pre-trained" / "54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt",
        "base/post-trained ema_bf16 checkpoint": path / "base" / "post-trained" / "e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt",
    }
    return _status_from_expected(
        path=path,
        model_id=model_id,
        family="cosmos25_predict",
        expected=expected,
        runtime_candidates=PREDICT_RUNTIME_CANDIDATES,
        loader="runtime.cosmos25_bridge + nvidia-cosmos/cosmos-predict2.5",
        detail=(
            "Cosmos-Predict2.5-14B local repo-format BF16 checkpoints are present when both base/pre-trained "
            "and base/post-trained .pt files validate. This is not a Diffusers snapshot; it requires the "
            "nvidia-cosmos/cosmos-predict2.5 runtime bridge before generation."
        ),
    )


def _cosmos25_transfer_status(path: Path, model_id: str) -> Cosmos25Status:
    expected = {
        "README.md": path / "README.md",
        "auto/multiview checkpoint A": path / "auto" / "multiview" / "4ecc66e9-df19-4aed-9802-0d11e057287a_ema_bf16.pt",
        "auto/multiview checkpoint B": path / "auto" / "multiview" / "b5ab002d-a120-4fbf-a7f9-04af8615710b_ema_bf16.pt",
        "distilled/general/edge checkpoint": path / "distilled" / "general" / "edge" / "41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt",
    }
    return _status_from_expected(
        path=path,
        model_id=model_id,
        family="cosmos25_transfer",
        expected=expected,
        runtime_candidates=TRANSFER_RUNTIME_CANDIDATES,
        loader="runtime.cosmos25_bridge + nvidia-cosmos/cosmos-transfer2.5",
        detail=(
            "Cosmos-Transfer2.5-2B local repo-format BF16 checkpoints are present when the auto/multiview "
            "and distilled/general/edge .pt files validate. This is not a Diffusers snapshot; it requires "
            "the nvidia-cosmos/cosmos-transfer2.5 runtime bridge before generation."
        ),
    )


def _status_from_expected(
    *,
    path: Path,
    model_id: str,
    family: str,
    expected: dict[str, Path],
    runtime_candidates: tuple[Path, ...],
    loader: str,
    detail: str,
) -> Cosmos25Status:
    present = tuple(name for name, artifact_path in expected.items() if artifact_path.is_file())
    missing = tuple(name for name, artifact_path in expected.items() if not artifact_path.is_file())
    runtime_roots = tuple(str(candidate) for candidate in runtime_candidates)
    runtime_present = tuple(str(candidate) for candidate in runtime_candidates if (candidate / ".git").exists() or candidate.is_dir())
    blockers = tuple(f"missing artifact: {name}" for name in missing)
    if not runtime_present:
        blockers += ("missing upstream Cosmos 2.5 runtime checkout",)
    status = "candidate_cosmos25_repo_checkpoint" if not missing and runtime_present else "needs_cosmos25_runtime"
    return Cosmos25Status(
        model_id=model_id,
        model_path=str(path),
        family=family,
        status=status,
        runnable=False,
        preferred_env="cosmos25_py310",
        loader=loader,
        recommended_dtype="bfloat16",
        supports_text=True,
        supports_image=True,
        supports_video=True,
        detail=detail,
        blockers=blockers or ("Cosmos 2.5 runtime bridge not wired into load_world_model yet.",),
        present_artifacts=present,
        missing_artifacts=missing,
        runtime_candidates=runtime_roots,
    )



def build_cosmos25_predict_launch_plan(
    model_path: str | Path,
    *,
    runtime_root: str | Path = PREDICT_RUNTIME_CANDIDATES[0],
    input_file: str = "assets/base/snowy_stop_light.json",
    output_dir: str = "/data/tmp/model-stack-smokes/cosmos25/predict25_14b",
    model: str = "14B/post-trained",
    checkpoint_variant: str = "post-trained",
    inference_type: str = "text2world",
    offload_diffusion_model: bool = True,
    offload_tokenizer: bool = True,
    offload_text_encoder: bool = True,
    disable_guardrails: bool = True,
) -> Cosmos25LaunchPlan:
    path = Path(model_path)
    checkpoint_name = (
        "e21d2a49-4747-44c8-ba44-9f6f9243715f_ema_bf16.pt"
        if checkpoint_variant == "post-trained"
        else "54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt"
    )
    checkpoint_path = path / "base" / checkpoint_variant / checkpoint_name
    root = Path(runtime_root)
    command = [
        "conda", "run", "--no-capture-output", "-n", "cosmos25_py310",
        "python", "examples/inference.py",
        "-i", input_file,
        "-o", output_dir,
        f"--model={model}",
        f"--checkpoint-path={checkpoint_path}",
        f"--inference-type={inference_type}",
    ]
    if offload_diffusion_model:
        command.append("--offload-diffusion-model")
    if offload_tokenizer:
        command.append("--offload-tokenizer")
    if offload_text_encoder:
        command.append("--offload-text-encoder")
    if disable_guardrails:
        command.append("--disable-guardrails")
    return Cosmos25LaunchPlan(
        env=_cosmos25_env(root),
        command=tuple(command),
        cwd=str(root),
        mode="cosmos25_predict_official_example",
        checkpoint_path=str(checkpoint_path),
        output_dir=output_dir,
    )


def build_cosmos25_transfer_launch_plan(
    model_path: str | Path,
    *,
    runtime_root: str | Path = TRANSFER_RUNTIME_CANDIDATES[0],
    input_file: str = "assets/robot_example/distilled/edge/robot_edge_spec.json",
    output_dir: str = "/data/tmp/model-stack-smokes/cosmos25/transfer25_edge_distilled",
    model: str = "edge/distilled",
    checkpoint_path: str | None = None,
    disable_guardrails: bool = True,
) -> Cosmos25LaunchPlan:
    path = Path(model_path)
    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path else path / "distilled" / "general" / "edge" / "41f07f13-f2e4-4e34-ba4c-86f595acbc20_ema_bf16.pt"
    root = Path(runtime_root)
    command = [
        "conda", "run", "--no-capture-output", "-n", "cosmos25_py310",
        "python", "examples/inference.py",
        "-i", input_file,
        "-o", output_dir,
        f"--model={model}",
        f"--checkpoint-path={resolved_checkpoint}",
        "--disable-guardrails" if disable_guardrails else "",
    ]
    command = [part for part in command if part]
    return Cosmos25LaunchPlan(
        env=_cosmos25_env(root, experimental=True),
        command=tuple(command),
        cwd=str(root),
        mode="cosmos25_transfer_official_example",
        checkpoint_path=str(resolved_checkpoint),
        output_dir=output_dir,
    )


def _cosmos25_env(runtime_root: Path, *, experimental: bool = False) -> dict[str, str]:
    pythonpath = ":".join(
        str(path)
        for path in (
            runtime_root,
            runtime_root / "packages" / "cosmos-oss",
            runtime_root / "packages" / "cosmos-cuda",
        )
    )
    env = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": pythonpath,
        "HF_HOME": "/data/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/data/huggingface/hub",
        "TRANSFORMERS_CACHE": "/data/huggingface/hub",
    }
    if experimental:
        env["COSMOS_EXPERIMENTAL_CHECKPOINTS"] = "1"
    return env
