from __future__ import annotations

import json
import os
import subprocess
import time
from uuid import uuid4
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from runtime.hunyuan3d_lightx2v_bridge import Hunyuan3DLightX2VPaths, hunyuan3d_lightx2v_status

HF_CACHE = Path("/data/huggingface/hub")
LIGHTX2V_ROOT = Path("/data/clone/third_party/LightX2V")


@dataclass(frozen=True)
class ThreeDGenStatus:
    model_id: str
    family: str
    local_path: str
    status: str
    runnable: bool
    preferred_env: str
    runtime_bridge: str
    api_strategy: str
    outputs: tuple[str, ...]
    present_artifacts: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    compat_patches: tuple[str, ...] = ()
    dependency_profile: dict[str, str] | None = None
    detail: str = ""


@dataclass(frozen=True)
class ThreeDGenConflictReport:
    status: str
    bridge_strategy: str
    trellis: ThreeDGenStatus
    hunyuan3d: ThreeDGenStatus
    conflicts: tuple[str, ...]
    bridge_patches: tuple[str, ...]


@dataclass(frozen=True)
class ThreeDGenRequest:
    backend: str
    image_path: str
    output_dir: str
    prompt: str | None = None
    model_id: str | None = None
    model_path: str | None = None
    output_format: str = "glb"
    seed: int = 42
    variant: str | None = None
    texture: bool = False
    extra_args: dict[str, Any] | None = None


@dataclass(frozen=True)
class ThreeDGenResult:
    backend: str
    status: str
    env: str
    command: tuple[str, ...]
    request_path: str
    output_dir: str
    artifacts: dict[str, str]
    returncode: int | None = None
    elapsed_sec: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: str | None = None
    dry_run: bool = False


class ThreeDGenBridgeError(RuntimeError):
    pass


def _hf_snapshot(repo_cache: Path) -> Path | None:
    ref = repo_cache / "refs" / "main"
    if not ref.is_file():
        return None
    rev = ref.read_text(encoding="utf-8").strip()
    snap = repo_cache / "snapshots" / rev
    return snap if snap.exists() else None


def trellis2_status(model_id: str = "microsoft/TRELLIS.2-4B", hf_cache: Path = HF_CACHE) -> ThreeDGenStatus:
    repo_cache = hf_cache / "models--microsoft--TRELLIS.2-4B"
    snap = _hf_snapshot(repo_cache)
    present: list[str] = []
    missing: list[str] = []
    pipeline_name = None
    if snap is None:
        missing.append("HF snapshot microsoft/TRELLIS.2-4B")
        local_path = str(repo_cache)
    else:
        local_path = str(snap)
        pipeline = snap / "pipeline.json"
        if pipeline.is_file():
            present.append("pipeline.json")
            try:
                pipeline_name = json.loads(pipeline.read_text(encoding="utf-8")).get("name")
            except json.JSONDecodeError:
                missing.append("valid pipeline.json")
        else:
            missing.append("pipeline.json")
        expected = (
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        )
        for rel in expected:
            if (snap / rel).is_file():
                present.append(rel)
            else:
                missing.append(rel)

    blockers = []
    if missing:
        blockers.insert(0, "missing local TRELLIS artifacts: " + ", ".join(missing))

    return ThreeDGenStatus(
        model_id=model_id,
        family="trellis2",
        local_path=local_path,
        status="verified_trellis2_official_runtime_bridge" if not missing else "incomplete_trellis2_snapshot",
        runnable=not missing,
        preferred_env="trellis",
        runtime_bridge="runtime.three_d_gen_bridge.trellis2_worker_official_runtime",
        api_strategy="out_of_process_worker_returning_glb_or_mesh_bundle",
        outputs=("glb", "mesh", "texture", "gaussian_or_radiance_field_optional"),
        present_artifacts=tuple(present),
        missing_artifacts=tuple(missing),
        blockers=tuple(blockers),
        compat_patches=(
            "trellis2_official_runtime_source_registration",
            "trellis2_o_voxel_exporter",
            "hf_cache_env_routing",
            "shared_mesh_artifact_contract_glb_obj_ply_texture",
        ),
        dependency_profile={
            "python": "3.10.16",
            "torch": "2.6.0+cu124",
            "diffusers": "0.36.0",
            "transformers": "4.57.6",
            "spconv": "2.3.8 (spconv-cu121)",
            "kaolin": "0.17.0",
            "nvdiffrast": "0.4.0",
            "numpy": "1.26.4",
            "pipeline": pipeline_name or "Trellis2ImageTo3DPipeline",
            "runtime_source": "/data/clone/third_party/TRELLIS.2",
            "runtime_revision": "75fbf0183001ed9876c8dbb35de6b68552ee08bd",
            "load_report": "reports/world-model-smokes/trellis2.official_runtime.load_only.trellis.json",
            "generation_report": "reports/world-model-smokes/trellis2.official_runtime.tiny_generate.trellis.json",
        },
        detail="Local TRELLIS.2-4B assets are HF-cache snapshots with pipeline.json and safetensors modules. Model-stack uses the official microsoft/TRELLIS.2 runtime at /data/clone/third_party/TRELLIS.2, routes HF cache to /data/huggingface, and exports GLB through the installed o_voxel extension. Load-only and 512/1-step GLB generation smokes pass in trellis.",
    )


def hunyuan3d_status(model_path: str | Path = "/arxiv/models/Hunyuan3D-2mv", model_id: str = "Hunyuan3D-2mv") -> ThreeDGenStatus:
    h = hunyuan3d_lightx2v_status(Hunyuan3DLightX2VPaths(model_path=Path(model_path), lightx2v_root=LIGHTX2V_ROOT), model_id=model_id)
    blockers = list(h.blockers) if not h.runnable else []
    compat = [
        "official_hy3dgen_runtime_source_registration",
        "hunyuan3d_checkpoint_schema_router_prefers_hy3dgen_for_hy3dgen_targets",
        "hunyuan3d_mv_single_image_front_view_adapter",
        "hunyuan3d_empty_mesh_structured_status",
    ]

    return ThreeDGenStatus(
        model_id=h.model_id,
        family="hunyuan3d_hy3dgen",
        local_path=h.model_path,
        status="verified_hy3dgen_bridge" if h.runnable else h.status,
        runnable=h.runnable,
        preferred_env="ai",
        runtime_bridge="runtime.three_d_gen_bridge.hunyuan3d_worker_hy3dgen",
        api_strategy="out_of_process_worker_returning_glb_or_mesh_bundle",
        outputs=("glb", "mesh"),
        present_artifacts=h.present_artifacts,
        missing_artifacts=h.missing_artifacts,
        blockers=tuple(blockers),
        compat_patches=tuple(compat),
        dependency_profile={
            "python": "3.11.11",
            "torch": "2.10.0+cu128",
            "diffusers": "0.39.0.dev0",
            "transformers": "4.57.6",
            "safetensors": "0.8.0rc1",
            "numpy": "2.2.6",
            "runtime_source": "/data/clone/third_party/Hunyuan3D-2",
            "load_report": "reports/world-model-smokes/hunyuan3d.hy3dgen.load_only.ai.json",
            "generation_report": "reports/world-model-smokes/hunyuan3d.hy3dgen.smoke_generate.ai.json",
        },
        detail="Hunyuan3D-2mv is verified through the official hy3dgen runtime in ai. LightX2V is retained only as an asset/import probe; generation uses hy3dgen and a single-image front-view adapter for the mv pipeline. Smoke export passed with 5 steps, octree 128, output GLB.",
    )


def compare_trellis_hunyuan3d() -> ThreeDGenConflictReport:
    trellis = trellis2_status()
    hunyuan = hunyuan3d_status()
    conflicts = (
        "Python ABI split: TRELLIS env is Python 3.10; Hunyuan3D LightX2V route is Python 3.11.",
        "Torch/CUDA split: TRELLIS env is torch 2.6.0+cu124; Hunyuan3D ai env is torch 2.10.0+cu128.",
        "Sparse geometry deps are TRELLIS-only today: spconv-cu121, kaolin, nvdiffrast. They are absent from ai.",
        "Hunyuan3D execution uses official hy3dgen from /data/clone/third_party/Hunyuan3D-2 in ai; TRELLIS.2 execution uses /data/clone/third_party/TRELLIS.2 in trellis.",
        "Numerics/package split: TRELLIS uses numpy 1.26.4/safetensors 0.7.0; ai uses numpy 2.2.6/safetensors 0.8.0rc1.",
        "Runtime API split is handled by official source routing: TRELLIS.2 uses /data/clone/third_party/TRELLIS.2 in trellis, while Hunyuan3D uses /data/clone/third_party/Hunyuan3D-2 in ai.",
    )
    patches = (
        "Use a common model-stack 3D API and dispatch each backend in a subprocess/worker pinned to its env.",
        "Normalize outputs to a mesh artifact bundle: glb preferred, with obj/ply/texture sidecars as needed.",
        "Add backend status probes that never torch.load pickle checkpoints; inspect JSON/safetensors/zip metadata only.",
        "For Hunyuan3D, route hy3dgen-target configs to official hy3dgen instead of the LightX2V x_embedder loader.",
        "For TRELLIS, use the official TRELLIS.2 source and o_voxel exporter; keep it isolated in the trellis env.",
    )
    return ThreeDGenConflictReport(
        status="needs_out_of_process_3d_gen_bridge",
        bridge_strategy="single model-stack API, env-isolated backend workers, shared mesh artifact contract",
        trellis=trellis,
        hunyuan3d=hunyuan,
        conflicts=conflicts,
        bridge_patches=patches,
    )



def _backend_env(backend: str) -> str:
    if backend == "trellis":
        return "trellis"
    if backend == "hunyuan3d":
        return "ai"
    raise ValueError(f"unsupported 3D backend: {backend}")


def _default_model_path(backend: str) -> str:
    if backend == "hunyuan3d":
        return "/arxiv/models/Hunyuan3D-2mv"
    if backend == "trellis":
        status = trellis2_status()
        return status.local_path
    raise ValueError(f"unsupported 3D backend: {backend}")


def _default_model_id(backend: str) -> str:
    if backend == "hunyuan3d":
        return "Hunyuan3D-2mv"
    if backend == "trellis":
        return "microsoft/TRELLIS.2-4B"
    raise ValueError(f"unsupported 3D backend: {backend}")


def _request_payload(request: ThreeDGenRequest) -> dict[str, Any]:
    payload = asdict(request)
    payload["model_id"] = request.model_id or _default_model_id(request.backend)
    payload["model_path"] = request.model_path or _default_model_path(request.backend)
    return payload


def write_3d_request(request: ThreeDGenRequest, output_dir: str | Path | None = None) -> Path:
    payload = _request_payload(request)
    out_dir = Path(output_dir or request.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    request_path = out_dir / f"three_d_gen_request_{request.backend}_{uuid4().hex}.json"
    request_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return request_path


def build_3d_worker_command(request_path: str | Path, backend: str, result_path: str | Path | None = None) -> tuple[str, ...]:
    env_name = _backend_env(backend)
    cmd = [
        "conda",
        "run",
        "-n",
        env_name,
        "env",
        "PYTHONNOUSERSITE=1",
        "PYTHONPATH=.",
        "HF_HOME=/data/huggingface",
        "HUGGINGFACE_HUB_CACHE=/data/huggingface/hub",
        "TRANSFORMERS_CACHE=/data/huggingface/hub",
        "python",
        "scripts/three_d_gen_worker.py",
        "--request-json",
        str(request_path),
    ]
    if result_path is not None:
        cmd.extend(["--result-json", str(result_path)])
    return tuple(cmd)


def expected_artifacts(request: ThreeDGenRequest) -> dict[str, str]:
    output_dir = Path(request.output_dir)
    stem = request.extra_args.get("output_stem", "mesh") if request.extra_args else "mesh"
    artifacts: dict[str, str] = {}
    if request.output_format == "glb":
        artifacts["glb"] = str(output_dir / f"{stem}.glb")
    else:
        artifacts[request.output_format] = str(output_dir / f"{stem}.{request.output_format}")
    if request.texture:
        artifacts["textured_glb"] = str(output_dir / f"{stem}_textured.glb")
    return artifacts


def generate_3d(request: ThreeDGenRequest, *, dry_run: bool = False, timeout_sec: int | None = None) -> ThreeDGenResult:
    if request.backend not in {"trellis", "hunyuan3d"}:
        raise ValueError(f"unsupported 3D backend: {request.backend}")
    if not request.image_path:
        raise ValueError("image_path is required")
    out_dir = Path(request.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    req_path = write_3d_request(request, out_dir)
    result_path = out_dir / f"three_d_gen_result_{request.backend}_{req_path.stem.rsplit('_', 1)[-1]}.json"
    cmd = build_3d_worker_command(req_path, request.backend, result_path)
    artifacts = expected_artifacts(request)
    env_name = _backend_env(request.backend)
    if dry_run:
        return ThreeDGenResult(
            backend=request.backend,
            status="dry_run",
            env=env_name,
            command=cmd,
            request_path=str(req_path),
            output_dir=str(out_dir),
            artifacts=artifacts,
            dry_run=True,
        )

    start = time.monotonic()
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=timeout_sec)
    elapsed = time.monotonic() - start
    stdout_tail = proc.stdout[-4000:]
    stderr_tail = proc.stderr[-4000:]
    status = "ok" if proc.returncode == 0 else "failed"
    error = None if proc.returncode == 0 else f"worker exited {proc.returncode}"
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            status = payload.get("status", status)
            artifacts = payload.get("artifacts", artifacts)
            error = payload.get("error", error)
        except json.JSONDecodeError:
            error = error or "worker result JSON was invalid"
    return ThreeDGenResult(
        backend=request.backend,
        status=status,
        env=env_name,
        command=cmd,
        request_path=str(req_path),
        output_dir=str(out_dir),
        artifacts=artifacts,
        returncode=proc.returncode,
        elapsed_sec=elapsed,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        error=error,
    )


def to_json(obj: Any) -> dict[str, Any]:
    return asdict(obj)
