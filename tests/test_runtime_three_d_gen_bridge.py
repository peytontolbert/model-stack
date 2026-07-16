from __future__ import annotations

import json

from runtime.three_d_gen_bridge import ThreeDGenRequest, build_3d_worker_command, generate_3d, compare_trellis_hunyuan3d, trellis2_status


def test_trellis2_status_uses_env_isolated_bridge():
    status = trellis2_status()

    assert status.family == "trellis2"
    assert status.preferred_env == "trellis"
    assert status.api_strategy == "out_of_process_worker_returning_glb_or_mesh_bundle"
    assert "torch" in (status.dependency_profile or {})
    assert status.status == "verified_trellis2_official_runtime_bridge"
    assert status.runnable is True
    assert not status.blockers


def test_compare_trellis_hunyuan3d_records_version_conflicts():
    report = compare_trellis_hunyuan3d()

    assert report.status == "needs_out_of_process_3d_gen_bridge"
    assert report.trellis.preferred_env == "trellis"
    assert report.hunyuan3d.preferred_env == "ai"
    assert any("Python ABI" in conflict for conflict in report.conflicts)
    assert any("Torch/CUDA" in conflict for conflict in report.conflicts)
    assert any("subprocess" in patch for patch in report.bridge_patches)
    assert report.hunyuan3d.status == "verified_hy3dgen_bridge"
    assert report.hunyuan3d.runnable is True
    assert report.trellis.status == "verified_trellis2_official_runtime_bridge"
    assert report.trellis.runnable is True


def test_generate_3d_hunyuan3d_dry_run_builds_ai_worker_command(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"not-a-real-image")
    out = tmp_path / "out"

    result = generate_3d(ThreeDGenRequest(backend="hunyuan3d", image_path=str(image), output_dir=str(out)), dry_run=True)

    assert result.status == "dry_run"
    assert result.env == "ai"
    assert result.dry_run is True
    assert result.command[:4] == ("conda", "run", "-n", "ai")
    assert result.artifacts["glb"].endswith("mesh.glb")
    payload = json.loads(open(result.request_path, encoding="utf-8").read())
    assert payload["backend"] == "hunyuan3d"
    assert payload["model_id"] == "Hunyuan3D-2mv"


def test_build_3d_worker_command_routes_trellis_env(tmp_path):
    cmd = build_3d_worker_command(tmp_path / "request.json", "trellis", tmp_path / "result.json")

    assert cmd[:4] == ("conda", "run", "-n", "trellis")
    assert "scripts/three_d_gen_worker.py" in cmd
    assert "--result-json" in cmd
