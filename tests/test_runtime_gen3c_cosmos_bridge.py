from __future__ import annotations

import json
import zipfile

from runtime.gen3c_cosmos_bridge import gen3c_cosmos_status


def test_gen3c_cosmos_status_validates_repo_checkpoint_without_loading_tensors(tmp_path):
    model_dir = tmp_path / "GEN3C-Cosmos-7B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"input_types": ["Cosmos_GEN3C"], "model_size": "7b"}), encoding="utf-8")
    (model_dir / "README.md").write_text("GEN3C", encoding="utf-8")
    with zipfile.ZipFile(model_dir / "model.pt", "w") as archive:
        archive.writestr("iter_000012900_ema_model/data.pkl", b"metadata")
        archive.writestr("iter_000012900_ema_model/data/0", b"tensor")

    status = gen3c_cosmos_status(model_dir)

    assert status.status == "needs_gen3c_cosmos_predict1_runtime"
    assert status.runnable is False
    assert status.preferred_env == "gen3c_cosmos_predict1_or_custom_bridge"
    assert status.checkpoint_format == "pytorch_zip"
    assert status.checkpoint_prefix == "iter_000012900_ema_model"
    assert status.checkpoint_entries == 2
    assert status.checkpoint_storage_entries == 1
    assert any("Do not torch.load" in blocker for blocker in status.blockers)


def test_gen3c_cosmos_status_reports_missing_assets(tmp_path):
    model_dir = tmp_path / "GEN3C-Cosmos-7B"
    model_dir.mkdir()

    status = gen3c_cosmos_status(model_dir)

    assert status.status == "incomplete_gen3c_cosmos_snapshot"
    assert status.runnable is False
    assert "model.pt" in status.missing_artifacts
