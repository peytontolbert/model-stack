from __future__ import annotations

import json

from runtime.compatibility import compatibility_report, report_as_dict


def test_classic_mobilellm_reports_cache_and_tokenizer_patches(tmp_path):
    model = tmp_path / "MobileLLM-125M"
    model.mkdir()
    (model / "modeling_mobilellm.py").write_text("", encoding="utf-8")
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mobilellm",
                "auto_map": {"AutoModelForCausalLM": "modeling_mobilellm.MobileLLMForCausalLM"},
            }
        ),
        encoding="utf-8",
    )

    payload = report_as_dict(compatibility_report(model, model_id="MobileLLM-125M"))
    patch_ids = {patch["id"] for patch in payload["patches"]}

    assert "transformers_mobilellm_legacy_cache" in patch_ids
    assert "transformers_mobilellm_slow_tokenizer" in patch_ids
    assert any("use_cache=False" in patch["patch"] for patch in payload["patches"])
    assert any("LlamaTokenizer" in patch["patch"] for patch in payload["patches"])


def test_cosmos_anomaly_reports_apply_chunking_shim_candidate(tmp_path):
    model = tmp_path / "Cosmos-Embed1-448p-anomaly-detection"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "cosmos_embed1"}), encoding="utf-8")

    payload = report_as_dict(compatibility_report(model))
    patch = next(patch for patch in payload["patches"] if patch["id"] == "transformers_apply_chunking_to_forward_compat")

    assert "apply_chunking_to_forward" in patch["expected_api"]
    assert "shim" in patch["patch"]


def test_nemo_archive_reports_archive_preference_patch(tmp_path):
    model = tmp_path / "parakeet-rnnt-0.6b"
    model.mkdir()
    (model / "parakeet-rnnt-0.6b.nemo").write_bytes(b"placeholder")

    payload = report_as_dict(compatibility_report(model, model_id="parakeet-rnnt-0.6b"))
    patch = next(patch for patch in payload["patches"] if patch["id"] == "nemo_prefer_archive_over_external_transformers_metadata")

    assert "*.nemo" in patch["patch"]
    assert "Transformers 5.x" in patch["patch"]


def test_sequence_classifier_reports_num_labels_patch(tmp_path):
    model = tmp_path / "bug-localization"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertForSequenceClassification"],
                "num_labels": 1,
            }
        ),
        encoding="utf-8",
    )
    header = {
        "classifier.weight": {"dtype": "F32", "shape": [2, 768], "data_offsets": [0, 0]},
        "classifier.bias": {"dtype": "F32", "shape": [2], "data_offsets": [0, 0]},
    }
    raw = json.dumps(header).encode("utf-8")
    (model / "model.safetensors").write_bytes(len(raw).to_bytes(8, "little") + raw)

    payload = report_as_dict(compatibility_report(model, model_id="repository_library/bug-localization"))
    patch = next(patch for patch in payload["patches"] if patch["id"] == "transformers_classifier_head_num_labels_from_checkpoint")

    assert "config.num_labels=1" in patch["current_probe"]
    assert "classifier.weight[0]=2" in patch["current_probe"]
    assert "override config.num_labels" in patch["patch"]
