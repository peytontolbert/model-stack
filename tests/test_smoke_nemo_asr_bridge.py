from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MODULE_PATH = _REPO_ROOT / "scripts" / "smoke_nemo_asr_bridge.py"
_SPEC = importlib.util.spec_from_file_location("smoke_nemo_asr_bridge", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
smoke_nemo_asr_bridge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke_nemo_asr_bridge)


def test_restore_failure_status_marks_transformers_api_mismatch():
    exc = RuntimeError("signature mismatch in transformers.AutoModel.from_pretrained")

    status, detail = smoke_nemo_asr_bridge._restore_failure_status(exc)

    assert status == "needs_nemo_model_specific_env"
    assert "AutoModel.from_pretrained" in detail


def test_restore_failure_status_keeps_generic_nemo_failures_separate():
    exc = RuntimeError("archive restore failed while loading checkpoint")

    status, detail = smoke_nemo_asr_bridge._restore_failure_status(exc)

    assert status == "nemo_restore_failed"
    assert "archive restore failed" in detail
