# Model Dependency Diagnostics

Use `scripts/model_dependency_diagnostics.py` when a model is local but an env may not have the right dependency versions. This is more specific than the first-wave verifier: it checks bridge-level minimums, model-declared versions, local custom Python imports, README install hints, and installed package versions in the current conda env.

The script does not load full weights. It is safe for large models.

## Commands

Cross-model conflict report:

```bash
conda run -n ai env PYTHONPATH=. python scripts/model_dependency_conflicts.py MOSS-SoundEffect-v2.0 black-forest-labs--FLUX.2-dev Wan-AI--Wan2.2-S2V-14B parakeet-rnnt-0.6b --env-name ai --markdown-out reports/dependency-conflicts/diffusion-wan-nemo.ai.md --json-out reports/dependency-conflicts/diffusion-wan-nemo.ai.json
```

This report groups package requirements across models, shows installed versus required versions, and points back to the source that introduced the requirement. For imports discovered in local model Python files, it includes `file:line:scope`, where scope is the enclosing module/function/class.

Diffusers/audio diffusion example:

```bash
conda run -n ai env PYTHONPATH=. python scripts/model_dependency_diagnostics.py MOSS-SoundEffect-v2.0 --env-name ai --markdown-out reports/dependency-diagnostics/MOSS-SoundEffect-v2.0.ai.md --json-out reports/dependency-diagnostics/MOSS-SoundEffect-v2.0.ai.json
```

NeMo ASR env comparison:

```bash
conda run -n ai env PYTHONPATH=. python scripts/model_dependency_diagnostics.py parakeet-rnnt-0.6b --env-name ai --markdown-out reports/dependency-diagnostics/parakeet-rnnt-0.6b.ai.md --json-out reports/dependency-diagnostics/parakeet-rnnt-0.6b.ai.json
conda run -n py311build env PYTHONPATH=. python scripts/model_dependency_diagnostics.py parakeet-rnnt-0.6b --env-name py311build --markdown-out reports/dependency-diagnostics/parakeet-rnnt-0.6b.py311build.md --json-out reports/dependency-diagnostics/parakeet-rnnt-0.6b.py311build.json
conda run -n trellis env PYTHONPATH=. python scripts/model_dependency_diagnostics.py parakeet-rnnt-0.6b --env-name trellis --markdown-out reports/dependency-diagnostics/parakeet-rnnt-0.6b.trellis.md --json-out reports/dependency-diagnostics/parakeet-rnnt-0.6b.trellis.json
```

Custom-layout video model example:

```bash
conda run -n ai env PYTHONPATH=. python scripts/model_dependency_diagnostics.py Wan-AI--Wan2.2-S2V-14B --env-name ai --markdown-out reports/dependency-diagnostics/Wan-AI--Wan2.2-S2V-14B.ai.md --json-out reports/dependency-diagnostics/Wan-AI--Wan2.2-S2V-14B.ai.json
```

## Status Meaning

| Status | Meaning |
| --- | --- |
| `ok` | Installed version satisfies a declared or bridge-level minimum. |
| `blocked` | Package is missing or installed version is below the required version. |
| `review` | The script found a requirement/import but cannot safely compare it as a normal version. |

## Current Useful Examples

| Report | What it tells us |
| --- | --- |
| `reports/dependency-diagnostics/MOSS-SoundEffect-v2.0.ai.md` | `ai` satisfies the coarse Diffusers, Torch, Transformers, Accelerate, and Safetensors requirements for the local MOSS snapshot. |
| `reports/dependency-diagnostics/parakeet-rnnt-0.6b.ai.md` | `ai` is blocked for NeMo ASR by Python 3.11 and missing `nemo_toolkit`; it also has Transformers below the model card hint. |
| `reports/dependency-diagnostics/parakeet-rnnt-0.6b.py311build.md` | `py311build` has newer Torch but is still blocked by Python 3.11 and missing `nemo_toolkit`. |
| `reports/dependency-diagnostics/parakeet-rnnt-0.6b.trellis.md` | `trellis` is blocked by Python 3.10, Torch below the NeMo Speech target, and missing `nemo_toolkit`. |
| `reports/dependency-diagnostics/Wan-AI--Wan2.2-S2V-14B.ai.md` | The model is local but custom-layout, so dependency checks alone are not enough; it needs a Wan-specific bridge/env path. |
| `reports/dependency-conflicts/diffusion-wan-nemo.ai.md` | Cross-model comparison showing that `ai` satisfies MOSS/FLUX/Wan coarse packages, while Parakeet is blocked by Python, missing NeMo, and a higher declared Transformers requirement. |

## How To Use This While Integrating

1. Run the first-wave verifier to classify local layout: `scripts/verify_model_stack_models.py`.
2. For any model that fails or needs env work, run this dependency diagnostic in the candidate envs.
3. Prefer fixing the env only when the diagnostic shows package/version blockers.
4. Prefer writing a custom bridge when layout is `custom_layout` even if packages are present.
5. Add a generated report for every model/env decision we rely on.
