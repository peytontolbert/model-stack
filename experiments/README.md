# Experiments

`experiments/` contains exploratory scripts that are useful for research but are
not stable runtime APIs. Code here may depend on optional files, local research
artifacts, or experimental workflows.

Do not import from `experiments/` in production runtime paths.

## Current Contents

| File | Purpose |
| --- | --- |
| `repo_conditioned_fast_weights.py` | Derives per-layer fast-weight hyperparameters from a repository embedding and can save NPZ/PT outputs plus a manifest. |
| `indistributionqa.txt` | Research note/data artifact for in-distribution QA experiments. |

## Fast-Weight Conditioning Script

`repo_conditioned_fast_weights.py` builds or reuses repository/subgraph
embeddings, derives conservative Hebbian fast-weight parameters, and writes:

```text
fast_weights.npz
manifest.json
fast_weights.pt  # optional, when --save-torch is used
```

Typical invocation shape:

```bash
python experiments/repo_conditioned_fast_weights.py \
  --repo /path/to/repo \
  --out artifacts/fast_weights \
  --d-model 4096 \
  --layers 32
```

The script can condition on selected modules or files through `--modules` and
`--files`, and can optionally include text-derived features through
`--include-text`.

## Rules

- Keep experiment outputs under `artifacts/` or an explicit output directory.
- Do not make runtime packages depend on experiment modules.
- Promote reusable code into a package before using it in training, serving, or
  export paths.
- Document new scripts here with their inputs, outputs, and stability level.

