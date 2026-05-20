# Artifact Registry

`registry/` provides a small file-backed artifact registry for local model,
checkpoint, dataset, and evaluation artifacts. It records artifact path,
checksum, stage, and arbitrary metadata in a JSON index.

The registry does not copy artifact bytes. It records resolved paths and verifies
them by checksum.

## API

```python
from registry import ArtifactRegistry

reg = ArtifactRegistry("artifacts/registry")
record = reg.register(
    "checkpoints/run-42/model.safetensors",
    name="run-42",
    stage="candidate",
    metadata={"task": "language-modeling"},
)

assert reg.verify(record.id)
reg.promote(record.id, "production")
```

## Data Model

`ArtifactRecord` fields:

| Field | Meaning |
| --- | --- |
| `id` | Stable ID derived from artifact name and checksum. |
| `name` | Human-readable artifact family or run name. |
| `path` | Resolved artifact path. |
| `checksum` | SHA-256 checksum of the artifact file. |
| `stage` | Lifecycle stage such as `pending`, `candidate`, or `production`. |
| `metadata` | Free-form JSON metadata. |

The index is stored at:

```text
<registry-root>/index.json
```

## Operations

- `register(path, name, metadata=None, stage="pending")`
- `list(stage=None, name=None)`
- `get(art_id)`
- `promote(art_id, stage)`
- `verify(art_id)`
- `retain_last_n(name, stage, n)`

## Governance Integration

Checksum calculation uses `governance.signature.sha256_file`. Governance
artifacts such as model cards, SBOMs, receipts, checksums, and lineage graphs
can be generated separately under `governance/` and then registered here as
artifacts.

## Limits

- The registry is local and file-backed; it does not provide locking for
  concurrent writers.
- Artifact bytes are not copied or garbage-collected by registry operations.
- `retain_last_n` prunes index entries only; it does not delete files.
