from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

from specs.config import ModelConfig
from examples.repo_grounded_adapters.modules.embedding import build_repo_embedding

from experiments.repo_conditioned_fast_weights import (
    derive_fast_weight_hparams,
    save_npz_fast_weights,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _repo_root()
    out_dir = root / "examples" / "06_repo_fast_weights" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(d_model=384, n_heads=6, n_layers=6, d_ff=1536, vocab_size=32000, attn_impl="flash")

    emb = build_repo_embedding(str(root), dim=1536, seed=0, include_text=False)

    hparams = derive_fast_weight_hparams(
        emb["z"],
        d_model=cfg.d_model,
        num_layers=cfg.n_layers,
        capacity=16,
        seed=0,
        layer_gate="zmean",
    )

    manifest: Dict[str, Any] = {
        "repo": str(root),
        "embed_dim": int(emb["z"].shape[0]) if isinstance(emb.get("z"), np.ndarray) else 0,
        "d_model": int(cfg.d_model),
        "layers": int(cfg.n_layers),
        "capacity_default": int(hparams.get("capacity_default", 0)),
    }

    save_npz_fast_weights(str(out_dir), hparams=hparams, manifest=manifest)
    print("wrote:", str(out_dir))


if __name__ == "__main__":
    main()


