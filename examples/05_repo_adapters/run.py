from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

from specs.config import ModelConfig
from examples.repo_grounded_adapters.modules.adapter import save_npz
from examples.repo_grounded_adapters.modules.embedding import build_repo_embedding, generate_lora_from_embedding

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = _repo_root()
    out_dir = root / "examples" / "05_repo_adapters" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use a small model shape for a quick artifact; align with repo's tiny configs
    cfg = ModelConfig(d_model=384, n_heads=6, n_layers=6, d_ff=1536, vocab_size=32000, attn_impl="flash")

    # Build a repository embedding for this repository
    emb = build_repo_embedding(str(root), dim=1536, seed=0, include_text=False)

    # Generate lightweight LoRA-style adapters from the embedding
    adapters = generate_lora_from_embedding(
        emb["z"],
        d_model=cfg.d_model,
        num_layers=cfg.n_layers,
        rank=8,
        seed=0,
        targets=["q_proj", "o_proj", "up_proj", "down_proj"],
        target_shapes=None,
        layer_gate="zmean",
        target_weights=None,
        learn_bias=False,
    )

    manifest: Dict[str, Any] = {
        "repo": str(root),
        "embed_dim": int(emb["z"].shape[0]) if isinstance(emb.get("z"), np.ndarray) else 0,
        "d_model": int(cfg.d_model),
        "layers": int(cfg.n_layers),
        "rank": int(adapters.get("rank", 0)),
        "targets": adapters.get("targets"),
    }

    save_npz(str(out_dir), embedding=emb, adapters=adapters, manifest=manifest)
    print("wrote:", str(out_dir))


if __name__ == "__main__":
    main()


