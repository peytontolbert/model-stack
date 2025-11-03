from .embedding import build_repo_embedding, build_subgraph_embedding_from_graph
from .adapter import (
    generate_lora_from_embedding,
    generate_lora_from_embedding_torch,
    save_npz,
    load_adapters_npz,
    detect_target_shapes_from_model,
    detect_target_shapes_from_model_full,
)
from .model import ensure_snapshot, build_local_llama_from_snapshot

