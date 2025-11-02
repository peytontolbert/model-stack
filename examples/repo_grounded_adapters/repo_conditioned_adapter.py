import os
import json
import math
import argparse
import re
import subprocess
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from datetime import datetime
import hashlib

# Defaults and seeds for feature hashing (restored after move)
EMBED_DIM_DEFAULT: int = 1536
# Offsets for different feature groups (sym, doc, mod, top, text)
HASH_SEEDS: Tuple[int, int, int, int, int] = (0, 17, 29, 41, 53)

try:
    from examples.repo_grounded_adapters.code_graph import CodeGraph
except Exception:
    # Fallback for repo layout where the package directory has a hyphen
    import importlib.util
    alt_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "examples", "repo_grounded_adapters", "code_graph.py")
    )
    spec = importlib.util.spec_from_file_location("examples.repo_grounded_adapters.code_graph", alt_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    CodeGraph = getattr(mod, "CodeGraph")

# Optional tensor utils for contraction planning (torch backend)
try:
    from examples.repo_grounded_adapters.tensor_utils import contract as _contract
except Exception:
    _contract = None  # type: ignore
    from tensor.tensor_utils import contract as _contract


def _stable_hash(text: str, seed: int = 0) -> int:
    """64-bit FNV-1a; returns positive 63-bit for stable modulo ops."""
    mask64 = (1 << 64) - 1
    h = (1469598103934665603 ^ (seed & mask64)) & mask64
    fnv_prime = 1099511628211
    for ch in text.encode("utf-8", errors="ignore"):
        h ^= ch
        h = (h * fnv_prime) & mask64
    return h & ((1 << 63) - 1)


def _feature_hash(values: List[Tuple[str, float]], dim: int, seed: int) -> np.ndarray:
    """Vectorized feature hashing with stable sign seed (≈10× faster)."""
    if not values:
        return np.zeros((dim,), dtype=np.float32)
    keys = [k for (k, _w) in values]
    weights = np.array([float(w) for (_k, w) in values], dtype=np.float32)
    idx = np.fromiter(((
        _stable_hash(k, seed=seed) % dim
    ) for k in keys), dtype=np.int64)
    # use seed+1 for sign; lowest bit decides sign
    signs = np.fromiter(((
        1.0 if ((_stable_hash(k + "#", seed=seed + 1) & 1) == 0) else -1.0
    ) for k in keys), dtype=np.float32)
    vec = np.zeros((dim,), dtype=np.float32)
    np.add.at(vec, idx, signs * weights)
    nrm = float(np.linalg.norm(vec))
    return (vec / nrm) if nrm > 0 else vec


def build_repo_embedding(
    repo_root: str,
    *,
    dim: int = EMBED_DIM_DEFAULT,
    seed: int = 0,
    include_text: bool = False,
    text_max_bytes: int = 0,
    max_text_tokens: int = 0,
    text_weight: float = 0.25,
    graph_prop_hops: int = 0,
    graph_prop_damp: float = 0.85,
    ignore: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    g = CodeGraph.load_or_build(repo_root, ignore=ignore)

    # Collect lightweight features
    sym_feats: List[Tuple[str, float]] = []
    doc_feats: List[Tuple[str, float]] = []
    mod_feats: List[Tuple[str, float]] = []
    test_counts: Dict[str, int] = {}

    for mod, nodes in g.pytest_nodes_by_module.items():
        test_counts[mod] = len(nodes or [])

    for fqn, s in g.symbols_by_fqn.items():
        base = f"{s.kind}:{s.name}"
        sym_feats.append((base, 1.0))
        if s.signature:
            sym_feats.append((f"sig:{s.signature}", 0.5))
        if s.returns:
            sym_feats.append((f"ret:{s.returns}", 0.5))
        if s.doc:
            # downweight docs by length
            d = (s.doc or "").strip()
            if d:
                head = d.splitlines()[0][:160]
                doc_feats.append((f"doc:{head}", 0.25))
        mod_feats.append((f"mod:{s.module}", 0.2))

    # Module import topology (indegree/outdegree proxies) with optional graph propagation
    indeg: Dict[str, int] = {}
    outdeg: Dict[str, int] = {m: len(deps) for m, deps in g.module_imports.items()}
    modules = list(g.modules.keys())
    mod_idx: Dict[str, int] = {m: i for i, m in enumerate(modules)}
    for m, deps in g.module_imports.items():
        for d in deps:
            indeg[d] = indeg.get(d, 0) + 1
    base_vec = np.zeros((len(modules),), dtype=np.float32)
    for m in modules:
        i = mod_idx[m]
        base_vec[i] = float(indeg.get(m, 0) + outdeg.get(m, 0) + test_counts.get(m, 0))
    prop_vec = base_vec.copy()
    if int(graph_prop_hops) > 0:
        # Build sparse adjacency (imports treated as undirected for smoothing)
        neigh: List[List[int]] = [[] for _ in modules]
        for m, deps in g.module_imports.items():
            i = mod_idx[m]
            for d in deps:
                if d in mod_idx:
                    j = mod_idx[d]
                    neigh[i].append(j)
                    neigh[j].append(i)
        vec = prop_vec
        damp = float(graph_prop_damp)
        for _ in range(max(0, int(graph_prop_hops))):
            nxt = np.zeros_like(vec)
            for i, ns in enumerate(neigh):
                if not ns:
                    continue
                s = 0.0
                for j in ns:
                    s += float(vec[j])
                nxt[i] = (1.0 - damp) * float(base_vec[i]) + damp * (s / float(len(ns)))
            vec = nxt
        prop_vec = vec
        # normalize
        nrm = float(np.linalg.norm(prop_vec))
        if nrm > 0:
            prop_vec = prop_vec / nrm
    topo_feats: List[Tuple[str, float]] = []
    for m in modules:
        topo_feats.append((f"indeg:{m}", float(indeg.get(m, 0))))
        topo_feats.append((f"outdeg:{m}", float(outdeg.get(m, 0))))
        if test_counts.get(m):
            topo_feats.append((f"tests:{m}", float(test_counts[m])))
        if int(graph_prop_hops) > 0:
            topo_feats.append((f"prop:{m}", float(prop_vec[mod_idx[m]])))

    z_sym = _feature_hash(sym_feats, dim, seed + HASH_SEEDS[0])
    z_doc = _feature_hash(doc_feats, dim, seed + HASH_SEEDS[1])
    z_mod = _feature_hash(mod_feats, dim, seed + HASH_SEEDS[2])
    z_top = _feature_hash(topo_feats, dim, seed + HASH_SEEDS[3])

    # Optional: include raw repository text/code hashed into the embedding
    z_text = np.zeros((dim,), dtype=np.float32)
    if include_text and text_max_bytes and text_max_bytes > 0:
        text_feats: List[Tuple[str, float]] = []
        # Prefer CodeGraph's indexed files if available; otherwise walk the repo
        files: List[str] = []
        try:
            files = list(getattr(g, "indexed_files", []) or [])
        except Exception:
            files = []
        if not files:
            # Fallback: collect common source files
            exts = {".py", ".md", ".rst", ".txt", ".json", ".toml", ".yaml", ".yml", ".ini"}
            # Normalize ignore list
            ignore_list = [os.path.normpath(p) for p in (ignore or [])]
            def _is_ignored(rel: str) -> bool:
                r = os.path.normpath(rel)
                for pat in ignore_list:
                    if r == pat or r.startswith(pat + os.sep):
                        return True
                return False
            for root, dirs, fnames in os.walk(repo_root):
                rel_root = os.path.relpath(root, repo_root)
                if _is_ignored(rel_root):
                    dirs[:] = []
                    continue
                # prune ignored subdirs
                dirs[:] = [d for d in dirs if not _is_ignored(os.path.join(rel_root, d))]
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in exts:
                        fp = os.path.join(root, fn)
                        rel_fp = os.path.relpath(fp, repo_root)
                        if _is_ignored(rel_fp):
                            continue
                        files.append(fp)
        # Normalize file paths
        norm_files: List[str] = []
        for f in files:
            norm_files.append(f if os.path.isabs(f) else os.path.join(repo_root, f))

        # bytes_budget: None means unlimited; token_cap 0 means unlimited
        bytes_budget: Optional[int] = int(text_max_bytes) if int(text_max_bytes) > 0 else None
        token_cap = int(max(0, int(max_text_tokens))) if max_text_tokens is not None else 0
        tokens_emitted = 0
        for fpath in norm_files:
            if bytes_budget is not None and bytes_budget <= 0:
                break
            try:
                with open(fpath, "rb") as fh:
                    if bytes_budget is None:
                        raw = fh.read()
                    else:
                        raw = fh.read(min(bytes_budget, int(text_max_bytes)))
                # Skip binary-like files
                if b"\x00" in raw:
                    continue
                if bytes_budget is not None:
                    bytes_budget -= len(raw)
                # Decode and normalize
                text = raw.decode("utf-8", errors="ignore").lower()
                # Lightweight tokenization: words and identifiers
                tokens = re.findall(r"[a-zA-Z0-9_]+", text)
                if not tokens:
                    continue
                # Build 3-grams with stride 2 for coverage vs. cost
                n = 3
                stride = 2
                for i in range(0, max(0, len(tokens) - n + 1), stride):
                    ngram = " ".join(tokens[i : i + n])
                    text_feats.append((f"text:{ngram}", float(text_weight)))
                    tokens_emitted += 1
                    if token_cap and tokens_emitted >= token_cap:
                        break
                if token_cap and tokens_emitted >= token_cap:
                    break
            except Exception:
                continue

        if text_feats:
            z_text = _feature_hash(text_feats, dim, seed + HASH_SEEDS[4])

    z = (z_sym + z_doc + z_mod + z_top + z_text)
    norm = float(np.linalg.norm(z))
    if norm > 0:
        z = z / norm

    result = {
        "z": z.astype(np.float32),
        "z_sym": z_sym.astype(np.float32),
        "z_doc": z_doc.astype(np.float32),
        "z_mod": z_mod.astype(np.float32),
        "z_top": z_top.astype(np.float32),
    }
    if include_text:
        result["z_text"] = z_text.astype(np.float32)
    return result


def build_subgraph_embedding_from_graph(
    g: Any,
    *,
    dim: int = EMBED_DIM_DEFAULT,
    seed: int = 0,
    include_modules: Optional[List[str]] = None,
    include_files: Optional[List[str]] = None,
    include_text: bool = False,
    text_max_bytes: int = 0,
    max_text_tokens: int = 0,
    text_weight: float = 0.25,
    graph_prop_hops: int = 0,
    graph_prop_damp: float = 0.85,
) -> Dict[str, np.ndarray]:
    """Build an embedding like build_repo_embedding but restricted to a subgraph.

    Args:
        g: A CodeGraph instance already built for the repository.
        include_modules: Only include symbols and topology from these modules.
        include_files: Only include text features from these files.
    """
    mods_set = set(include_modules or [])
    files_set = set(include_files or [])

    sym_feats: List[Tuple[str, float]] = []
    doc_feats: List[Tuple[str, float]] = []
    mod_feats: List[Tuple[str, float]] = []
    test_counts: Dict[str, int] = {}

    for mod, nodes in getattr(g, "pytest_nodes_by_module", {}).items():
        if mods_set and mod not in mods_set:
            continue
        test_counts[mod] = len(nodes or [])

    for fqn, s in getattr(g, "symbols_by_fqn", {}).items():
        if mods_set and s.module not in mods_set:
            continue
        base = f"{s.kind}:{s.name}"
        sym_feats.append((base, 1.0))
        if s.signature:
            sym_feats.append((f"sig:{s.signature}", 0.5))
        if s.returns:
            sym_feats.append((f"ret:{s.returns}", 0.5))
        if s.doc:
            d = (s.doc or "").strip()
            if d:
                head = d.splitlines()[0][:160]
                doc_feats.append((f"doc:{head}", 0.25))
        mod_feats.append((f"mod:{s.module}", 0.2))

    # Topology restricted to selected modules with optional propagation
    indeg: Dict[str, int] = {}
    outdeg: Dict[str, int] = {}
    for m, deps in getattr(g, "module_imports", {}).items():
        if mods_set and m not in mods_set:
            continue
        outdeg[m] = len([d for d in deps if (not mods_set) or (d in mods_set)])
        for d in deps:
            if mods_set and d not in mods_set:
                continue
            indeg[d] = indeg.get(d, 0) + 1
    topo_feats: List[Tuple[str, float]] = []
    itermods = list(mods_set) if mods_set else list(getattr(g, "modules", {}).keys())
    mod_idx: Dict[str, int] = {m: i for i, m in enumerate(itermods)}
    base_vec = np.zeros((len(itermods),), dtype=np.float32)
    for m in itermods:
        base_vec[mod_idx[m]] = float(indeg.get(m, 0) + outdeg.get(m, 0))
    prop_vec = base_vec.copy()
    if int(graph_prop_hops) > 0:
        neigh: List[List[int]] = [[] for _ in itermods]
        for m, deps in getattr(g, "module_imports", {}).items():
            if mods_set and m not in mods_set:
                continue
            i = mod_idx[m]
            for d in deps:
                if (not mods_set) or (d in mods_set):
                    j = mod_idx.get(d)
                    if j is None:
                        continue
                    neigh[i].append(j)
                    neigh[j].append(i)
        vec = prop_vec
        damp = float(graph_prop_damp)
        for _ in range(max(0, int(graph_prop_hops))):
            nxt = np.zeros_like(vec)
            for i, ns in enumerate(neigh):
                if not ns:
                    continue
                s = 0.0
                for j in ns:
                    s += float(vec[j])
                nxt[i] = (1.0 - damp) * float(base_vec[i]) + damp * (s / float(len(ns)))
            vec = nxt
        prop_vec = vec
        nrm = float(np.linalg.norm(prop_vec))
        if nrm > 0:
            prop_vec = prop_vec / nrm
    for m in itermods:
        topo_feats.append((f"indeg:{m}", float(indeg.get(m, 0))))
        topo_feats.append((f"outdeg:{m}", float(outdeg.get(m, 0))))
        if test_counts.get(m):
            topo_feats.append((f"tests:{m}", float(test_counts[m])))
        if int(graph_prop_hops) > 0:
            topo_feats.append((f"prop:{m}", float(prop_vec[mod_idx[m]])))

    z_sym = _feature_hash(sym_feats, dim, seed + HASH_SEEDS[0])
    z_doc = _feature_hash(doc_feats, dim, seed + HASH_SEEDS[1])
    z_mod = _feature_hash(mod_feats, dim, seed + HASH_SEEDS[2])
    z_top = _feature_hash(topo_feats, dim, seed + HASH_SEEDS[3])

    z_text = np.zeros((dim,), dtype=np.float32)
    if include_text and text_max_bytes and text_max_bytes > 0:
        text_feats: List[Tuple[str, float]] = []
        # Text features only from included files if provided; else none
        files: List[str] = []
        if files_set:
            files = [f for f in getattr(g, "indexed_files", []) if f in files_set or os.path.relpath(f, getattr(g, "root", ".")) in files_set]
        bytes_budget: Optional[int] = int(text_max_bytes) if int(text_max_bytes) > 0 else None
        token_cap = int(max(0, int(max_text_tokens))) if max_text_tokens is not None else 0
        tokens_emitted = 0
        for fpath in files:
            if bytes_budget is not None and bytes_budget <= 0:
                break
            try:
                with open(fpath, "rb") as fh:
                    if bytes_budget is None:
                        raw = fh.read()
                    else:
                        raw = fh.read(min(bytes_budget, int(text_max_bytes)))
                if b"\x00" in raw:
                    continue
                if bytes_budget is not None:
                    bytes_budget -= len(raw)
                text = raw.decode("utf-8", errors="ignore").lower()
                toks = re.findall(r"[a-zA-Z0-9_]+", text)
                if not toks:
                    continue
                n = 3
                stride = 2
                for i in range(0, max(0, len(toks) - n + 1), stride):
                    ngram = " ".join(toks[i : i + n])
                    text_feats.append((f"text:{ngram}", float(text_weight)))
                    tokens_emitted += 1
                    if token_cap and tokens_emitted >= token_cap:
                        break
                if token_cap and tokens_emitted >= token_cap:
                    break
            except Exception:
                continue
        if text_feats:
            z_text = _feature_hash(text_feats, dim, seed + HASH_SEEDS[4])

    z = (z_sym + z_doc + z_mod + z_top + z_text)
    norm = float(np.linalg.norm(z))
    if norm > 0:
        z = z / norm

    result = {
        "z": z.astype(np.float32),
        "z_sym": z_sym.astype(np.float32),
        "z_doc": z_doc.astype(np.float32),
        "z_mod": z_mod.astype(np.float32),
        "z_top": z_top.astype(np.float32),
    }
    if include_text:
        result["z_text"] = z_text.astype(np.float32)
    return result


def _make_random_matrix(shape: Tuple[int, int], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Xavier uniform
    limit = math.sqrt(6.0 / float(shape[0] + shape[1]))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def generate_lora_from_embedding(
    z: np.ndarray,
    *,
    d_model: int,
    num_layers: int,
    rank: int = 8,
    seed: int = 0,
    targets: Optional[List[str]] = None,
    target_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
    layer_gate: str = "zmean",
    target_weights: Optional[Dict[str, float]] = None,
    learn_bias: bool = False,
) -> Dict[str, List[Dict[str, Dict[str, np.ndarray]]]]:
    if targets is None:
        targets = ["q_proj", "o_proj", "up_proj"]

    z = z.astype(np.float32)
    gates: List[float] = []
    layers: List[Dict[str, Dict[str, np.ndarray]]] = []

    for layer_idx in range(num_layers):
        layer_state: Dict[str, Dict[str, np.ndarray]] = {}
        # derive per-layer seeds from z by hashing a projection
        key = int((_stable_hash(f"layer:{layer_idx}", seed) ^ _stable_hash(str(float(z[0])), seed + 7)) & ((1 << 31) - 1))
        # gate schedule
        frac = float(layer_idx) / float(max(1, num_layers - 1))
        if layer_gate == "cosine":
            gate = float(0.5 * (1.0 - math.cos(math.pi * frac)))
        elif layer_gate == "hump":
            gate = float(max(0.0, math.sin(math.pi * frac)))
        elif layer_gate == "linear":
            gate = float(frac)
        else:  # zmean
            gate = float((np.tanh(z[(layer_idx * 13) % len(z)]) + 1.0) * 0.5)
        gates.append(gate)
        # Pair MLP projections: reuse seed and gate across up/gate/down; up/gate share A/B
        mlp_seed = key ^ _stable_hash("mlp_pair", seed)
        up_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None
        for tgt in targets:
            # A: d_out x r ; B: r x d_in
            if target_shapes and tgt in target_shapes:
                d_out, d_in = target_shapes[tgt]
            else:
                d_out, d_in = d_model, d_model
            # Coupled seeds for MLP blocks to align up/gate/down
            if tgt in ("up_proj", "gate_proj"):
                if up_pair is None:
                    A = _make_random_matrix((int(d_out), rank), seed=mlp_seed ^ _stable_hash("up:A", seed))
                    B = _make_random_matrix((rank, int(d_in)), seed=mlp_seed ^ _stable_hash("up:B", seed + 1))
                    up_pair = (A, B)
                else:
                    A, B = up_pair
            elif tgt == "down_proj":
                A = _make_random_matrix((int(d_out), rank), seed=mlp_seed ^ _stable_hash("down:A", seed))
                B = _make_random_matrix((rank, int(d_in)), seed=mlp_seed ^ _stable_hash("down:B", seed + 1))
            else:
                A = _make_random_matrix((int(d_out), rank), seed=key ^ _stable_hash(tgt + ":A", seed))
                B = _make_random_matrix((rank, int(d_in)), seed=key ^ _stable_hash(tgt + ":B", seed + 1))
            # Fan-in/fan-out scaling: A *= 1/sqrt(rank); optional B zeroing controlled by zero_B flag at call site via target_weights special key
            if rank > 0:
                A = (1.0 / float(max(1.0, math.sqrt(float(rank))))) * A
            # modulate by low-d projection of z (wrap-safe segment)
            start = (layer_idx * 31) % len(z)
            idx = (np.arange(32) + start) % len(z)
            seg = z[idx]
            alpha = float(np.clip(np.mean(seg) * 1.5, -1.0, 1.0))
            A = (1.0 + alpha * gate) * A
            B = (1.0 - alpha * gate) * B
            if target_weights and tgt in target_weights:
                tw = float(target_weights[tgt])
                s = float(max(0.0, tw)) ** 0.5
                A = s * A
                B = s * B
            e: Dict[str, np.ndarray] = {"A": A, "B": B, "gate": np.array([gate], dtype=np.float32)}
            if learn_bias:
                e["bias"] = np.zeros((int(d_out),), dtype=np.float32)
            layer_state[tgt] = e
        layers.append(layer_state)

    return {"layers": layers, "rank": rank, "d_model": d_model, "targets": targets, "gates": np.array(gates, dtype=np.float32)}


def generate_lora_from_embedding_torch(
    z: np.ndarray,
    *,
    d_model: int,
    num_layers: int,
    rank: int = 8,
    seed: int = 0,
    targets: Optional[List[str]] = None,
    target_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
    einsum_opt: str = "auto",
    layer_gate: str = "zmean",
    target_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, List[Dict[str, Dict[str, np.ndarray]]]]:
    import torch  # local import to avoid hard dep when unused

    if targets is None:
        targets = ["q_proj", "o_proj", "up_proj"]

    zt = torch.from_numpy(z.astype(np.float32))
    layers: List[Dict[str, Dict[str, np.ndarray]]] = []
    gates: List[float] = []
    # Global seeding for reproducibility in case external torch ops run
    try:
        torch.manual_seed(int(seed))
    except Exception:
        pass

    for layer_idx in range(num_layers):
        layer_state: Dict[str, Dict[str, np.ndarray]] = {}
        key = int((_stable_hash(f"layer:{layer_idx}", seed) ^ _stable_hash(str(float(z[0])), seed + 7)) & ((1 << 31) - 1))
        frac = float(layer_idx) / float(max(1, num_layers - 1))
        if layer_gate == "cosine":
            gate = float(0.5 * (1.0 - math.cos(math.pi * frac)))
        elif layer_gate == "hump":
            gate = float(max(0.0, math.sin(math.pi * frac)))
        elif layer_gate == "linear":
            gate = float(frac)
        else:
            gate = float((np.tanh(z[(layer_idx * 13) % len(z)]) + 1.0) * 0.5)
        gates.append(gate)
        # deterministic torch RNG
        gen = torch.Generator(device="cpu")
        gen.manual_seed(key)
        mlp_seed = key ^ _stable_hash("mlp_pair", seed)
        up_pair: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        for tgt in targets:
            if target_shapes and tgt in target_shapes:
                d_out, d_in = target_shapes[tgt]
            else:
                d_out, d_in = d_model, d_model
            if tgt in ("up_proj", "gate_proj"):
                if up_pair is None:
                    A = (torch.rand((int(d_out), rank), generator=gen) * 2 - 1).to(torch.float32)
                    B = (torch.rand((rank, int(d_in)), generator=gen) * 2 - 1).to(torch.float32)
                    up_pair = (A, B)
                else:
                    A, B = up_pair
            elif tgt == "down_proj":
                # reuse generator but distinct suffix for down
                A = (torch.rand((int(d_out), rank), generator=gen) * 2 - 1).to(torch.float32)
                B = (torch.rand((rank, int(d_in)), generator=gen) * 2 - 1).to(torch.float32)
            else:
                A = (torch.rand((int(d_out), rank), generator=gen) * 2 - 1).to(torch.float32)
                B = (torch.rand((rank, int(d_in)), generator=gen) * 2 - 1).to(torch.float32)
            if rank > 0:
                A = A * (1.0 / float(max(1.0, math.sqrt(float(rank)))))
            # Modulate by contraction with a deterministic kernel vector derived from z
            start = (layer_idx * 31) % len(z)
            idx = (np.arange(32) + start) % len(z)
            seg = zt[idx]
            w = torch.sin(torch.linspace(0, 3.14159, steps=32))
            if _contract is not None and einsum_opt:
                alpha = torch.tanh(_contract("i,i->", seg, w, optimize=einsum_opt) / 8.0)
            else:
                alpha = torch.tanh((seg * w).sum() / 8.0)
            A = (1.0 + float(alpha.item()) * gate) * A
            B = (1.0 - float(alpha.item()) * gate) * B
            if target_weights and tgt in target_weights:
                tw = float(target_weights[tgt])
                s = float(max(0.0, tw)) ** 0.5
                A = (s * A)
                B = (s * B)
            e: Dict[str, np.ndarray] = {
                "A": A.numpy().astype(np.float32),
                "B": B.numpy().astype(np.float32),
                "gate": np.array([gate], dtype=np.float32),
            }
            # Torch path does not currently support learn_bias flag; add zero bias for parity if requested via target_weights special key later if needed
            layer_state[tgt] = e
        layers.append(layer_state)

    return {"layers": layers, "rank": rank, "d_model": d_model, "targets": targets, "gates": np.array(gates, dtype=np.float32)}

def _parse_target_shapes(arg: Optional[str]) -> Optional[Dict[str, Tuple[int, int]]]:
    if not arg:
        return None
    result: Dict[str, Tuple[int, int]] = {}
    try:
        parts = [p.strip() for p in str(arg).split(",") if p.strip()]
        for p in parts:
            if "=" not in p or ":" not in p:
                continue
            name, dims = p.split("=", 1)
            a, b = dims.split(":", 1)
            result[name.strip()] = (int(a), int(b))
        return result or None
    except Exception:
        return None

def _git_commit_sha(repo_root: str) -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha if sha else None
    except Exception:
        return None


def _git_tree_sha(repo_root: str) -> Optional[str]:
    """Return the HEAD tree SHA if available (pins exact tracked file set)."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD^{tree}"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha if sha else None
    except Exception:
        return None


def _detect_target_shapes_from_model(model_id: str) -> Optional[Dict[str, Tuple[int, int]]]:
    try:
        from transformers import AutoConfig  # type: ignore

        cfg = AutoConfig.from_pretrained(model_id)
        d_model = int(getattr(cfg, "hidden_size", 0) or 0)
        inter = int(getattr(cfg, "intermediate_size", 0) or 0)
        if d_model <= 0:
            return None
        shapes: Dict[str, Tuple[int, int]] = {
            "q_proj": (d_model, d_model),
            "o_proj": (d_model, d_model),
        }
        if inter > 0:
            shapes["up_proj"] = (inter, d_model)
            # Optionally anticipate down_proj if user extends targets later
            shapes["down_proj"] = (d_model, inter)
        return shapes
    except Exception:
        return None


def _detect_target_shapes_from_model_full(model_id: str, target_regex: Optional[str] = None) -> Optional[Dict[str, Tuple[int, int]]]:
    """Load model on CPU and enumerate linear layers to infer shapes; filter by regex if provided."""
    try:
        import torch  # local
        from transformers import AutoModelForCausalLM  # type: ignore

        rx = re.compile(str(target_regex)) if target_regex else None
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )
        model.eval()
        shapes: Dict[str, Tuple[int, int]] = {}
        # Inspect first transformer block if present for canonical names; else full traversal
        root = getattr(model, "model", model)
        layer0 = None
        try:
            layer0 = root.layers[0]
        except Exception:
            layer0 = root
        for name, mod in layer0.named_modules():
            try:
                w = getattr(mod, "weight", None)
                if w is None or getattr(w, "ndim", 0) != 2:
                    continue
                short = name.split(".")[-1]
                if rx is not None:
                    if not rx.search(short):
                        continue
                else:
                    if not short.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")):
                        continue
                d_out, d_in = tuple(w.shape)
                shapes[short] = (int(d_out), int(d_in))
            except Exception:
                continue
        return shapes or None
    except Exception:
        return None


def _detect_target_names_from_model_full(model_id: str, target_regex: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Return mapping from short target name (last token) to full module path for export."""
    try:
        import torch  # local
        from transformers import AutoModelForCausalLM  # type: ignore

        rx = re.compile(str(target_regex)) if target_regex else None
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=None,
        )
        model.eval()
        names: Dict[str, str] = {}
        root = getattr(model, "model", model)
        layer0 = None
        try:
            layer0 = root.layers[0]
        except Exception:
            layer0 = root
        for name, mod in layer0.named_modules():
            try:
                w = getattr(mod, "weight", None)
                if w is None or getattr(w, "ndim", 0) != 2:
                    continue
                short = name.split(".")[-1]
                if rx is not None:
                    if not rx.search(short) and not rx.search(name):
                        continue
                else:
                    if not short.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")):
                        continue
                names[short] = name
            except Exception:
                continue
        return names or None
    except Exception:
        return None


def save_npz(out_dir: str, *, embedding: Dict[str, np.ndarray], adapters: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "embedding.npz"), **embedding)
    # flatten adapter arrays
    flat: Dict[str, np.ndarray] = {}
    for i, layer in enumerate(adapters["layers"]):
        for name, tensors in layer.items():
            flat[f"L{i}.{name}.A"] = tensors["A"]
            flat[f"L{i}.{name}.B"] = tensors["B"]
            gate_val = float(tensors["gate"][0]) if isinstance(tensors.get("gate"), np.ndarray) else float(tensors.get("gate", 0.0))
            flat[f"L{i}.{name}.gate"] = np.array(gate_val, dtype=np.float32)
            if "bias" in tensors and isinstance(tensors["bias"], np.ndarray):
                flat[f"L{i}.{name}.bias"] = tensors["bias"]
    np.savez_compressed(os.path.join(out_dir, "adapters.npz"), **flat)
    open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8").write(json.dumps(manifest, indent=2))


def _save_peft_like(out_dir: str, adapters: Dict[str, Any], *, r: int, alpha: int, target_modules: List[str], bias: str = "none", int8: bool = False, target_paths: Optional[Dict[str, str]] = None) -> None:
    """Write a minimal PEFT LoRA config + tensors for quick benchmarking.

    Note: This is a best-effort exporter; users may still need to map names depending on the model arch.
    """
    try:
        import json as _json
        cfg = {
            "peft_type": "LORA",
            "r": int(r),
            "lora_alpha": int(alpha),
            "target_modules": target_modules,
            "lora_dropout": 0.0,
            "bias": str(bias),
            "task_type": "CAUSAL_LM",
        }
        open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8").write(_json.dumps(cfg, indent=2))
        # Save tensors in a stable torch format if available
        try:
            import torch as _torch  # type: ignore

            state: Dict[str, Any] = {}
            def _map_path(i: int, name: str) -> str:
                # Map target name to likely module path; best-effort for LLaMA-like arch
                if target_paths and name in target_paths:
                    return f"base_model.model.{target_paths[name]}"
                if name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    return f"base_model.model.model.layers.{i}.self_attn.{name}"
                elif name in ("up_proj", "down_proj", "gate_proj"):
                    return f"base_model.model.model.layers.{i}.mlp.{name}"
                else:
                    return f"base_model.model.model.layers.{i}.{name}"
            for i, layer in enumerate(adapters["layers"]):
                for name, tensors in layer.items():
                    base = _map_path(i, name)
                    A = _torch.from_numpy(tensors["A"]).contiguous()
                    B = _torch.from_numpy(tensors["B"]).contiguous()
                    if int8:
                        try:
                            # Per-tensor affine quantization
                            scale_A = float(A.abs().max().item() / 127.0) if A.numel() > 0 else 1.0
                            A = _torch.quantize_per_tensor(A, scale=max(scale_A, 1e-8), zero_point=0, dtype=_torch.qint8)
                            scale_B = float(B.abs().max().item() / 127.0) if B.numel() > 0 else 1.0
                            B = _torch.quantize_per_tensor(B, scale=max(scale_B, 1e-8), zero_point=0, dtype=_torch.qint8)
                        except Exception:
                            pass
                    state[f"{base}.lora_A.weight"] = A
                    state[f"{base}.lora_B.weight"] = B
            _torch.save(state, os.path.join(out_dir, "adapter_model.bin"))
        except Exception:
            pass
    except Exception:
        pass


def load_adapters_npz(path: str) -> Dict[str, List[Dict[str, Dict[str, np.ndarray]]]]:
    data = np.load(path)
    # infer indices
    layers: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    for key in data.files:
        # L{idx}.{name}.{A|B|gate}
        parts = key.split(".")
        if len(parts) != 3:
            continue
        lid = int(parts[0][1:])
        name = parts[1]
        kind = parts[2]
        layers.setdefault(lid, {}).setdefault(name, {})[kind] = data[key]
    ordered = [layers[i] for i in sorted(layers.keys())]
    return {"layers": ordered}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--embed-dim", type=int, default=1536)
    p.add_argument("--ignore", action="append", default=None, help="Relative path (repeatable) to ignore, like .gitignore; e.g., --ignore cloned_repos")
    p.add_argument("--include-text", action="store_true", help="Hash raw repo text/code into the embedding")
    p.add_argument("--text-max-bytes", type=int, default=0, help="Max total bytes of text to hash (0 disables)")
    p.add_argument("--max-text-tokens", type=int, default=0, help="Hard cap on number of hashed n-grams from repo text (0 disables)")
    p.add_argument("--text-weight", type=float, default=0.25, help="Relative weight for text features in embedding")
    p.add_argument("--d-model", type=int, required=True)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--targets", default="q_proj,o_proj,up_proj")
    p.add_argument("--target-shapes", default=None, help="CSV like q_proj=4096:4096,o_proj=4096:4096,up_proj=14336:4096")
    p.add_argument("--probe-model", default=None, help="If set, auto-detect target shapes from this HF model id")
    p.add_argument("--probe-model-full", action="store_true", help="Load HF model and infer all Linear shapes via named_modules; filter with --target-regex if provided")
    p.add_argument("--save-torch", action="store_true", help="Also save adapters.pt (requires torch)")
    p.add_argument("--peft-json", action="store_true", help="Also write a minimal PEFT-compatible config and weights")
    p.add_argument("--peft-bias", choices=["none", "lora_only"], default="none", help="Bias handling mode in PEFT config")
    p.add_argument("--peft-int8", action="store_true", help="Quantize exported LoRA weights to int8 in adapter_model.bin")
    p.add_argument("--learn-bias", action="store_true", help="Generate and export optional per-target bias vectors (initialized to 0)")
    p.add_argument("--modules", default=None, help="Comma-separated module names to restrict embedding to (subgraph)")
    p.add_argument("--files", default=None, help="Comma-separated relative file paths to include raw text from (subgraph)")
    p.add_argument("--graph-prop-hops", type=int, default=0, help="Propagation hops over module graph for topology features")
    p.add_argument("--graph-prop-damp", type=float, default=0.85, help="Damping factor for graph propagation (0..1)")
    p.add_argument("--gen-backend", choices=["numpy", "torch"], default="numpy", help="Adapter generation backend")
    p.add_argument("--einsum-opt", default="auto", help="einsum optimize flag for torch backend (auto/greedy/path)")
    p.add_argument("--target-regex", default=None, help="Regex filter to select target projection names (after probe-model if provided)")
    p.add_argument("--layer-gate", choices=["zmean", "cosine", "hump", "linear"], default="zmean", help="Per-layer gate schedule")
    p.add_argument("--target-weights", default=None, help="CSV like q_proj=1,k_proj=1,v_proj=1,o_proj=1.1,up_proj=1.1,down_proj=1.05")
    p.add_argument("--zero-b", action="store_true", help="Zero-initialize B matrices (official LoRA init); keeps A scaled by 1/sqrt(r)")
    # Provenance & dataset capture
    p.add_argument("--record-sources", action="store_true", help="Write sources.jsonl listing files used by the embedding")
    p.add_argument("--record-hashes", action="store_true", help="Include sha256 and size for each source file (slower)")
    p.add_argument("--sources-max", type=int, default=0, help="Max files to record (0 = unlimited)")
    args = p.parse_args()
    # Optional seeding for reproducibility
    try:
        if args.seed is not None:
            import random as _random
            _random.seed(int(args.seed))
            np.random.seed(int(args.seed))
    except Exception:
        pass

    # Build embedding: subgraph path if --modules/--files are provided
    modules_list = [m.strip() for m in str(args.modules).split(",") if m.strip()] if args.modules else None
    files_list = [f.strip() for f in str(args.files).split(",") if f.strip()] if args.files else None
    ignore_list = [s for s in (args.ignore or []) if s]
    g = CodeGraph.load_or_build(args.repo, ignore=ignore_list)
    if modules_list or files_list:
        emb = build_subgraph_embedding_from_graph(
            g,
            dim=args.embed_dim,
            seed=args.seed,
            include_modules=modules_list,
            include_files=files_list,
            include_text=args.include_text,
            text_max_bytes=args.text_max_bytes,
            max_text_tokens=args.max_text_tokens,
            text_weight=args.text_weight,
            graph_prop_hops=int(args.graph_prop_hops),
            graph_prop_damp=float(args.graph_prop_damp),
        )
    else:
        emb = build_repo_embedding(
            args.repo,
            dim=args.embed_dim,
            seed=args.seed,
            include_text=args.include_text,
            text_max_bytes=args.text_max_bytes,
            max_text_tokens=args.max_text_tokens,
            text_weight=args.text_weight,
            graph_prop_hops=int(args.graph_prop_hops),
            graph_prop_damp=float(args.graph_prop_damp),
            ignore=ignore_list,
        )
    target_shapes = _parse_target_shapes(args.target_shapes)
    if (not target_shapes) and args.probe_model:
        auto_shapes = None
        if args.probe_model_full:
            auto_shapes = _detect_target_shapes_from_model_full(str(args.probe_model), args.target_regex)
        if not auto_shapes:
            auto_shapes = _detect_target_shapes_from_model(str(args.probe_model))
        if auto_shapes:
            target_shapes = auto_shapes
    # Build targets list (regex-aware)
    default_targets = [t.strip() for t in (args.targets or "").split(",") if t.strip()] or ["q_proj", "o_proj", "up_proj"]
    targets_list = list(default_targets)
    # Broaden coverage when shapes are known: include k_proj/v_proj/down_proj/gate_proj if available
    if target_shapes:
        for extra in ("k_proj", "v_proj", "down_proj", "gate_proj"):
            if (extra in target_shapes) and (extra not in targets_list):
                targets_list.append(extra)
    if args.target_regex:
        try:
            rx = re.compile(str(args.target_regex))
            # Prefer probing keys if available
            keys = list((target_shapes or {}).keys()) or default_targets
            targets_list = [k for k in keys if rx.search(k)]
            if not targets_list:
                targets_list = [k for k in default_targets if rx.search(k)]
        except Exception:
            pass
    # Parse target weights
    def _parse_tw(arg: Optional[str]) -> Optional[Dict[str, float]]:
        if not arg:
            return None
        out: Dict[str, float] = {}
        try:
            parts = [p.strip() for p in str(arg).split(",") if p.strip()]
            for p in parts:
                if "=" not in p:
                    continue
                n, v = p.split("=", 1)
                out[n.strip()] = float(v)
            return out or None
        except Exception:
            return None
    tw = _parse_tw(args.target_weights)

    if args.gen_backend == "torch":
        adapters = generate_lora_from_embedding_torch(
            emb["z"],
            d_model=args.d_model,
            num_layers=args.layers,
            rank=args.rank,
            seed=args.seed,
            targets=targets_list,
            target_shapes=target_shapes,
            einsum_opt=args.einsum_opt,
            layer_gate=str(args.layer_gate),
            target_weights=tw,
        )
    else:
        adapters = generate_lora_from_embedding(
            emb["z"],
            d_model=args.d_model,
            num_layers=args.layers,
            rank=args.rank,
            seed=args.seed,
            targets=targets_list,
            target_shapes=target_shapes,
            layer_gate=str(args.layer_gate),
            target_weights=tw,
            learn_bias=bool(args.learn_bias),
        )

    # Optionally zero out B matrices post-generation to match official LoRA init
    if bool(args.zero_b):
        try:
            for layer in adapters.get("layers", []):
                for name, tensors in layer.items():
                    tensors["B"] = np.zeros_like(tensors["B"])  # type: ignore[index]
        except Exception:
            pass

    # CodeGraph stats
    try:
        modules_count = len(getattr(g, "modules", {}) or {})
    except Exception:
        modules_count = None
    try:
        symbols_count = len(getattr(g, "symbols_by_fqn", {}) or {})
    except Exception:
        symbols_count = None
    try:
        imports_edges = sum(len(v or []) for v in getattr(g, "module_imports", {}).values())
    except Exception:
        imports_edges = None
    try:
        indexed_files = list(getattr(g, "indexed_files", []) or [])
    except Exception:
        indexed_files = []

    # Selection & sources
    files_rel: List[str] = []
    if files_list:
        files_rel = [f if os.path.isabs(f) else os.path.relpath(os.path.join(getattr(g, "root", args.repo), f), getattr(g, "root", args.repo)) for f in files_list]
    elif indexed_files:
        files_rel = [os.path.relpath(f, getattr(g, "root", args.repo)) for f in indexed_files]
    # Optionally write sources.jsonl
    if args.record_sources and files_rel:
        out_sources = os.path.join(args.out, "sources.jsonl")
        os.makedirs(args.out, exist_ok=True)
        cap = max(0, int(args.sources_max))
        count = 0
        with open(out_sources, "w", encoding="utf-8") as fh:
            for rel in files_rel:
                if cap and count >= cap:
                    break
                abs_fp = os.path.join(getattr(g, "root", args.repo), rel)
                entry: Dict[str, Any] = {"path": rel}
                if args.record_hashes:
                    try:
                        b = open(abs_fp, "rb").read()
                        entry["sha256"] = hashlib.sha256(b).hexdigest()
                        entry["bytes"] = len(b)
                    except Exception:
                        entry["sha256"] = None
                fh.write(json.dumps(entry) + "\n")
                count += 1

    manifest = {
        "repo": os.path.abspath(args.repo),
        "commit": _git_commit_sha(args.repo),
        "tree": _git_tree_sha(args.repo),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "schema_version": 1,
        "embed_dim": args.embed_dim,
        "include_text": bool(args.include_text),
        "text_max_bytes": int(args.text_max_bytes),
        "max_text_tokens": int(args.max_text_tokens),
        "text_weight": float(args.text_weight),
        "graph_prop_hops": int(args.graph_prop_hops),
        "graph_prop_damp": float(args.graph_prop_damp),
        "d_model": args.d_model,
        "layers": args.layers,
        "rank": args.rank,
        "seed": args.seed,
        "targets": adapters.get("targets"),
        "target_shapes": target_shapes,
        "probe_model": (str(args.probe_model) if args.probe_model else None),
        "gen_backend": args.gen_backend,
        "layer_gate": str(args.layer_gate),
        "target_regex": str(args.target_regex) if args.target_regex else None,
        "target_weights": tw,
        "selection": {
            "modules": modules_list,
            "files": files_list,
        },
        "code_graph": {
            "modules_count": modules_count,
            "symbols_count": symbols_count,
            "imports_edges": imports_edges,
            "indexed_files_count": len(indexed_files),
        },
        "sources": {
            "count": len(files_rel),
            "sample": files_rel[: min(20, len(files_rel))],
            "recorded": bool(args.record_sources),
            "record_hashes": bool(args.record_hashes),
            "record_cap": int(args.sources_max),
        },
    }

    save_npz(args.out, embedding=emb, adapters=adapters, manifest=manifest)
    if args.peft_json:
        try:
            tm = list((adapters.get("targets") or []))
            # Try to discover model-agnostic target paths when probe-model-full is used
            target_paths = None
            if args.probe_model and args.probe_model_full:
                target_paths = _detect_target_names_from_model_full(str(args.probe_model), args.target_regex)
            _save_peft_like(
                args.out,
                adapters,
                r=int(args.rank),
                alpha=int(args.rank * 2),
                target_modules=tm,
                bias=str(args.peft_bias),
                int8=bool(args.peft_int8),
                target_paths=target_paths,
            )
        except Exception:
            pass

    if args.save_torch:
        try:
            import torch  # type: ignore

            tstate: Dict[str, torch.Tensor] = {}
            for i, layer in enumerate(adapters["layers"]):
                for name, tensors in layer.items():
                    tstate[f"L{i}.{name}.A"] = torch.from_numpy(tensors["A"])  # type: ignore
                    tstate[f"L{i}.{name}.B"] = torch.from_numpy(tensors["B"])  # type: ignore
                    tstate[f"L{i}.{name}.gate"] = torch.from_numpy(tensors["gate"])  # type: ignore
            torch.save(tstate, os.path.join(args.out, "adapters.pt"))
        except Exception:
            pass

    # Print a concise completion summary
    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": os.path.abspath(args.out),
                "files": [
                    os.path.join(os.path.abspath(args.out), "embedding.npz"),
                    os.path.join(os.path.abspath(args.out), "adapters.npz"),
                    os.path.join(os.path.abspath(args.out), "manifest.json"),
                ]
                + (
                    [os.path.join(os.path.abspath(args.out), "adapters.pt")] if args.save_torch else []
                ),
                "embed_dim": args.embed_dim,
                "z_norm": float(np.linalg.norm(emb["z"])) if isinstance(emb.get("z"), np.ndarray) else None,
                "layers": args.layers,
                "rank": args.rank,
                "targets": adapters.get("targets"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


