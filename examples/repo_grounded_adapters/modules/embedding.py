from typing import Any, Dict, List, Optional
import numpy as np

from examples.repo_grounded_adapters.code_graph import CodeGraph

EMBED_DIM_DEFAULT = 128
HASH_SEEDS = [1337, 1338, 1339, 1340, 1341]

def auto_model_dims(model_id: str, cache_dir: Optional[str]) -> Tuple[int, int]:
    try:
        from transformers import AutoConfig  # type: ignore

        cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
        d_model = int(getattr(cfg, "hidden_size", 0) or 0)
        return n_layers, d_model
    except Exception:
        return 0, 0


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

