from typing import Any, Dict, List, Optional, Tuple
import os
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import pathspec  # type: ignore
except Exception:  # pragma: no cover
    pathspec = None  # type: ignore

from examples.repo_grounded_adapters.code_graph import CodeGraph

EMBED_DIM_DEFAULT = 128
HASH_SEEDS = [1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344]

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


def _accumulate_sparse(acc: Dict[int, float], key: str, weight: float, dim: int, seed: int) -> None:
    idx = int(_stable_hash(key, seed=seed) % dim)
    sign = 1.0 if ((_stable_hash(key + "#", seed=seed + 1) & 1) == 0) else -1.0
    acc[idx] = float(acc.get(idx, 0.0)) + float(sign * weight)


def _dense_from_sparse(acc: Dict[int, float], dim: int) -> np.ndarray:
    if not acc:
        return np.zeros((dim,), dtype=np.float32)
    vec = np.zeros((dim,), dtype=np.float32)
    for i, v in acc.items():
        if 0 <= int(i) < dim:
            vec[int(i)] = vec[int(i)] + float(v)
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
    calls_weight: float = 0.25,
    types_weight: float = 0.20,
    tests_weight: float = 0.15,
    graph_prop_hops: int = 0,
    graph_prop_damp: float = 0.85,
    ignore: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    g = CodeGraph.load_or_build(repo_root, ignore=ignore)

    # Collect lightweight features
    sym_feats: List[Tuple[str, float]] = []
    doc_feats: List[Tuple[str, float]] = []
    mod_feats: List[Tuple[str, float]] = []
    type_feats: List[Tuple[str, float]] = []
    test_counts: Dict[str, int] = {}

    for mod, nodes in g.pytest_nodes_by_module.items():
        test_counts[mod] = len(nodes or [])

    for fqn, s in g.symbols_by_fqn.items():
        base = f"{s.kind}:{s.name}"
        sym_feats.append((base, 1.0))
        if s.signature:
            # keep in sym for backward-compat
            sym_feats.append((f"sig:{s.signature}", 0.5))
            type_feats.append((f"sig:{s.signature}", 1.0))
        if s.returns:
            sym_feats.append((f"ret:{s.returns}", 0.5))
            type_feats.append((f"ret:{s.returns}", 1.0))
        if s.doc:
            # downweight docs by length
            d = (s.doc or "").strip()
            if d:
                head = d.splitlines()[0][:160]
                doc_feats.append((f"doc:{head}", 0.25))
        mod_feats.append((f"mod:{s.module}", 0.2))

    # Call-graph features (multi-view)
    call_feats: List[Tuple[str, float]] = []
    try:
        for caller, callee in g.calls:
            call_feats.append((f"call:{caller}->{callee}", 1.0))
    except Exception:
        call_feats = []

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
    z_types = _feature_hash(type_feats, dim, seed + HASH_SEEDS[5])
    z_calls = _feature_hash(call_feats, dim, seed + HASH_SEEDS[6])

    # Optional: include raw repository text/code hashed into the embedding
    z_text = np.zeros((dim,), dtype=np.float32)
    if include_text and text_max_bytes and text_max_bytes > 0:
        # Streamed sparse accumulator for large repos
        text_acc: Dict[int, float] = {}
        # Prefer CodeGraph's indexed files if available; otherwise walk the repo
        files: List[str] = []
        try:
            files = list(getattr(g, "indexed_files", []) or [])
        except Exception:
            files = []
        if not files:
            # Fallback: collect common source files
            exts = {".py", ".md", ".rst", ".txt", ".json", ".toml", ".yaml", ".yml", ".ini"}
            # Normalize ignore list + .gitignore pathspec
            ignore_list = [os.path.normpath(p) for p in (ignore or [])]
            pspec = None
            try:
                gi = os.path.join(repo_root, ".gitignore")
                if pathspec is not None and os.path.exists(gi):
                    with open(gi, "r", encoding="utf-8", errors="ignore") as fh:
                        lines = [ln.rstrip("\n") for ln in fh]
                    pspec = pathspec.PathSpec.from_lines("gitwildmatch", lines)
            except Exception:
                pspec = None
            def _is_ignored(rel: str) -> bool:
                r = os.path.normpath(rel)
                if pspec is not None:
                    if pspec.match_file(r.replace(os.sep, "/")):
                        return True
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
        # Normalize file paths and pre-filter by extension
        exts_all = {".py", ".md", ".rst", ".txt", ".json", ".toml", ".yaml", ".yml", ".ini"}
        norm_files: List[str] = []
        for f in files:
            p = f if os.path.isabs(f) else os.path.join(repo_root, f)
            if os.path.splitext(p)[1].lower() in exts_all:
                norm_files.append(p)

        # bytes_budget: None means unlimited; token_cap 0 means unlimited
        bytes_budget: Optional[int] = int(text_max_bytes) if int(text_max_bytes) > 0 else None
        token_cap = int(max(0, int(max_text_tokens))) if max_text_tokens is not None else 0
        tokens_emitted = 0

        def _process_file(fp: str) -> Tuple[int, int, Dict[int, float]]:
            try:
                # Per-file cap: avoid reading giant files fully when many are available
                per_cap = int(text_max_bytes)
                with open(fp, "rb") as fh:
                    raw = fh.read(per_cap)
                if b"\x00" in raw:
                    return (0, 0, {})
                n_bytes = len(raw)
                text = raw.decode("utf-8", errors="ignore").lower()
                toks = re.findall(r"[a-zA-Z0-9_]+", text)
                if not toks:
                    return (n_bytes, 0, {})
                n = 3
                stride = 2
                acc_local: Dict[int, float] = {}
                ng_count = 0
                for i in range(0, max(0, len(toks) - n + 1), stride):
                    key = f"text:{' '.join(toks[i:i+n])}"
                    _accumulate_sparse(acc_local, key, float(text_weight), dim, seed + HASH_SEEDS[4])
                    ng_count += 1
                return (n_bytes, ng_count, acc_local)
            except Exception:
                return (0, 0, {})

        # Concurrent read/tokenize with early short-circuit
        with ThreadPoolExecutor(max_workers=min(8, max(2, os.cpu_count() or 8))) as ex:
            futures = [ex.submit(_process_file, fp) for fp in norm_files]
            for fut in as_completed(futures):
                if bytes_budget is not None and bytes_budget <= 0:
                    break
                n_bytes, ng_count, acc_local = fut.result()
                if bytes_budget is not None:
                    bytes_budget -= n_bytes
                    if bytes_budget is not None and bytes_budget <= 0:
                        # still add what we got, then stop
                        pass
                if not acc_local:
                    continue
                tokens_emitted += int(ng_count)
                for k, v in acc_local.items():
                    text_acc[k] = float(text_acc.get(k, 0.0)) + float(v)
                if token_cap and tokens_emitted >= token_cap:
                    break

        if text_acc:
            z_text = _dense_from_sparse(text_acc, dim)

    # Per-family normalize (layer-norm style), then weighted sum
    def _unit(x: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(x))
        return (x / n) if n > 0 else x

    sym_w = 1.0
    doc_w = 1.0
    mod_w = 1.0
    top_w = 1.0
    txt_w = float(text_weight)

    z = (
        sym_w * _unit(z_sym)
        + doc_w * _unit(z_doc)
        + mod_w * _unit(z_mod)
        + top_w * _unit(z_top)
        + txt_w * _unit(z_text)
        + float(types_weight) * _unit(z_types)
        + float(calls_weight) * _unit(z_calls)
        + float(tests_weight) * _unit(np.zeros_like(z_sym))  # tests view added below
    )
    norm = float(np.linalg.norm(z))
    if norm > 0:
        z = z / norm

    # Tests view (counts already influence topology; add dedicated view from test nodes)
    z_tests = np.zeros((dim,), dtype=np.float32)
    try:
        test_feats: List[Tuple[str, float]] = []
        for mod, nodes in g.pytest_nodes_by_module.items():
            if nodes:
                test_feats.append((f"tests:{mod}:{len(nodes)}", 1.0))
        if test_feats:
            z_tests = _feature_hash(test_feats, dim, seed + HASH_SEEDS[7])
    except Exception:
        z_tests = np.zeros((dim,), dtype=np.float32)

    # Re-add tests view (unit) to z
    z = z + float(tests_weight) * _unit(z_tests)

    # Sparsity diagnostics (fraction non-zero)
    def _sparse_frac(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float((np.count_nonzero(x) / float(x.size)))

    result = {
        "z": z.astype(np.float32),
        "z_sym": z_sym.astype(np.float32),
        "z_doc": z_doc.astype(np.float32),
        "z_mod": z_mod.astype(np.float32),
        "z_top": z_top.astype(np.float32),
        "z_types": z_types.astype(np.float32),
        "z_calls": z_calls.astype(np.float32),
        "z_tests": z_tests.astype(np.float32),
        "sparsity": {
            "z_sym": _sparse_frac(z_sym),
            "z_doc": _sparse_frac(z_doc),
            "z_mod": _sparse_frac(z_mod),
            "z_top": _sparse_frac(z_top),
            "z_text": _sparse_frac(z_text),
            "z_types": _sparse_frac(z_types),
            "z_calls": _sparse_frac(z_calls),
            "z_tests": _sparse_frac(z_tests),
        },
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
    calls_weight: float = 0.25,
    types_weight: float = 0.20,
    tests_weight: float = 0.15,
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
    type_feats: List[Tuple[str, float]] = []
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
            type_feats.append((f"sig:{s.signature}", 1.0))
        if s.returns:
            sym_feats.append((f"ret:{s.returns}", 0.5))
            type_feats.append((f"ret:{s.returns}", 1.0))
    # Call-graph features limited to selected modules
    call_feats: List[Tuple[str, float]] = []
    try:
        for caller, callee in getattr(g, "calls", []) or []:
            cm = caller.rsplit(".", 1)[0] if "." in caller else caller
            if mods_set and cm not in mods_set:
                continue
            call_feats.append((f"call:{caller}->{callee}", 1.0))
    except Exception:
        call_feats = []
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
    z_types = _feature_hash(type_feats, dim, seed + HASH_SEEDS[5])
    z_calls = _feature_hash(call_feats, dim, seed + HASH_SEEDS[6])

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

        def _process_file(fp: str) -> Tuple[int, List[Tuple[str, float]]]:
            try:
                per_cap = int(text_max_bytes)
                with open(fp, "rb") as fh:
                    raw = fh.read(per_cap)
                if b"\x00" in raw:
                    return (0, [])
                n_bytes = len(raw)
                text = raw.decode("utf-8", errors="ignore").lower()
                toks = re.findall(r"[a-zA-Z0-9_]+", text)
                if not toks:
                    return (n_bytes, [])
                n = 3
                stride = 2
                feats: List[Tuple[str, float]] = []
                for i in range(0, max(0, len(toks) - n + 1), stride):
                    feats.append((f"text:{' '.join(toks[i:i+n])}", float(text_weight)))
                return (n_bytes, feats)
            except Exception:
                return (0, [])

        with ThreadPoolExecutor(max_workers=min(8, max(2, os.cpu_count() or 8))) as ex:
            futures = [ex.submit(_process_file, fp) for fp in files]
            for fut in as_completed(futures):
                if bytes_budget is not None and bytes_budget <= 0:
                    break
                n_bytes, feats = fut.result()
                if bytes_budget is not None:
                    bytes_budget -= n_bytes
                    if bytes_budget is not None and bytes_budget <= 0:
                        pass
                if not feats:
                    continue
                if token_cap:
                    remaining = max(0, token_cap - tokens_emitted)
                    if remaining <= 0:
                        break
                    if len(feats) > remaining:
                        feats = feats[:remaining]
                text_feats.extend(feats)
                tokens_emitted += len(feats)
                if token_cap and tokens_emitted >= token_cap:
                    break
        if text_feats:
            z_text = _feature_hash(text_feats, dim, seed + HASH_SEEDS[4])

    def _unit(x: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(x))
        return (x / n) if n > 0 else x

    z = (
        _unit(z_sym)
        + _unit(z_doc)
        + _unit(z_mod)
        + _unit(z_top)
        + float(text_weight) * _unit(z_text)
        + float(types_weight) * _unit(z_types)
        + float(calls_weight) * _unit(z_calls)
    )
    norm = float(np.linalg.norm(z))
    if norm > 0:
        z = z / norm

    # Tests view
    z_tests = np.zeros((dim,), dtype=np.float32)
    try:
        test_feats: List[Tuple[str, float]] = []
        for mod, nodes in getattr(g, "pytest_nodes_by_module", {}).items():
            if mods_set and mod not in mods_set:
                continue
            if nodes:
                test_feats.append((f"tests:{mod}:{len(nodes)}", 1.0))
        if test_feats:
            z_tests = _feature_hash(test_feats, dim, seed + HASH_SEEDS[7])
    except Exception:
        z_tests = np.zeros((dim,), dtype=np.float32)
    z = z + float(tests_weight) * _unit(z_tests)

    result = {
        "z": z.astype(np.float32),
        "z_sym": z_sym.astype(np.float32),
        "z_doc": z_doc.astype(np.float32),
        "z_mod": z_mod.astype(np.float32),
        "z_top": z_top.astype(np.float32),
        "z_types": z_types.astype(np.float32),
        "z_calls": z_calls.astype(np.float32),
        "z_tests": z_tests.astype(np.float32),
    }
    if include_text:
        result["z_text"] = z_text.astype(np.float32)
    return result

