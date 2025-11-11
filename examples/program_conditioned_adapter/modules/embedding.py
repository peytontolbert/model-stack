from typing import Any, Dict, List, Optional, Tuple
import os
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import pathspec  # type: ignore
except Exception:  # pragma: no cover
    pathspec = None  # type: ignore
from examples.program_conditioned_adapter.modules.program_graph import ProgramGraph, Artifact  # type: ignore

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


def _artifact_rel_path(art: Artifact) -> Optional[str]:
    try:
        # Expect URIs like program://<program_id>/artifact/<rel_path>
        u = art.uri
        if "/artifact/" in u:
            return u.split("/artifact/", 1)[-1]
        return None
    except Exception:
        return None


def build_program_embedding(
    pg: ProgramGraph,
    *,
    sources_root: Optional[str] = None,
    dim: int = EMBED_DIM_DEFAULT,
    seed: int = 0,
    include_text: bool = False,
    text_max_bytes: int = 0,
    max_text_tokens: int = 0,
    text_weight: float = 0.25,
    calls_weight: float = 0.25,
    types_weight: float = 0.20,  # kept for signature parity; may be neutral here
    tests_weight: float = 0.15,  # kept for parity; may be neutral here
    contracts_kv: Optional[Dict[str, str]] = None,
    contracts_weight: float = 0.10,
    graph_prop_hops: int = 0,
    graph_prop_damp: float = 0.85,
    ignore: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Program-agnostic embedding from a ProgramGraph."""
    # Symbolic/channel features from entities
    sym_feats: List[Tuple[str, float]] = []
    mod_feats: List[Tuple[str, float]] = []
    owners: List[str] = []
    try:
        for e in pg.entities():
            sym_feats.append((f"{e.kind}:{e.name}", 1.0))
            if e.owner:
                owners.append(e.owner)
                mod_feats.append((f"owner:{e.owner}", 0.2))
    except Exception:
        pass

    # Topology from edges aggregated by owner and entity degrees
    indeg: Dict[str, int] = {}
    outdeg: Dict[str, int] = {}
    ent_indeg: Dict[str, int] = {}
    ent_outdeg: Dict[str, int] = {}
    try:
        for ed in pg.edges():
            ent_outdeg[ed.src] = ent_outdeg.get(ed.src, 0) + 1
            ent_indeg[ed.dst] = ent_indeg.get(ed.dst, 0) + 1
        # Map entity degrees to owners if available
        ent_owner: Dict[str, Optional[str]] = {}
        try:
            for e in pg.entities():
                ent_owner[e.id] = e.owner
        except Exception:
            ent_owner = {}
        for ent, d in ent_indeg.items():
            ow = ent_owner.get(ent)
            if ow:
                indeg[ow] = indeg.get(ow, 0) + int(d)
        for ent, d in ent_outdeg.items():
            ow = ent_owner.get(ent)
            if ow:
                outdeg[ow] = outdeg.get(ow, 0) + int(d)
    except Exception:
        pass

    # Add simple file/dir topology derived from artifacts to ensure z_top density
    dir_counts: Dict[str, int] = {}
    try:
        for art in pg.artifacts("source"):
            rel = _artifact_rel_path(art)
            if not rel:
                continue
            dname = os.path.dirname(rel).replace("\\", "/")
            dir_counts[dname] = dir_counts.get(dname, 0) + 1
    except Exception:
        # Fallback: approximate by walking sources_root if available
        if sources_root:
            try:
                for dirpath, _dirnames, filenames in os.walk(sources_root):
                    rel_dir = os.path.relpath(dirpath, sources_root).replace("\\", "/")
                    if rel_dir == ".":
                        rel_dir = ""
                    if filenames:
                        dir_counts[rel_dir] = dir_counts.get(rel_dir, 0) + len(filenames)
            except Exception:
                pass
    topo_feats: List[Tuple[str, float]] = []
    owners_unique = sorted(set(owners))
    for ow in owners_unique:
        topo_feats.append((f"indeg:{ow}", float(indeg.get(ow, 0))))
        topo_feats.append((f"outdeg:{ow}", float(outdeg.get(ow, 0))))
    for d, c in dir_counts.items():
        topo_feats.append((f"dir:{d}", float(c)))

    # No dedicated doc/types/calls views at generic layer; set to zeros
    z_sym = _feature_hash(sym_feats, dim, seed + HASH_SEEDS[0])
    z_doc = np.zeros((dim,), dtype=np.float32)
    z_mod = _feature_hash(mod_feats, dim, seed + HASH_SEEDS[2])
    z_top = _feature_hash(topo_feats, dim, seed + HASH_SEEDS[3])
    z_types = np.zeros((dim,), dtype=np.float32)
    z_calls = np.zeros((dim,), dtype=np.float32)
    # Optional contracts channel: hash key/value hints from ProgramContracts
    z_contracts = np.zeros((dim,), dtype=np.float32)
    try:
        if contracts_kv:
            feats_c: List[Tuple[str, float]] = []
            for k, v in list(contracts_kv.items()):
                if v is None:
                    continue
                feats_c.append((f"contracts:{str(k)}:{str(v)}", 1.0))
            if feats_c:
                z_contracts = _feature_hash(feats_c, dim, seed + HASH_SEEDS[6])
    except Exception:
        z_contracts = np.zeros((dim,), dtype=np.float32)

    # Optional textual channel from artifacts(kind="source")
    z_text = np.zeros((dim,), dtype=np.float32)
    if include_text and text_max_bytes and text_max_bytes > 0 and sources_root:
        ignore_list = [os.path.normpath(p) for p in (ignore or [])]
        def _ignored(rel: str) -> bool:
            r = os.path.normpath(rel)
            for pat in ignore_list:
                if r == pat or r.startswith(pat + os.sep):
                    return True
            return False
        text_acc: Dict[int, float] = {}
        bytes_budget: Optional[int] = int(text_max_bytes) if int(text_max_bytes) > 0 else None
        token_cap = int(max(0, int(max_text_tokens))) if max_text_tokens is not None else 0
        tokens_emitted = 0
        files: List[str] = []
        try:
            for art in pg.artifacts("source"):
                rel = _artifact_rel_path(art)
                if not rel or _ignored(rel):
                    continue
                ap = os.path.abspath(os.path.join(sources_root, rel))
                if os.path.isfile(ap):
                    files.append(ap)
        except Exception:
            files = []
        # Fallback if ProgramGraph provided no artifacts
        if not files:
            try:
                exts_all = {".py", ".md", ".rst", ".txt", ".json", ".toml", ".yaml", ".yml", ".ini"}
                for root, dirs, fnames in os.walk(sources_root):
                    rel_root = os.path.relpath(root, sources_root)
                    if _ignored(rel_root):
                        dirs[:] = []
                        continue
                    dirs[:] = [d for d in dirs if not _ignored(os.path.join(rel_root, d))]
                    for fn in fnames:
                        fp = os.path.join(root, fn)
                        rel_fp = os.path.relpath(fp, sources_root)
                        if _ignored(rel_fp):
                            continue
                        if os.path.splitext(fp)[1].lower() in exts_all:
                            files.append(fp)
            except Exception:
                files = []
        def _process_file(fp: str) -> Tuple[int, int, Dict[int, float]]:
            try:
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
        with ThreadPoolExecutor(max_workers=min(8, max(2, os.cpu_count() or 8))) as ex:
            futures = [ex.submit(_process_file, fp) for fp in files]
            for fut in as_completed(futures):
                if bytes_budget is not None and bytes_budget <= 0:
                    break
                n_bytes, ng_count, acc_local = fut.result()
                if bytes_budget is not None:
                    bytes_budget -= n_bytes
                    if bytes_budget is not None and bytes_budget <= 0:
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

    def _unit(x: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(x))
        return (x / n) if n > 0 else x
    z = (
        _unit(z_sym)
        + _unit(z_mod)
        + _unit(z_top)
        + float(text_weight) * _unit(z_text)
        + float(types_weight) * _unit(z_types)
        + float(calls_weight) * _unit(z_calls)
        + float(max(0.0, contracts_weight)) * _unit(z_contracts)
        + float(tests_weight) * _unit(np.zeros_like(z_sym))
    )
    nrm = float(np.linalg.norm(z))
    if nrm > 0:
        z = z / nrm
    result = {
        "z": z.astype(np.float32),
        "z_sym": z_sym.astype(np.float32),
        "z_doc": z_doc.astype(np.float32),
        "z_mod": z_mod.astype(np.float32),
        "z_top": z_top.astype(np.float32),
        "z_types": z_types.astype(np.float32),
        "z_calls": z_calls.astype(np.float32),
        "z_contracts": z_contracts.astype(np.float32),
        "z_tests": np.zeros((dim,), dtype=np.float32),
        "sparsity": {
            "z_sym": float((np.count_nonzero(z_sym) / float(max(1, z_sym.size)))),
            "z_doc": float((np.count_nonzero(z_doc) / float(max(1, z_doc.size)))),
            "z_mod": float((np.count_nonzero(z_mod) / float(max(1, z_mod.size)))),
            "z_top": float((np.count_nonzero(z_top) / float(max(1, z_top.size)))),
            "z_text": float((np.count_nonzero(z_text) / float(max(1, z_text.size)))),
            "z_types": float((np.count_nonzero(z_types) / float(max(1, z_types.size)))),
            "z_calls": float((np.count_nonzero(z_calls) / float(max(1, z_calls.size)))),
            "z_contracts": float((np.count_nonzero(z_contracts) / float(max(1, z_contracts.size)))),
            "z_tests": 0.0,
        },
    }
    if include_text:
        result["z_text"] = z_text.astype(np.float32)
    return result


def build_subgraph_embedding_from_program(
    pg: ProgramGraph,
    *,
    sources_root: Optional[str] = None,
    dim: int = EMBED_DIM_DEFAULT,
    seed: int = 0,
    include_owners: Optional[List[str]] = None,
    include_artifact_paths: Optional[List[str]] = None,
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
    owners_set = set(include_owners or [])
    files_set_norm: Optional[Set[str]] = None
    if include_artifact_paths:
        files_set_norm = set([p.replace("\\", "/") for p in include_artifact_paths])
    # Entities filtered by owners
    sym_feats: List[Tuple[str, float]] = []
    mod_feats: List[Tuple[str, float]] = []
    try:
        for e in pg.entities():
            if owners_set and (e.owner not in owners_set):
                continue
            sym_feats.append((f"{e.kind}:{e.name}", 1.0))
            if e.owner:
                mod_feats.append((f"owner:{e.owner}", 0.2))
    except Exception:
        pass
    # Topology filtered by owners
    indeg: Dict[str, int] = {}
    outdeg: Dict[str, int] = {}
    try:
        ent_owner: Dict[str, Optional[str]] = {}
        for e in pg.entities():
            ent_owner[e.id] = e.owner
        for ed in pg.edges():
            ow_src = ent_owner.get(ed.src)
            ow_dst = ent_owner.get(ed.dst)
            if owners_set and (ow_src not in owners_set and ow_dst not in owners_set):
                continue
            if ow_src:
                outdeg[ow_src] = outdeg.get(ow_src, 0) + 1
            if ow_dst:
                indeg[ow_dst] = indeg.get(ow_dst, 0) + 1
    except Exception:
        pass
    topo_feats: List[Tuple[str, float]] = []
    for ow in sorted(set([o for o in (include_owners or []) if o])):
        topo_feats.append((f"indeg:{ow}", float(indeg.get(ow, 0))))
        topo_feats.append((f"outdeg:{ow}", float(outdeg.get(ow, 0))))
    # Text channel: only from included artifact paths
    z_text = np.zeros((dim,), dtype=np.float32)
    if include_text and text_max_bytes and text_max_bytes > 0 and sources_root and files_set_norm:
        text_acc: Dict[int, float] = {}
        bytes_budget: Optional[int] = int(text_max_bytes) if int(text_max_bytes) > 0 else None
        token_cap = int(max(0, int(max_text_tokens))) if max_text_tokens is not None else 0
        tokens_emitted = 0
        files: List[str] = []
        try:
            for art in pg.artifacts("source"):
                rel = _artifact_rel_path(art)
                if not rel:
                    continue
                # allow both repo-absolute and rel forms in allowlist
                rel_norm = rel.replace("\\", "/")
                abs_norm = os.path.abspath(os.path.join(sources_root, rel_norm))
                if (rel_norm in files_set_norm) or (abs_norm in files_set_norm):
                    files.append(abs_norm)
        except Exception:
            files = []
        def _process_file(fp: str) -> Tuple[int, int, Dict[int, float]]:
            try:
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
        with ThreadPoolExecutor(max_workers=min(8, max(2, os.cpu_count() or 8))) as ex:
            futures = [ex.submit(_process_file, fp) for fp in files]
            for fut in as_completed(futures):
                if bytes_budget is not None and bytes_budget <= 0:
                    break
                n_bytes, ng_count, acc_local = fut.result()
                if bytes_budget is not None:
                    bytes_budget -= n_bytes
                if not acc_local:
                    continue
                tokens_emitted += int(ng_count)
                for k, v in acc_local.items():
                    text_acc[k] = float(text_acc.get(k, 0.0)) + float(v)
                if token_cap and tokens_emitted >= token_cap:
                    break
        if text_acc:
            z_text = _dense_from_sparse(text_acc, dim)
    z_sym = _feature_hash(sym_feats, dim, seed + HASH_SEEDS[0])
    z_mod = _feature_hash(mod_feats, dim, seed + HASH_SEEDS[2])
    z_top = _feature_hash(topo_feats, dim, seed + HASH_SEEDS[3])
    z_types = np.zeros((dim,), dtype=np.float32)
    z_calls = np.zeros((dim,), dtype=np.float32)
    def _unit(x: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(x))
        return (x / n) if n > 0 else x
    z = (
        _unit(z_sym)
        + _unit(z_mod)
        + _unit(z_top)
        + float(text_weight) * _unit(z_text)
        + float(types_weight) * _unit(z_types)
        + float(calls_weight) * _unit(z_calls)
        + float(tests_weight) * _unit(np.zeros_like(z_sym))
    )
    nrm = float(np.linalg.norm(z))
    if nrm > 0:
        z = z / nrm
    result = {
        "z": z.astype(np.float32),
        "z_sym": z_sym.astype(np.float32),
        "z_doc": np.zeros((dim,), dtype=np.float32),
        "z_mod": z_mod.astype(np.float32),
        "z_top": z_top.astype(np.float32),
        "z_types": z_types.astype(np.float32),
        "z_calls": z_calls.astype(np.float32),
        "z_tests": np.zeros((dim,), dtype=np.float32),
    }
    if include_text:
        result["z_text"] = z_text.astype(np.float32)
    return result

def join_embeddings(z_old: Optional[np.ndarray], z_new: Optional[np.ndarray], *, w_old: float = 1.0, w_new: float = 1.0) -> Optional[np.ndarray]:
    """Monotone anytime join of two embedding vectors.

    Returns a unit-normalized convex-like combination w_old*unit(z_old) + w_new*unit(z_new).
    If both are None, returns None.
    """
    try:
        def _unit(a: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(a))
            return (a / n) if n > 0 else a
        if z_old is None and z_new is None:
            return None
        if z_old is None:
            return _unit((max(0.0, float(w_new)) * _unit(z_new.astype(np.float32))))
        if z_new is None:
            return _unit((max(0.0, float(w_old)) * _unit(z_old.astype(np.float32))))
        a = max(0.0, float(w_old)) * _unit(z_old.astype(np.float32))
        b = max(0.0, float(w_new)) * _unit(z_new.astype(np.float32))
        return _unit(a + b)
    except Exception:
        return z_new if z_new is not None else z_old
