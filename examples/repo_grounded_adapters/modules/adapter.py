from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import math
import os
import json
import re
from .embedding import _stable_hash

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

