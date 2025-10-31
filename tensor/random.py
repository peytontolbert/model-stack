import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic(deterministic: bool = True):
    try:
        torch.use_deterministic_algorithms(deterministic)  # type: ignore[attr-defined]
    except Exception:
        pass
    torch.backends.cudnn.benchmark = not deterministic  # type: ignore[attr-defined]
    torch.backends.cudnn.deterministic = deterministic  # type: ignore[attr-defined]



# Counter-based RNG utilities
def _mix_seed_counter(seed: int, counter: int) -> int:
    # Simple 64-bit mix (SplitMix64-like) to derive a stream seed from (seed,counter)
    z = (seed ^ (counter + 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return int(z & 0x7FFFFFFFFFFFFFFF)


def philox_stream(shape: tuple[int, ...], seed: int, counter: int) -> torch.Tensor:
    """Stateless random stream: returns U[0,1) floats shaped as `shape`.

    CPU/CUDA parity: works on both; values may differ across devices but are deterministic per (seed,counter).
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_mix_seed_counter(int(seed), int(counter)))
    return torch.rand(tuple(int(s) for s in shape), generator=gen, dtype=torch.float32)


class rng_scope:
    """Context manager that isolates RNG state, safe for CUDA graph capture.

    - Seeds CPU (and CUDA without touching manual_seed_all during capture)
    - Optionally toggles deterministic algorithms
    - Restores CPU RNG state on exit
    """

    def __init__(self, seed: int, *, deterministic: bool = True):
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self._cpu_state = None

    def __enter__(self):
        from .compile import cuda_graph_seed_scope
        self._cpu_state = torch.random.get_rng_state()
        # Determinism toggles (best-effort)
        try:
            torch.use_deterministic_algorithms(self.deterministic)  # type: ignore[attr-defined]
        except Exception:
            pass
        # Seed in a way that's graph-capture friendly
        self._seed_scope = cuda_graph_seed_scope(self.seed)
        self._seed_scope.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if hasattr(self, "_seed_scope") and self._seed_scope is not None:
                self._seed_scope.__exit__(exc_type, exc_val, exc_tb)
        finally:
            if self._cpu_state is not None:
                torch.random.set_rng_state(self._cpu_state)
        return False


def dropout_mask(shape: tuple[int, ...], p: float, stream: torch.Tensor | None = None) -> torch.Tensor:
    """Generate a boolean keep mask with keep-prob (1 - p), independent of global RNG.

    If `stream` is provided, it must be a tensor of U[0,1) with `shape`.
    """
    if not (0.0 <= p < 1.0):
        raise ValueError("p must be in [0,1)")
    if stream is None:
        stream = philox_stream(shape, seed=0, counter=0)
    if tuple(stream.shape) != tuple(shape):
        raise ValueError("stream shape mismatch")
    return (stream >= float(p))

