from typing import List, Union, Tuple, Optional, Sequence, TypeVar, Any

import numpy as np
import torch
from functools import lru_cache


T = TypeVar("T", bound=torch.Tensor)
TensorLike = Union[List, np.ndarray, torch.Tensor]
Shape = Union[Tuple[int, ...], List[int]]


class TensorOps:
    @staticmethod
    def create_tensor(
        data: TensorLike,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not isinstance(data, (list, np.ndarray, torch.Tensor)):
            raise TypeError("Data must be a list, numpy array, or tensor")
        if isinstance(data, (list, np.ndarray)) and len(data) == 0:
            raise ValueError("Data cannot be empty")

        if isinstance(data, torch.Tensor):
            tensor = data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.tensor(data)

        if dtype is not None:
            tensor = tensor.to(dtype)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    @staticmethod
    def batch_dot(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
        if tensor_a.dim() != 2 or tensor_b.dim() != 2:
            raise ValueError("Tensors must be 2D (batch_size, n)")
        if tensor_a.shape != tensor_b.shape:
            raise ValueError("Tensor shapes must match")
        return torch.sum(tensor_a * tensor_b, dim=1)

    @staticmethod
    def normalize(tensor: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
        if dim >= tensor.dim():
            raise ValueError(f"Invalid dimension {dim} for tensor of dim {tensor.dim()}")
        return tensor / (tensor.norm(dim=dim, keepdim=True) + eps)

    @staticmethod
    def reshape_batch(tensor: torch.Tensor, batch_size: int, shape: Shape) -> torch.Tensor:
        total_size = int(batch_size * int(np.prod(shape)))
        if tensor.numel() != total_size:
            raise ValueError(f"Tensor size {tensor.numel()} doesn't match target size {total_size}")
        return tensor.view(batch_size, *shape)

    @staticmethod
    def concatenate(tensor_list: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        if not tensor_list:
            raise ValueError("Tensor list cannot be empty")
        return torch.cat(tensor_list, dim=dim)

    @staticmethod
    def split_batch(tensor: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if batch_size > tensor.size(0):
            raise ValueError("Batch size cannot be larger than tensor size")
        return torch.split(tensor, batch_size, dim=0)

    @staticmethod
    def type_convert(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return tensor.to(dtype)

    @staticmethod
    def gather_along_dim(tensor: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
        if dim >= tensor.dim():
            raise ValueError(f"Invalid dimension {dim} for tensor of dim {tensor.dim()}")
        return torch.gather(tensor, dim, indices)

    @staticmethod
    def masked_fill(tensor: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor:
        if tensor.shape != mask.shape:
            raise ValueError("Tensor and mask shapes must match")
        return tensor.masked_fill(mask, value)


tensor_ops = TensorOps()


# ----- Tensor contraction utilities (planning and cached execution) ----- #
try:
    import opt_einsum as _oe  # type: ignore
except Exception:
    _oe = None  # type: ignore


def _key_for_tensors(expr: str, tensors: Sequence[torch.Tensor]) -> Tuple[Any, ...]:
    shapes = tuple(tuple(int(d) for d in t.shape) for t in tensors)
    dtypes = tuple(str(t.dtype) for t in tensors)
    devices = tuple(str(t.device) for t in tensors)
    return (expr, shapes, dtypes, devices)


@lru_cache(maxsize=512)
def plan_einsum(expr: str, key: Tuple[Any, ...]) -> Optional[Any]:
    """Return a contraction path plan for expr and tensor metadata key.

    Uses opt_einsum if available; falls back to numpy path metadata, which
    we cannot directly apply in torch but cache to avoid re-planning overhead.
    """
    if _oe is not None:
        try:
            # Build a dummy shape list from key and ask opt_einsum for a path
            # The actual contraction will use the plan with the torch backend
            return _oe.contract_path(expr, *[(0,) * len(s) for s in key[1]])[0]  # type: ignore[index]
        except Exception:
            return None
    # numpy fallback (metadata only)
    try:
        import numpy as _np  # local import

        # Create small dummy arrays to get a path without heavy memory
        dummies = [
            _np.zeros([max(1, int(d)) for d in shape[:2]], dtype=_np.float32)
            for shape in key[1]
        ]  # type: ignore[index]
        _path, _ = _np.einsum_path(expr, *dummies, optimize="greedy")
        return _path
    except Exception:
        return None


def contract(expr: str, *tensors: torch.Tensor, optimize: Union[bool, str, Any] = "auto") -> torch.Tensor:
    """Contract tensors via einsum with optional path optimization.

    - If opt_einsum is installed, uses opt_einsum.contract with backend='torch'.
    - Else falls back to torch.einsum.
    - optimize='auto' will cache a plan keyed by (expr, shapes, dtypes, devices).
    - Passing a specific plan object (from plan_einsum) is supported when using opt_einsum.
    """
    if _oe is not None:
        try:
            if optimize == "auto":
                key = _key_for_tensors(expr, tensors)
                path = plan_einsum(expr, key)
                return _oe.contract(expr, *tensors, backend="torch", optimize=path if path is not None else "greedy")
            else:
                return _oe.contract(expr, *tensors, backend="torch", optimize=optimize)
        except Exception:
            return torch.einsum(expr, *tensors)
    # Fallback: torch.einsum only
    return torch.einsum(expr, *tensors)


def detect_band_mask(mask: torch.Tensor, max_bandwidth: int = 1024) -> Optional[int]:
    """Heuristically detect if an attention mask encodes a banded structure.

    Expects a boolean mask broadcastable to [B, H, Lq, Lk] where True means keep.
    Returns estimated half-bandwidth (w) if banded, else None.
    """
    try:
        if mask.dtype != torch.bool:
            return None
        while mask.dim() < 4:
            mask = mask.unsqueeze(0)
        sample = mask[0, 0].to(torch.bool)  # [Lq, Lk]
        Lq, Lk = sample.shape[-2], sample.shape[-1]
        # Take a few rows and estimate contiguous True region around diagonal
        rows = [0, Lq // 4, Lq // 2, (3 * Lq) // 4, Lq - 1]
        est = 0
        for r in rows:
            if r < 0 or r >= Lq:
                continue
            row = sample[r]
            idx = torch.nonzero(row, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            lo = int(idx.min().item())
            hi = int(idx.max().item())
            # distance from diagonal
            center = r
            est = max(est, max(abs(lo - center), abs(hi - center)))
        if est > 0 and est <= int(max_bandwidth):
            return int(est)
        return None
    except Exception:
        return None

