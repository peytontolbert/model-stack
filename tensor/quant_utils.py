import torch


def quant_scale_per_channel(x: torch.Tensor, axis: int = -1, method: str = "absmax") -> torch.Tensor:
    if method == "absmax":
        return x.abs().amax(dim=axis, keepdim=True).clamp_min(1e-8) / 127.0
    # fallback to absmax if mse not implemented
    return x.abs().amax(dim=axis, keepdim=True).clamp_min(1e-8) / 127.0


def pack_int8_weight_linear(weight: torch.Tensor, axis: int = 0):
    # per-row (axis=0) scales by default; CPU-friendly fallback
    if not isinstance(weight, torch.Tensor):
        raise TypeError("weight must be a torch.Tensor")
    scales = quant_scale_per_channel(weight, axis=axis)
    q = torch.clamp((weight / scales).round(), -127, 127).to(torch.int8)
    meta = {"axis": axis, "group_size": None, "zero_point": 0}
    return q, scales.squeeze(axis), meta



# Calibrators and groupwise helpers
class QuantMeta:
    def __init__(self, bitwidth: int, symmetric: bool = True, group: int | None = None, axis: int = -1, zp_dtype: torch.dtype = torch.int32):
        self.bitwidth = int(bitwidth)
        self.symmetric = bool(symmetric)
        self.group = None if group is None else int(group)
        self.axis = int(axis)
        self.zp_dtype = zp_dtype


def groupwise_absmax(x: torch.Tensor, group: int = 64, axis: int = -1) -> torch.Tensor:
    n = x.size(axis)
    if n % group != 0:
        pad = group - (n % group)
        pad_shape = list(x.shape)
        pad_shape[axis] = pad
        pad_tensor = x.new_zeros(pad_shape)
        x = torch.cat([x, pad_tensor], dim=axis)
        n = x.size(axis)
    shape = list(x.shape)
    shape[axis] = n // group
    shape.insert(axis + 1, group)
    xg = x.view(*shape)
    return xg.abs().amax(dim=axis + 1, keepdim=True)


def fold_scales(weight: torch.Tensor, scale: torch.Tensor, axis: int = 0) -> torch.Tensor:
    while scale.ndim < weight.ndim:
        scale = scale.unsqueeze(-1)
    if axis == 0:
        return weight * scale
    if axis == -1 or axis == (weight.ndim - 1):
        return weight * scale.transpose(0, -1)
    return weight * scale


def unfold_scales(weight: torch.Tensor, scale: torch.Tensor, axis: int = 0) -> torch.Tensor:
    while scale.ndim < weight.ndim:
        scale = scale.unsqueeze(-1)
    if axis == 0:
        return weight / scale.clamp_min(1e-12)
    if axis == -1 or axis == (weight.ndim - 1):
        return weight / scale.transpose(0, -1).clamp_min(1e-12)
    return weight / scale.clamp_min(1e-12)


def hist_calibrator(num_bins: int = 2048):
    """Return a stateful calibrator with .update(x) -> {scales, zp}.

    Simplified: tracks absmax per-channel; histogram not persisted.
    """
    class _C:
        def __init__(self, bins: int):
            self.bins = int(bins)

        def update(self, x: torch.Tensor, axis: int = -1, bitwidth: int = 8, symmetric: bool = True):
            scale = quant_scale_per_channel(x, axis=axis, method="absmax")
            zp = torch.zeros_like(scale, dtype=torch.int32)
            return {"scales": scale, "zp": zp, "bitwidth": bitwidth, "symmetric": symmetric}

    return _C(num_bins)


def mse_calibrator(x: torch.Tensor, bitwidth: int = 8, axis: int = -1):
    """One-shot MSE calibrator (simplified): returns scales, zp for given bitwidth.

    Uses absmax scale as a fast approximation.
    """
    scale = quant_scale_per_channel(x, axis=axis, method="absmax")
    zp = torch.zeros_like(scale, dtype=torch.int32)
    return {"scales": scale, "zp": zp, "bitwidth": int(bitwidth)}


def int8_clip_activation_(x: torch.Tensor, scale: float) -> torch.Tensor:
    lo = -127.0 * float(scale)
    hi = 127.0 * float(scale)
    return x.clamp_(min=lo, max=hi)

