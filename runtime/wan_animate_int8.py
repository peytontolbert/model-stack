"""Selective INT8 conversion utilities for Wan Animate transformer blocks.

Wan Animate is numerically sensitive outside its projection matrices.  This
adapter therefore quantizes only ``nn.Linear`` weights per output channel and
keeps norms, embeddings, convolutional paths, and biases in their original
BF16/FP32 compute dtypes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import nn
from safetensors.torch import load_file as load_safetensors_file, save_file as save_safetensors_file

from compress.quantization import QuantizedLinearInt8


@dataclass(frozen=True)
class WanInt8Config:
    calibration: str = "absmax"
    percentile: float = 0.999
    activation_quant: str = "dynamic_int8"
    activation_quant_bits: int = 8
    activation_quant_method: str = "absmax"
    activation_quant_percentile: float = 0.999
    min_weight_elements: int = 16_384


@dataclass(frozen=True)
class WanInt8Inventory:
    quantized: tuple[str, ...]
    skipped: tuple[str, ...]


_BF16_ONLY_NAME_PARTS = (
    "norm",
    "embed",
    "embedding",
    "modulation",
    "scale",
    "gate",
)


def _split_parent(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parent = root
    pieces = dotted_name.split(".")
    for piece in pieces[:-1]:
        parent = getattr(parent, piece)
    return parent, pieces[-1]


def _should_quantize(name: str, module: nn.Linear, config: WanInt8Config) -> bool:
    lowered = name.lower()
    if any(part in lowered for part in _BF16_ONLY_NAME_PARTS):
        return False
    return module.weight is not None and module.weight.numel() >= config.min_weight_elements


@torch.no_grad()
def convert_wan_linears_to_int8(
    model: nn.Module,
    config: WanInt8Config = WanInt8Config(),
) -> WanInt8Inventory:
    """Replace eligible Linear modules with ModelStack INT8 linear modules.

    Run this on CPU before device placement or FSDP wrapping.  The resulting
    weights are deterministic for a fixed BF16 source checkpoint and config.
    """
    quantized: list[str] = []
    skipped: list[str] = []
    candidates = list(model.named_modules())
    for name, module in candidates:
        if not name or not isinstance(module, nn.Linear):
            continue
        if not _should_quantize(name, module, config):
            skipped.append(name)
            continue
        parent, child_name = _split_parent(model, name)
        replacement = QuantizedLinearInt8(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
        ).to(device=module.weight.device)
        replacement.from_float(
            module,
            calibration=config.calibration,
            percentile=config.percentile,
            activation_quant=config.activation_quant,
            activation_quant_bits=config.activation_quant_bits,
            activation_quant_method=config.activation_quant_method,
            activation_quant_percentile=config.activation_quant_percentile,
        )
        if module.bias is not None and replacement.bias is not None:
            replacement.bias.data = replacement.bias.data.to(dtype=module.bias.dtype)
        setattr(parent, child_name, replacement)
        quantized.append(name)
    return WanInt8Inventory(tuple(quantized), tuple(skipped))


def replace_named_linears_with_int8(model: nn.Module, names: Iterable[str]) -> None:
    """Install empty INT8 modules before loading an exported artifact.

    This intentionally does no conversion.  ``load_state_dict`` supplies the
    qweight/scales and preserves the original BF16 bias dtype from the
    artifact, avoiding a transient BF16 copy of the full transformer.
    """
    for name in names:
        parent, child_name = _split_parent(model, name)
        module = getattr(parent, child_name)
        if isinstance(module, QuantizedLinearInt8):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"expected nn.Linear at {name}, found {type(module)!r}")
        replacement = QuantizedLinearInt8(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
        ).to(device=module.weight.device)
        if module.bias is not None:
            replacement.bias = nn.Parameter(
                torch.empty_like(module.bias), requires_grad=module.bias.requires_grad
            )
        setattr(parent, child_name, replacement)


def quantized_module_names_from_state(state: dict[str, torch.Tensor], prefix: str = "") -> tuple[str, ...]:
    """Derive local module paths from state keys such as ``foo.qweight``."""
    result: list[str] = []
    for key in state:
        if not key.endswith(".qweight"):
            continue
        name = key[: -len(".qweight")]
        if prefix:
            if not name.startswith(prefix):
                continue
            name = name[len(prefix):]
        result.append(name)
    return tuple(sorted(result))


def _canonical_json(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_int8_manifest(
    output_dir: str | Path,
    *,
    source_checkpoint: str | Path,
    config: WanInt8Config,
    inventory: WanInt8Inventory,
    rank: int,
    world_size: int,
) -> Path:
    output_dir = Path(output_dir)
    payload = {
        "format": "modelstack.wan-animate.int8.v1",
        "source_checkpoint": str(source_checkpoint),
        "rank": int(rank),
        "world_size": int(world_size),
        "config": asdict(config),
        "quantized_modules": list(inventory.quantized),
        "skipped_modules": list(inventory.skipped),
    }
    payload["fingerprint"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    destination = output_dir / f"wan_animate_int8.rank{rank:02d}.manifest.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(payload) + b"\n")
    os.replace(temporary, destination)
    return destination


def save_int8_rank_local_state(
    model: nn.Module,
    output_dir: str | Path,
    *,
    rank: int,
) -> Path:
    """Atomically save the rank-local quantized state on CPU."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"wan_animate_int8.rank{rank:02d}.pt"
    temporary = destination.with_suffix(".tmp")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, temporary)
    os.replace(temporary, destination)
    return destination


def block_state_dict(model: nn.Module, block_index: int) -> dict[str, torch.Tensor]:
    """Return a CPU state dictionary for one Wan transformer block.

    Keys are local to ``model.blocks[block_index]`` so a runtime can hydrate
    only its active block group.  The tensors include int8 weights, their
    per-channel scales, and the BF16/FP32 parameters deliberately excluded
    from conversion.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError("Wan INT8 export requires a model.blocks ModuleList")
    if block_index < 0 or block_index >= len(blocks):
        raise IndexError(f"block index {block_index} is outside model.blocks")
    return {name: tensor.detach().cpu().contiguous() for name, tensor in blocks[block_index].state_dict().items()}


def _save_tensor_state(state: dict[str, torch.Tensor], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if destination.suffix == ".safetensors":
        save_safetensors_file(state, str(temporary))
    else:
        torch.save(state, temporary)
    os.replace(temporary, destination)
    return destination


def _load_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_safetensors_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def _preferred_state_path(path_without_suffix: Path) -> Path:
    safetensors_path = path_without_suffix.with_suffix(".safetensors")
    if safetensors_path.is_file():
        return safetensors_path
    return path_without_suffix.with_suffix(".pt")


def save_int8_block_state(
    model: nn.Module,
    output_dir: str | Path,
    *,
    block_index: int,
    storage_format: str = "pt",
) -> Path:
    """Atomically persist one quantized block, avoiding a monolithic artifact."""
    output_dir = Path(output_dir)
    blocks_dir = output_dir / "blocks"
    suffix = ".safetensors" if storage_format == "safetensors" else ".pt"
    destination = blocks_dir / f"block_{block_index:02d}{suffix}"
    return _save_tensor_state(block_state_dict(model, block_index), destination)


def write_int8_block_manifest(
    output_dir: str | Path,
    *,
    source_checkpoint: str | Path,
    config: WanInt8Config,
    inventory: WanInt8Inventory,
    num_blocks: int,
    storage_formats: Iterable[str] = ("pt",),
) -> Path:
    """Write the source/config fingerprint consumed by the block offloader."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_block: dict[str, list[str]] = {str(index): [] for index in range(num_blocks)}
    for name in inventory.quantized:
        match = re.match(r"^blocks\.(\d+)\.(.+)$", name)
        if match:
            by_block[match.group(1)].append(match.group(2))
    payload = {
        "format": "modelstack.wan-animate.int8.blocks.v1",
        "source_checkpoint": str(source_checkpoint),
        "config": asdict(config),
        "num_blocks": int(num_blocks),
        "quantized_modules": list(inventory.quantized),
        "quantized_modules_by_block": by_block,
        "skipped_modules": list(inventory.skipped),
        "storage_formats": sorted(set(str(fmt) for fmt in storage_formats)),
    }
    payload["fingerprint"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    destination = output_dir / "manifest.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(_canonical_json(payload) + b"\n")
    os.replace(temporary, destination)
    return destination


class WanBlockOffloader:
    """Keep one Wan block group resident and prefetch the next group.

    The owner calls ``before_block(index)`` immediately before each block
    forward.  This is deliberately separate from Wan's model class so it can
    be attached to upstream code without copying its forward implementation.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        *,
        device: torch.device | str,
        group_size: int = 1,
        prefetch: bool = True,
    ) -> None:
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.blocks = blocks
        self.device = torch.device(device)
        self.group_size = int(group_size)
        self.prefetch = bool(prefetch and self.device.type == "cuda")
        self._resident_group: int | None = None
        self._prefetched_group: int | None = None
        self._stream = torch.cuda.Stream(device=self.device) if self.prefetch else None

    def _group(self, block_index: int) -> int:
        return block_index // self.group_size

    def _group_range(self, group: int) -> range:
        start = group * self.group_size
        return range(start, min(start + self.group_size, len(self.blocks)))

    @staticmethod
    def _clear_int8_caches(module: nn.Module) -> None:
        for child in module.modules():
            invalidate = getattr(child, "_invalidate_weight_cache", None)
            if callable(invalidate):
                invalidate()

    def _move_group(self, group: int, device: torch.device, *, non_blocking: bool) -> None:
        for index in self._group_range(group):
            block = self.blocks[index]
            block.to(device=device, non_blocking=non_blocking)
            self._clear_int8_caches(block)

    def _evict_group(self, group: int) -> None:
        self._move_group(group, torch.device("cpu"), non_blocking=False)

    def _prefetch_group(self, group: int) -> None:
        if group * self.group_size >= len(self.blocks) or self._prefetched_group == group:
            return
        if self._stream is None:
            self._move_group(group, self.device, non_blocking=False)
        else:
            with torch.cuda.stream(self._stream):
                self._move_group(group, self.device, non_blocking=True)
        self._prefetched_group = group

    def before_block(self, block_index: int) -> None:
        group = self._group(block_index)
        if group != self._resident_group:
            if self._prefetched_group == group and self._stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(self._stream)
            else:
                self._move_group(group, self.device, non_blocking=False)
            prior = self._resident_group
            self._resident_group = group
            self._prefetched_group = None
            if prior is not None and prior != group:
                self._evict_group(prior)
        if self.prefetch:
            self._prefetch_group(group + 1)

    def close(self) -> None:
        if self._stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._stream)
            self._stream.synchronize()
        if self._resident_group is not None:
            self._evict_group(self._resident_group)
        if self._prefetched_group is not None:
            self._evict_group(self._prefetched_group)
        self._resident_group = None
        self._prefetched_group = None


def attach_wan_block_offload(
    model: nn.Module,
    *,
    device: torch.device | str,
    group_size: int = 1,
    prefetch: bool = True,
) -> WanBlockOffloader:
    """Attach an offloader to an upstream Wan model without replacing forward."""
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("Wan block offload requires model.blocks to be a ModuleList")
    offloader = WanBlockOffloader(blocks, device=device, group_size=group_size, prefetch=prefetch)
    for index, block in enumerate(blocks):
        original_forward = block.forward

        def wrapped_forward(*args, __index: int = index, __forward: Callable = original_forward, **kwargs):
            offloader.before_block(__index)
            return __forward(*args, **kwargs)

        block.forward = wrapped_forward  # type: ignore[method-assign]
    return offloader


def move_wan_non_block_modules(model: nn.Module, device: torch.device | str) -> None:
    """Place Wan embeddings, adapters and head without moving ``model.blocks``."""
    for name, child in model.named_children():
        if name != "blocks":
            child.to(device)
    freqs = getattr(model, "freqs", None)
    if isinstance(freqs, torch.Tensor):
        model.freqs = freqs.to(device)


def load_wan_int8_blocks(model: nn.Module, artifact_dir: str | Path) -> nn.Module:
    """Hydrate a freshly constructed Wan model from block INT8 artifacts.

    ``model`` must be constructed on CPU in the original checkpoint dtype.
    Nothing is moved to CUDA here; call :func:`attach_wan_block_offload` after
    this returns to establish the explicit GPU placement policy.
    """
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing Wan INT8 manifest: {manifest_path}")
    if not (artifact_dir / "non_blocks.safetensors").is_file() and not (artifact_dir / "non_blocks.pt").is_file():
        raise FileNotFoundError(f"missing Wan INT8 non-block state under: {artifact_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "modelstack.wan-animate.int8.blocks.v1":
        raise RuntimeError(f"unsupported Wan INT8 artifact format: {manifest.get('format')!r}")
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("Wan INT8 loader requires model.blocks to be a ModuleList")
    if int(manifest["num_blocks"]) != len(blocks):
        raise RuntimeError("INT8 artifact block count does not match the constructed Wan model")

    non_blocks_path = _preferred_state_path(artifact_dir / "non_blocks")
    non_blocks = _load_tensor_state(non_blocks_path)
    names = list(manifest.get("quantized_modules", ()))
    if not names:
        names = list(quantized_module_names_from_state(non_blocks))
        for index in range(len(blocks)):
            block_state = _load_tensor_state(_preferred_state_path(artifact_dir / "blocks" / f"block_{index:02d}"))
            names.extend(f"blocks.{index}.{name}" for name in quantized_module_names_from_state(block_state))
    replace_named_linears_with_int8(model, names)

    incompatible = model.load_state_dict(non_blocks, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("blocks.")]
    missing = [key for key in incompatible.missing_keys if not key.startswith("blocks.")]
    if unexpected or missing:
        raise RuntimeError(f"invalid Wan INT8 non-block state; missing={missing}, unexpected={unexpected}")
    for index, block in enumerate(blocks):
        state = _load_tensor_state(_preferred_state_path(artifact_dir / "blocks" / f"block_{index:02d}"))
        incompatible = block.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"invalid Wan INT8 block {index}: {incompatible}")
    return model
