from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compress.quantization import TrainableBitNetLinear
from compress.quantization import _bitnet_runtime_row_codes_and_scale
from runtime.native import cuda_kernel_ops, has_native_op
from runtime.ops import bitnet_runtime_row_quantize
from runtime.quant import bitnet_int8_linear_from_float as runtime_bitnet_int8_linear_from_float


@dataclass(frozen=True)
class LinearShape:
    name: str
    in_features: int
    out_features: int


BITNET_STE_MODES = ("dynamic_int8_ste", "dynamic_int4_ste")


PRESETS: dict[str, list[LinearShape]] = {
    "runtime_row_1024x7_relu2_mlp3": [
        LinearShape("attn_qkv_fused", 1024, 1536),
        LinearShape("attn_out", 1024, 1024),
        LinearShape("mlp_up_relu2", 1024, 3072),
        LinearShape("mlp_down_relu2", 3072, 1024),
        LinearShape("lm_head", 1024, 1024),
    ],
    "runtime_row_1024x7_swiglu_mlp2": [
        LinearShape("attn_qkv_fused", 1024, 1536),
        LinearShape("attn_out", 1024, 1024),
        LinearShape("swiglu_gate_up", 1024, 4096),
        LinearShape("swiglu_down", 2048, 1024),
        LinearShape("lm_head", 1024, 1024),
    ],
}


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    previous = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _parse_shapes(values: list[str]) -> list[LinearShape]:
    shapes: list[LinearShape] = []
    for idx, value in enumerate(values):
        parts = [part.strip() for part in value.split(":")]
        if len(parts) == 2:
            name = f"custom_{idx}"
            in_features, out_features = parts
        elif len(parts) == 3:
            name, in_features, out_features = parts
        else:
            raise ValueError("--shape must be IN:OUT or NAME:IN:OUT")
        shapes.append(LinearShape(name, int(in_features), int(out_features)))
    return shapes


def _make_layer(shape: LinearShape, *, dtype: torch.dtype, device: torch.device, bias: bool) -> TrainableBitNetLinear:
    del dtype
    layer = TrainableBitNetLinear(shape.in_features, shape.out_features, bias=bias).to(device=device)
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.125)
        if layer.bias is not None:
            layer.bias.normal_(mean=0.0, std=0.01)
    return layer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _consume_benchmark_output(value: object) -> None:
    if torch.is_tensor(value):
        value.detach().sum().item()
    elif isinstance(value, (tuple, list)):
        for item in value:
            _consume_benchmark_output(item)


def _time_once(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
    consume_output: bool,
) -> float:
    for _ in range(warmup):
        value = fn()
        if consume_output:
            _consume_benchmark_output(value)
    _sync(device)
    if device.type == "cuda":
        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                value = fn()
                if consume_output:
                    _consume_benchmark_output(value)
            end.record()
        _sync(device)
        return float(start.elapsed_time(end) / max(iters, 1))
    t0 = time.perf_counter()
    for _ in range(iters):
        value = fn()
        if consume_output:
            _consume_benchmark_output(value)
    return float((time.perf_counter() - t0) * 1000.0 / max(iters, 1))


def _time_median(
    fn: Callable[[], object],
    *,
    warmup: int,
    iters: int,
    repeats: int,
    device: torch.device,
    consume_output: bool = False,
) -> float:
    values = [
        _time_once(fn, warmup=warmup, iters=iters, device=device, consume_output=consume_output)
        for _ in range(max(1, repeats))
    ]
    return float(statistics.median(values))


def _clear_grads(layer: TrainableBitNetLinear, x: torch.Tensor) -> None:
    layer.weight.grad = None
    if layer.bias is not None:
        layer.bias.grad = None
    if x.grad is not None:
        x.grad = None


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


def _reset_torch_compile_cache() -> None:
    compiler = getattr(torch, "compiler", None)
    reset = getattr(compiler, "reset", None)
    if callable(reset):
        reset()
        return
    dynamo = getattr(torch, "_dynamo", None)
    reset = getattr(dynamo, "reset", None)
    if callable(reset):
        reset()


def _bitnet_optimized_training_env(mode: str, *, compile_module: bool) -> dict[str, str | None]:
    env_mode = None if mode == "dense_ste" else mode
    env: dict[str, str | None] = {
        "MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD": env_mode,
        "MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED": os.environ.get(
            "MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED",
            "1",
        ),
        "MODEL_STACK_TRAINABLE_BITNET_COMPILED_INT8_STE": "1" if compile_module and mode != "dense_ste" else None,
    }
    if mode != "dense_ste":
        env.update(
            {
                "MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE": os.environ.get(
                    "MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE",
                    "pg_h100_mlp",
                ),
                "MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT": os.environ.get(
                    "MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT",
                    "dynamic_int8_explicit_scale",
                ),
                "MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT": os.environ.get(
                    "MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT",
                    "dynamic_int8_transpose",
                ),
            }
        )
    return env


def _bitnet_training_env_with_grad_weight(
    mode: str,
    *,
    compile_module: bool,
    grad_weight_mode: str | None,
) -> dict[str, str | None]:
    env = _bitnet_optimized_training_env(mode, compile_module=compile_module)
    if mode != "dense_ste" and grad_weight_mode is not None:
        env["MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT"] = str(grad_weight_mode)
    return env


def _bitnet_shape_gate_expected_allows(
    env: dict[str, str | None],
    *,
    rows: int,
    in_features: int,
    out_features: int,
) -> bool | None:
    if env.get("MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD") is None:
        return None
    policy = str(env.get("MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE") or "").strip().lower()
    if policy in {"", "0", "false", "off", "none"}:
        return True
    if policy not in {"expansion_only", "pg_h100_expansion", "pg_h100_mlp", "h100_pg_v1", "h100_pg_v2"}:
        return True
    min_rows = int(os.environ.get("MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE_MIN_ROWS", "32768"))
    min_ratio = float(os.environ.get("MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE_MIN_OUT_IN_RATIO", "2.0"))
    if policy in {"pg_h100_mlp", "h100_pg_v2"}:
        mlp_shapes = {
            (1024, 2048),
            (2048, 1024),
            (1024, 3072),
            (3072, 1024),
        }
        return int(rows) >= min_rows and (int(in_features), int(out_features)) in mlp_shapes
    return int(rows) >= min_rows and int(out_features) >= int(math.ceil(float(in_features) * min_ratio))


def _bench_variant(
    shape: LinearShape,
    *,
    mode: str,
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    grad_out: torch.Tensor,
    rows: int,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool,
    input_grad: bool,
    compile_module: bool,
    warmup: int,
    iters: int,
    repeats: int,
    consume_output: bool,
    grad_weight_mode: str | None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    layer = _make_layer(shape, dtype=dtype, device=device, bias=bias)
    layer.load_state_dict(state_dict, strict=True)
    env = _bitnet_training_env_with_grad_weight(
        mode,
        compile_module=compile_module,
        grad_weight_mode=grad_weight_mode,
    )
    runner = layer
    if compile_module and device.type == "cuda":
        _reset_torch_compile_cache()
        with _temporary_env(env):
            runner = torch.compile(layer, dynamic=False, fullgraph=True)

    def train_step() -> None:
        _clear_grads(layer, x)
        with _temporary_env(env):
            out = runner(x)
        out.backward(grad_out)

    def forward_only() -> torch.Tensor:
        with torch.no_grad(), _temporary_env(env):
            return runner(x.detach())

    def weight_quantize_only() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return bitnet_runtime_row_quantize(layer.weight.detach(), eps=float(layer.eps))

    weight_quantize_ms = _time_median(
        weight_quantize_only,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        device=device,
        consume_output=consume_output,
    )
    forward_ms = _time_median(
        forward_only,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        device=device,
        consume_output=consume_output,
    )
    train_step_ms = _time_median(train_step, warmup=warmup, iters=iters, repeats=repeats, device=device)

    _clear_grads(layer, x)
    with _temporary_env(env):
        out = layer(x)
    out.backward(grad_out)
    tensors = {
        "out": out.detach(),
        "grad_weight": layer.weight.grad.detach().clone(),
    }
    if input_grad and x.grad is not None:
        tensors["grad_input"] = x.grad.detach().clone()
    result = {
        "mode": mode,
        "compile_module": bool(compile_module),
        "bitnet_training_env": {key: value for key, value in env.items() if value is not None},
        "bitnet_shape_gate_expected_allows": _bitnet_shape_gate_expected_allows(
            env,
            rows=int(rows),
            in_features=shape.in_features,
            out_features=shape.out_features,
        ),
        "weight_quantize_ms": weight_quantize_ms,
        "forward_ms": forward_ms,
        "train_step_ms": train_step_ms,
    }
    return result, tensors


def _bench_shape(
    shape: LinearShape,
    *,
    rows: int,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool,
    input_grad: bool,
    compile_module: bool,
    warmup: int,
    iters: int,
    repeats: int,
    consume_output: bool,
    grad_weight_mode: str | None,
) -> dict[str, object]:
    reference_layer = _make_layer(shape, dtype=dtype, device=device, bias=bias)
    state_dict = _clone_state_dict(reference_layer)
    x = torch.randn(rows, shape.in_features, device=device, dtype=dtype, requires_grad=input_grad)
    grad_out = torch.randn(rows, shape.out_features, device=device, dtype=dtype)
    dense, dense_tensors = _bench_variant(
        shape,
        mode="dense_ste",
        state_dict=state_dict,
        x=x,
        grad_out=grad_out,
        rows=rows,
        dtype=dtype,
        device=device,
        bias=bias,
        input_grad=input_grad,
        compile_module=compile_module,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        consume_output=consume_output,
        grad_weight_mode=grad_weight_mode,
    )
    variants = [dense]
    if device.type == "cuda":
        for mode in BITNET_STE_MODES:
            bitnet, bitnet_tensors = _bench_variant(
                shape,
                mode=mode,
                state_dict=state_dict,
                x=x,
                grad_out=grad_out,
                rows=rows,
                dtype=dtype,
                device=device,
                bias=bias,
                input_grad=input_grad,
                compile_module=compile_module,
                warmup=warmup,
                iters=iters,
                repeats=repeats,
                consume_output=consume_output,
                grad_weight_mode=grad_weight_mode,
            )
            bitnet["forward_speedup_vs_dense_ste"] = dense["forward_ms"] / bitnet["forward_ms"]
            bitnet["train_step_speedup_vs_dense_ste"] = dense["train_step_ms"] / bitnet["train_step_ms"]
            bitnet["out_max_abs_err_vs_dense_ste"] = float((bitnet_tensors["out"] - dense_tensors["out"]).abs().max().item())
            bitnet["grad_weight_max_abs_err_vs_dense_ste"] = float(
                (bitnet_tensors["grad_weight"] - dense_tensors["grad_weight"]).abs().max().item()
            )
            if "grad_input" in dense_tensors and "grad_input" in bitnet_tensors:
                bitnet["grad_input_max_abs_err_vs_dense_ste"] = float(
                    (bitnet_tensors["grad_input"] - dense_tensors["grad_input"]).abs().max().item()
                )
            variants.append(bitnet)

    return {
        "shape": shape.name,
        "rows": int(rows),
        "in_features": int(shape.in_features),
        "out_features": int(shape.out_features),
        "dtype": str(dtype),
        "device": str(device),
        "input_grad": bool(input_grad),
        "compile_module": bool(compile_module),
        "consume_output": bool(consume_output),
        "variants": variants,
    }


class _Relu2MlpPair(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype, device: torch.device, bias: bool) -> None:
        super().__init__()
        self.up = _make_layer(LinearShape("mlp_up_relu2", 1024, 3072), dtype=dtype, device=device, bias=bias)
        self.down = _make_layer(LinearShape("mlp_down_relu2", 3072, 1024), dtype=dtype, device=device, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.up(x)
        hidden = torch.relu(hidden).square()
        return self.down(hidden)


def _bench_relu2_mlp_pair_variant(
    *,
    mode: str,
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    grad_out: torch.Tensor,
    rows: int,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool,
    input_grad: bool,
    compile_module: bool,
    warmup: int,
    iters: int,
    repeats: int,
    consume_output: bool,
    grad_weight_mode: str | None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    mlp = _Relu2MlpPair(dtype=dtype, device=device, bias=bias)
    mlp.load_state_dict(state_dict, strict=True)
    env = _bitnet_training_env_with_grad_weight(
        mode,
        compile_module=compile_module,
        grad_weight_mode=grad_weight_mode,
    )
    runner = mlp
    if compile_module and device.type == "cuda":
        _reset_torch_compile_cache()
        with _temporary_env(env):
            runner = torch.compile(mlp, dynamic=False, fullgraph=True)

    def train_step() -> None:
        mlp.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        with _temporary_env(env):
            out = runner(x)
        out.backward(grad_out)

    def forward_only() -> torch.Tensor:
        with torch.no_grad(), _temporary_env(env):
            return runner(x.detach())

    forward_ms = _time_median(
        forward_only,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        device=device,
        consume_output=consume_output,
    )
    train_step_ms = _time_median(train_step, warmup=warmup, iters=iters, repeats=repeats, device=device)

    mlp.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None
    with _temporary_env(env):
        out = mlp(x)
    out.backward(grad_out)
    tensors = {
        "out": out.detach(),
        "up_grad_weight": mlp.up.weight.grad.detach().clone(),
        "down_grad_weight": mlp.down.weight.grad.detach().clone(),
    }
    if input_grad and x.grad is not None:
        tensors["grad_input"] = x.grad.detach().clone()
    result = {
        "mode": mode,
        "compile_module": bool(compile_module),
        "bitnet_training_env": {key: value for key, value in env.items() if value is not None},
        "bitnet_shape_gate_expected_allows": {
            "up": _bitnet_shape_gate_expected_allows(
                env,
                rows=int(rows),
                in_features=1024,
                out_features=3072,
            ),
            "down": _bitnet_shape_gate_expected_allows(
                env,
                rows=int(rows),
                in_features=3072,
                out_features=1024,
            ),
        },
        "forward_ms": forward_ms,
        "train_step_ms": train_step_ms,
    }
    return result, tensors


def _bench_relu2_mlp_pair(
    *,
    rows: int,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool,
    input_grad: bool,
    compile_module: bool,
    warmup: int,
    iters: int,
    repeats: int,
    consume_output: bool,
    grad_weight_mode: str | None,
) -> dict[str, object]:
    reference_mlp = _Relu2MlpPair(dtype=dtype, device=device, bias=bias)
    state_dict = _clone_state_dict(reference_mlp)
    x = torch.randn(rows, 1024, device=device, dtype=dtype, requires_grad=input_grad)
    grad_out = torch.randn(rows, 1024, device=device, dtype=dtype)
    dense, dense_tensors = _bench_relu2_mlp_pair_variant(
        mode="dense_ste",
        state_dict=state_dict,
        x=x,
        grad_out=grad_out,
        rows=rows,
        dtype=dtype,
        device=device,
        bias=bias,
        input_grad=input_grad,
        compile_module=compile_module,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        consume_output=consume_output,
        grad_weight_mode=grad_weight_mode,
    )
    variants = [dense]
    if device.type == "cuda":
        for mode in BITNET_STE_MODES:
            bitnet, bitnet_tensors = _bench_relu2_mlp_pair_variant(
                mode=mode,
                state_dict=state_dict,
                x=x,
                grad_out=grad_out,
                rows=rows,
                dtype=dtype,
                device=device,
                bias=bias,
                input_grad=input_grad,
                compile_module=compile_module,
                warmup=warmup,
                iters=iters,
                repeats=repeats,
                consume_output=consume_output,
                grad_weight_mode=grad_weight_mode,
            )
            bitnet["forward_speedup_vs_dense_ste"] = dense["forward_ms"] / bitnet["forward_ms"]
            bitnet["train_step_speedup_vs_dense_ste"] = dense["train_step_ms"] / bitnet["train_step_ms"]
            bitnet["out_max_abs_err_vs_dense_ste"] = float((bitnet_tensors["out"] - dense_tensors["out"]).abs().max().item())
            bitnet["up_grad_weight_max_abs_err_vs_dense_ste"] = float(
                (bitnet_tensors["up_grad_weight"] - dense_tensors["up_grad_weight"]).abs().max().item()
            )
            bitnet["down_grad_weight_max_abs_err_vs_dense_ste"] = float(
                (bitnet_tensors["down_grad_weight"] - dense_tensors["down_grad_weight"]).abs().max().item()
            )
            if "grad_input" in dense_tensors and "grad_input" in bitnet_tensors:
                bitnet["grad_input_max_abs_err_vs_dense_ste"] = float(
                    (bitnet_tensors["grad_input"] - dense_tensors["grad_input"]).abs().max().item()
                )
            variants.append(bitnet)

    return {
        "shape": "relu2_mlp_pair_1024_3072_1024",
        "rows": int(rows),
        "in_features": 1024,
        "hidden_features": 3072,
        "out_features": 1024,
        "dtype": str(dtype),
        "device": str(device),
        "input_grad": bool(input_grad),
        "compile_module": bool(compile_module),
        "consume_output": bool(consume_output),
        "variants": variants,
    }


class _RMSNorm(torch.nn.Module):
    def __init__(self, eps: float | None = None) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class _Rotary(torch.nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


def _attention_repeat_kv_mode() -> str:
    return os.getenv("MODEL_STACK_ATTENTION_REPEAT_KV", "").strip().lower()


def _maybe_repeat_kv_heads(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if num_kv_heads == num_heads:
        return k, v, False
    mode = _attention_repeat_kv_mode()
    if mode in {"", "0", "false", "off", "none", "gqa"}:
        return k, v, True
    groups = int(num_heads) // int(num_kv_heads)
    if mode in {"expand", "expand_reshape"}:
        bsz, kv_heads, seq_len, head_dim = k.shape
        k = k[:, :, None, :, :].expand(bsz, kv_heads, groups, seq_len, head_dim).reshape(
            bsz,
            kv_heads * groups,
            seq_len,
            head_dim,
        )
        v = v[:, :, None, :, :].expand(bsz, kv_heads, groups, seq_len, head_dim).reshape(
            bsz,
            kv_heads * groups,
            seq_len,
            head_dim,
        )
    else:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    return k, v, False


class _PGCausalSelfAttention(torch.nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        dtype: torch.dtype,
        device: torch.device,
        bias: bool,
        fused_qkv: bool,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(dim) // int(num_heads)
        self.fused_qkv = bool(fused_qkv)
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = _make_layer(LinearShape("attn_q", dim, dim), dtype=dtype, device=device, bias=bias)
        self.c_k = _make_layer(LinearShape("attn_k", dim, kv_dim), dtype=dtype, device=device, bias=bias)
        self.c_v = _make_layer(LinearShape("attn_v", dim, kv_dim), dtype=dtype, device=device, bias=bias)
        self.proj = _make_layer(LinearShape("attn_out", dim, dim), dtype=dtype, device=device, bias=bias)
        self.q_gain = torch.nn.Parameter(torch.full((num_heads,), qk_gain_init, device=device, dtype=torch.float32))
        self.rotary = _Rotary(self.head_dim, base=rope_base).to(device=device)

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kv_dim = self.num_kv_heads * self.head_dim
        use_fused = (
            self.fused_qkv
            and x.is_cuda
            and self.c_q.bias is None
            and self.c_k.bias is None
            and self.c_v.bias is None
            and _trainable_bitnet_dynamic_mode_enabled()
        )
        if use_fused:
            fused = _TrainableBitNetFusedQKVSTEFunction.apply(
                x,
                self.c_q.weight,
                self.c_k.weight,
                self.c_v.weight,
                float(self.c_q.eps),
                float(os.getenv("MODEL_STACK_TRAINABLE_BITNET_ACT_QUANT_PERCENTILE", "0.999")),
            )
            q_raw, k_raw, v_raw = torch.split(fused, (self.c_q.out_features, kv_dim, kv_dim), dim=-1)
            return q_raw, k_raw, v_raw
        return self.c_q(x), self.c_k(x), self.c_v(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        q_raw, k_raw, v_raw = self._project_qkv(x)
        q = q_raw.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_raw.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v_raw.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        k, v, enable_gqa = _maybe_repeat_kv_heads(
            k,
            v,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=enable_gqa,
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class _PGMLP(torch.nn.Module):
    def __init__(self, *, dim: int, mlp_mult: int, dtype: torch.dtype, device: torch.device, bias: bool) -> None:
        super().__init__()
        hidden = int(mlp_mult) * int(dim)
        self.fc = _make_layer(LinearShape("mlp_up_relu2", dim, hidden), dtype=dtype, device=device, bias=bias)
        self.proj = _make_layer(LinearShape("mlp_down_relu2", hidden, dim), dtype=dtype, device=device, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class _PGBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        dtype: torch.dtype,
        device: torch.device,
        bias: bool,
        fused_qkv: bool,
    ) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm()
        self.mlp_norm = _RMSNorm()
        self.attn = _PGCausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            rope_base=rope_base,
            qk_gain_init=qk_gain_init,
            dtype=dtype,
            device=device,
            bias=bias,
            fused_qkv=fused_qkv,
        )
        self.mlp = _PGMLP(dim=dim, mlp_mult=mlp_mult, dtype=dtype, device=device, bias=bias)
        self.attn_scale = torch.nn.Parameter(torch.ones(dim, device=device, dtype=torch.float32))
        self.mlp_scale = torch.nn.Parameter(torch.ones(dim, device=device, dtype=torch.float32))
        self.resid_mix = torch.nn.Parameter(
            torch.stack((torch.ones(dim, device=device), torch.zeros(dim, device=device))).float()
        )

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


def _make_pg_block(
    *,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: int,
    rope_base: float,
    qk_gain_init: float,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool,
    fused_qkv: bool,
) -> _PGBlock:
    return _PGBlock(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        rope_base=rope_base,
        qk_gain_init=qk_gain_init,
        dtype=dtype,
        device=device,
        bias=bias,
        fused_qkv=fused_qkv,
    )


def _clear_module_and_input_grads(module: torch.nn.Module, *inputs: torch.Tensor) -> None:
    module.zero_grad(set_to_none=True)
    for tensor in inputs:
        if tensor.grad is not None:
            tensor.grad = None


def _param_grad_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.grad.detach().clone()
        for name, param in module.named_parameters()
        if param.grad is not None and param.ndim >= 2
    }


def _max_abs_err_by_key(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor]) -> float:
    max_err = 0.0
    for key, lhs_tensor in lhs.items():
        rhs_tensor = rhs.get(key)
        if rhs_tensor is None:
            continue
        err = float((lhs_tensor - rhs_tensor).abs().max().item())
        max_err = max(max_err, err)
    return max_err


def _autocast_context(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


def _trainable_bitnet_dynamic_mode_enabled() -> bool:
    return os.getenv("MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD", "dense_ste").strip().lower() in {
        "int8_ste",
        "dynamic_int8_ste",
        "dynamic_int4_ste",
        "w2a8_ste",
        "w2a4_ste",
    }


class _TrainableBitNetFusedQKVSTEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        eps: float,
        act_quant_percentile: float,
    ) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError("fused QKV int8 STE forward requires CUDA")
        target_dtype = x.dtype if x.dtype.is_floating_point else q_weight.dtype
        x_local = x.to(dtype=target_dtype)
        q_qweight, q_scale = _bitnet_runtime_row_codes_and_scale(q_weight.detach(), eps=float(eps))
        k_qweight, k_scale = _bitnet_runtime_row_codes_and_scale(k_weight.detach(), eps=float(eps))
        v_qweight, v_scale = _bitnet_runtime_row_codes_and_scale(v_weight.detach(), eps=float(eps))
        qweight = torch.cat((q_qweight, k_qweight, v_qweight), dim=0).to(
            device=x_local.device,
            dtype=torch.int8,
        ).contiguous()
        row_scale = torch.cat((q_scale, k_scale, v_scale), dim=0).to(
            device=x_local.device,
            dtype=torch.float32,
        ).contiguous()
        out = runtime_bitnet_int8_linear_from_float(
            x_local,
            qweight,
            row_scale,
            None,
            pre_scale=None,
            act_quant_mode="dynamic_int8",
            act_scale=None,
            act_quant_bits=8,
            act_quant_method="absmax",
            act_quant_percentile=float(act_quant_percentile),
        )
        if ctx.needs_input_grad[0]:
            dequant_weight = qweight.to(dtype=target_dtype).mul(row_scale.to(dtype=target_dtype).unsqueeze(-1))
            ctx.save_for_backward(x_local, dequant_weight)
            ctx.has_dequant_weight = True
        else:
            ctx.save_for_backward(x_local)
            ctx.has_dequant_weight = False
        ctx.input_shape = tuple(x.shape)
        ctx.q_size = int(q_weight.shape[0])
        ctx.k_size = int(k_weight.shape[0])
        ctx.v_size = int(v_weight.shape[0])
        ctx.q_dtype = q_weight.dtype
        ctx.k_dtype = k_weight.dtype
        ctx.v_dtype = v_weight.dtype
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        x = saved[0]
        dequant_weight = saved[1] if getattr(ctx, "has_dequant_weight", False) else None
        grad_out_2d = grad_output.reshape(-1, grad_output.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])

        grad_input = None
        if ctx.needs_input_grad[0]:
            if dequant_weight is None:
                raise RuntimeError("fused QKV int8 STE backward missing dequantized weight")
            grad_input = grad_out_2d.to(dtype=dequant_weight.dtype).matmul(dequant_weight)
            grad_input = grad_input.reshape(ctx.input_shape).to(dtype=x.dtype)

        grad_fused_weight = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            grad_fused_weight = grad_out_2d.to(dtype=x.dtype).t().matmul(x_2d)

        grad_q_weight = grad_k_weight = grad_v_weight = None
        if grad_fused_weight is not None:
            q_end = ctx.q_size
            k_end = q_end + ctx.k_size
            if ctx.needs_input_grad[1]:
                grad_q_weight = grad_fused_weight[:q_end].to(dtype=ctx.q_dtype)
            if ctx.needs_input_grad[2]:
                grad_k_weight = grad_fused_weight[q_end:k_end].to(dtype=ctx.k_dtype)
            if ctx.needs_input_grad[3]:
                grad_v_weight = grad_fused_weight[k_end : k_end + ctx.v_size].to(dtype=ctx.v_dtype)
        return grad_input, grad_q_weight, grad_k_weight, grad_v_weight, None, None


def _bench_pg_block_variant(
    *,
    mode: str,
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    x0: torch.Tensor,
    grad_out: torch.Tensor,
    batch_size: int,
    seq_len: int,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: int,
    rope_base: float,
    qk_gain_init: float,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool,
    fused_qkv: bool,
    input_grad: bool,
    compile_module: bool,
    warmup: int,
    iters: int,
    repeats: int,
    consume_output: bool,
    grad_weight_mode: str | None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    rows = int(batch_size) * int(seq_len)
    block = _make_pg_block(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        rope_base=rope_base,
        qk_gain_init=qk_gain_init,
        dtype=dtype,
        device=device,
        bias=bias,
        fused_qkv=fused_qkv,
    )
    block.load_state_dict(state_dict, strict=True)
    env = _bitnet_training_env_with_grad_weight(
        mode,
        compile_module=compile_module,
        grad_weight_mode=grad_weight_mode,
    )
    runner = block
    if compile_module and device.type == "cuda":
        _reset_torch_compile_cache()
        with _temporary_env(env):
            runner = torch.compile(block, dynamic=False, fullgraph=True)

    def train_step() -> None:
        _clear_module_and_input_grads(block, x, x0)
        with _temporary_env(env), _autocast_context(device, dtype):
            out = runner(x, x0)
        out.backward(grad_out)

    def forward_only() -> torch.Tensor:
        with torch.no_grad(), _temporary_env(env), _autocast_context(device, dtype):
            return runner(x.detach(), x0.detach())

    forward_ms = _time_median(
        forward_only,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        device=device,
        consume_output=consume_output,
    )
    train_step_ms = _time_median(train_step, warmup=warmup, iters=iters, repeats=repeats, device=device)

    _clear_module_and_input_grads(block, x, x0)
    with _temporary_env(env), _autocast_context(device, dtype):
        out = block(x, x0)
    out.backward(grad_out)
    tensors = {
        "out": out.detach(),
        "param_grads": _param_grad_tensors(block),
    }
    if input_grad and x.grad is not None:
        tensors["grad_input"] = x.grad.detach().clone()
    if input_grad and x0.grad is not None:
        tensors["grad_x0"] = x0.grad.detach().clone()
    result = {
        "mode": mode,
        "compile_module": bool(compile_module),
        "bitnet_training_env": {key: value for key, value in env.items() if value is not None},
        "bitnet_shape_gate_expected_allows": {
            "mlp_up": _bitnet_shape_gate_expected_allows(
                env,
                rows=rows,
                in_features=dim,
                out_features=dim * mlp_mult,
            ),
            "mlp_down": _bitnet_shape_gate_expected_allows(
                env,
                rows=rows,
                in_features=dim * mlp_mult,
                out_features=dim,
            ),
            "attention": _bitnet_shape_gate_expected_allows(
                env,
                rows=rows,
                in_features=dim,
                out_features=dim,
            ),
        },
        "forward_ms": forward_ms,
        "train_step_ms": train_step_ms,
    }
    return result, tensors


def _bench_pg_block(
    *,
    batch_size: int,
    seq_len: int,
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    mlp_mult: int,
    rope_base: float,
    qk_gain_init: float,
    dtype: torch.dtype,
    device: torch.device,
    bias: bool,
    fused_qkv: bool,
    input_grad: bool,
    compile_module: bool,
    warmup: int,
    iters: int,
    repeats: int,
    consume_output: bool,
    grad_weight_mode: str | None,
) -> dict[str, object]:
    reference_block = _make_pg_block(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        rope_base=rope_base,
        qk_gain_init=qk_gain_init,
        dtype=dtype,
        device=device,
        bias=bias,
        fused_qkv=fused_qkv,
    )
    state_dict = _clone_state_dict(reference_block)
    x = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=input_grad)
    x0 = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype, requires_grad=input_grad)
    grad_out = torch.randn(batch_size, seq_len, dim, device=device, dtype=dtype)
    dense, dense_tensors = _bench_pg_block_variant(
        mode="dense_ste",
        state_dict=state_dict,
        x=x,
        x0=x0,
        grad_out=grad_out,
        batch_size=batch_size,
        seq_len=seq_len,
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        mlp_mult=mlp_mult,
        rope_base=rope_base,
        qk_gain_init=qk_gain_init,
        dtype=dtype,
        device=device,
        bias=bias,
        fused_qkv=fused_qkv,
        input_grad=input_grad,
        compile_module=compile_module,
        warmup=warmup,
        iters=iters,
        repeats=repeats,
        consume_output=consume_output,
        grad_weight_mode=grad_weight_mode,
    )
    variants = [dense]
    if device.type == "cuda":
        for mode in BITNET_STE_MODES:
            bitnet, bitnet_tensors = _bench_pg_block_variant(
                mode=mode,
                state_dict=state_dict,
                x=x,
                x0=x0,
                grad_out=grad_out,
                batch_size=batch_size,
                seq_len=seq_len,
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_mult=mlp_mult,
                rope_base=rope_base,
                qk_gain_init=qk_gain_init,
                dtype=dtype,
                device=device,
                bias=bias,
                fused_qkv=fused_qkv,
                input_grad=input_grad,
                compile_module=compile_module,
                warmup=warmup,
                iters=iters,
                repeats=repeats,
                consume_output=consume_output,
                grad_weight_mode=grad_weight_mode,
            )
            bitnet["forward_speedup_vs_dense_ste"] = dense["forward_ms"] / bitnet["forward_ms"]
            bitnet["train_step_speedup_vs_dense_ste"] = dense["train_step_ms"] / bitnet["train_step_ms"]
            bitnet["out_max_abs_err_vs_dense_ste"] = float((bitnet_tensors["out"] - dense_tensors["out"]).abs().max().item())
            bitnet["param_grad_max_abs_err_vs_dense_ste"] = _max_abs_err_by_key(
                bitnet_tensors["param_grads"],
                dense_tensors["param_grads"],
            )
            if "grad_input" in dense_tensors and "grad_input" in bitnet_tensors:
                bitnet["grad_input_max_abs_err_vs_dense_ste"] = float(
                    (bitnet_tensors["grad_input"] - dense_tensors["grad_input"]).abs().max().item()
                )
            if "grad_x0" in dense_tensors and "grad_x0" in bitnet_tensors:
                bitnet["grad_x0_max_abs_err_vs_dense_ste"] = float(
                    (bitnet_tensors["grad_x0"] - dense_tensors["grad_x0"]).abs().max().item()
                )
            variants.append(bitnet)

    return {
        "shape": f"pg_block_d{dim}_h{num_heads}_kv{num_kv_heads}_mlp{mlp_mult}",
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "tokens": int(batch_size) * int(seq_len),
        "dim": int(dim),
        "num_heads": int(num_heads),
        "num_kv_heads": int(num_kv_heads),
        "mlp_mult": int(mlp_mult),
        "dtype": str(dtype),
        "device": str(device),
        "input_grad": bool(input_grad),
        "compile_module": bool(compile_module),
        "fused_qkv": bool(fused_qkv),
        "consume_output": bool(consume_output),
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TrainableBitNetLinear forward+backward on PG shapes.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="runtime_row_1024x7_relu2_mlp3")
    parser.add_argument("--shape", action="append", default=[], help="Custom shape NAME:IN:OUT or IN:OUT")
    parser.add_argument("--no-preset-shapes", action="store_true", help="Benchmark only --shape entries.")
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--no-input-grad", action="store_true")
    parser.add_argument("--compile-module", action="store_true")
    parser.add_argument("--include-relu2-mlp-pair", action="store_true")
    parser.add_argument("--include-pg-block", action="store_true")
    parser.add_argument("--block-batch-size", type=int, default=2)
    parser.add_argument("--block-seq-len", type=int, default=4096)
    parser.add_argument("--block-dim", type=int, default=1024)
    parser.add_argument("--block-num-heads", type=int, default=16)
    parser.add_argument("--block-num-kv-heads", type=int, default=4)
    parser.add_argument("--block-mlp-mult", type=int, default=2)
    parser.add_argument("--block-rope-base", type=float, default=10000.0)
    parser.add_argument("--block-qk-gain-init", type=float, default=1.5)
    parser.add_argument("--block-fused-qkv", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--consume-output", action="store_true", help="Synchronously consume forward/quantize outputs while timing.")
    parser.add_argument(
        "--grad-weight-mode",
        default=None,
        help="Override MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT for BitNet variants.",
    )
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

        enable_cudnn_sdp(False)
        enable_flash_sdp(True)
        enable_mem_efficient_sdp(False)
        enable_math_sdp(False)
    shapes = ([] if args.no_preset_shapes else list(PRESETS[args.preset])) + _parse_shapes(args.shape)
    if not shapes:
        raise ValueError("no shapes selected; use a preset or pass --shape")
    header = {
        "header": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "dtype": str(dtype),
            "preset": args.preset,
            "compile_module": bool(args.compile_module),
            "training_forward_env": "MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD",
            "has_native_runtime_row_quantize": has_native_op("bitnet_runtime_row_quantize"),
            "has_cuda_runtime_row_quantize": "bitnet_runtime_row_quantize" in set(cuda_kernel_ops()),
        }
    }
    if args.jsonl:
        print(json.dumps(header, sort_keys=True))
    else:
        print(header)
    for shape in shapes:
        result = _bench_shape(
            shape,
            rows=args.rows,
            dtype=dtype,
            device=device,
            bias=bool(args.bias),
            input_grad=not bool(args.no_input_grad),
            compile_module=bool(args.compile_module),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            consume_output=bool(args.consume_output),
            grad_weight_mode=args.grad_weight_mode,
        )
        if args.jsonl:
            print(json.dumps(result, sort_keys=True))
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
    if args.include_pg_block:
        result = _bench_pg_block(
            batch_size=args.block_batch_size,
            seq_len=args.block_seq_len,
            dim=args.block_dim,
            num_heads=args.block_num_heads,
            num_kv_heads=args.block_num_kv_heads,
            mlp_mult=args.block_mlp_mult,
            rope_base=args.block_rope_base,
            qk_gain_init=args.block_qk_gain_init,
            dtype=dtype,
            device=device,
            bias=bool(args.bias),
            fused_qkv=bool(args.block_fused_qkv),
            input_grad=not bool(args.no_input_grad),
            compile_module=bool(args.compile_module),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            consume_output=bool(args.consume_output),
            grad_weight_mode=args.grad_weight_mode,
        )
        if args.jsonl:
            print(json.dumps(result, sort_keys=True))
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
    if args.include_relu2_mlp_pair:
        result = _bench_relu2_mlp_pair(
            rows=args.rows,
            dtype=dtype,
            device=device,
            bias=bool(args.bias),
            input_grad=not bool(args.no_input_grad),
            compile_module=bool(args.compile_module),
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            consume_output=bool(args.consume_output),
            grad_weight_mode=args.grad_weight_mode,
        )
        if args.jsonl:
            print(json.dumps(result, sort_keys=True))
        else:
            print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
