from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH_PATH = Path(__file__).with_name("bench_pg_bitnet_training_step.py")
spec = importlib.util.spec_from_file_location("pg_bitnet_training_step", BENCH_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {BENCH_PATH}")
pg_bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pg_bench
spec.loader.exec_module(pg_bench)


def _dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_once(fn: Callable[[], None], *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        _sync(device)
        return float(start.elapsed_time(end) / max(iters, 1))
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return float((time.perf_counter() - t0) * 1000.0 / max(iters, 1))


def _time_median(fn: Callable[[], None], *, warmup: int, iters: int, repeats: int, device: torch.device) -> float:
    values = [_time_once(fn, warmup=warmup, iters=iters, device=device) for _ in range(max(1, repeats))]
    return float(statistics.median(values))


def _env_for_mode(mode: str, compile_module: bool) -> dict[str, str | None]:
    return {
        "MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD": None if mode == "dense_ste" else mode,
        "MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED": os.environ.get(
            "MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED",
            "1",
        ),
        "MODEL_STACK_TRAINABLE_BITNET_COMPILED_INT8_STE": "1" if compile_module and mode != "dense_ste" else None,
    }


def _clone_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _load_runner(
    module: torch.nn.Module,
    *,
    state_dict: dict[str, torch.Tensor],
    env: dict[str, str | None],
    compile_module: bool,
    device: torch.device,
) -> torch.nn.Module:
    module.load_state_dict(state_dict, strict=True)
    if compile_module and device.type == "cuda":
        pg_bench._reset_torch_compile_cache()
        with pg_bench._temporary_env(env):
            return torch.compile(module, dynamic=False, fullgraph=True)
    return module


def _bench_component(
    name: str,
    module: torch.nn.Module,
    *,
    state_dict: dict[str, torch.Tensor],
    args_tuple: tuple[torch.Tensor, ...],
    grad_out: torch.Tensor,
    mode: str,
    dtype: torch.dtype,
    device: torch.device,
    compile_module: bool,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, object]:
    env = _env_for_mode(mode, compile_module)
    runner = _load_runner(module, state_dict=state_dict, env=env, compile_module=compile_module, device=device)

    def clear_grads() -> None:
        module.zero_grad(set_to_none=True)
        for item in args_tuple:
            if item.grad is not None:
                item.grad = None

    def forward_only() -> None:
        with torch.no_grad(), pg_bench._temporary_env(env), pg_bench._autocast_context(device, dtype):
            runner(*tuple(item.detach() for item in args_tuple))

    def train_step() -> None:
        clear_grads()
        with pg_bench._temporary_env(env), pg_bench._autocast_context(device, dtype):
            out = runner(*args_tuple)
        out.backward(grad_out)

    forward_ms = _time_median(forward_only, warmup=warmup, iters=iters, repeats=repeats, device=device)
    train_step_ms = _time_median(train_step, warmup=warmup, iters=iters, repeats=repeats, device=device)
    return {
        "component": name,
        "mode": mode,
        "compile_module": bool(compile_module),
        "forward_ms": forward_ms,
        "train_step_ms": train_step_ms,
        "backward_plus_overhead_ms": max(train_step_ms - forward_ms, 0.0),
    }


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x /= x.norm() + eps
    transposed = g.size(0) > g.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        xx_t = x @ x.T
        b_xx = torch.addmm(xx_t, xx_t, xx_t, beta=b, alpha=c)
        x = torch.addmm(x, b_xx, x, beta=a, alpha=1.0)
    return x.T if transposed else x


def _zeropower_via_newtonschulz5_batched(g: torch.Tensor, steps: int, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    x = x / x.flatten(1).norm(dim=1).clamp_min(eps).view(-1, 1, 1)
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        x = x.transpose(-2, -1)
    for _ in range(steps):
        xx_t = x @ x.transpose(-2, -1)
        b_xx = torch.baddbmm(xx_t, xx_t, xx_t, beta=b, alpha=c)
        x = torch.baddbmm(x, b_xx, x, beta=a, alpha=1.0)
    return x.transpose(-2, -1) if transposed else x


def _prepare_block_grads(
    module: torch.nn.Module,
    *,
    state_dict: dict[str, torch.Tensor],
    args_tuple: tuple[torch.Tensor, ...],
    grad_out: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.nn.Parameter]:
    module.load_state_dict(state_dict, strict=True)
    module.zero_grad(set_to_none=True)
    with pg_bench._autocast_context(device, dtype):
        out = module(*args_tuple)
    out.backward(grad_out)
    return [p for p in module.parameters() if p.ndim == 2 and p.grad is not None]


def _bench_muon_optimizer_variants(
    module: torch.nn.Module,
    *,
    state_dict: dict[str, torch.Tensor],
    args_tuple: tuple[torch.Tensor, ...],
    grad_out: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    backend_steps: int,
    optimizer_blocks: int,
    batched_min_bucket: int,
    row_norm: bool,
    warmup: int,
    iters: int,
    repeats: int,
) -> list[dict[str, object]]:
    matrix_params = _prepare_block_grads(
        module,
        state_dict=state_dict,
        args_tuple=args_tuple,
        grad_out=grad_out,
        dtype=dtype,
        device=device,
    )
    if optimizer_blocks > 1:
        base_params = matrix_params
        matrix_params = []
        for _ in range(optimizer_blocks):
            for p in base_params:
                q = torch.nn.Parameter(torch.randn_like(p))
                q.grad = torch.randn_like(p)
                matrix_params.append(q)
    momentum = [torch.zeros_like(p.grad) for p in matrix_params]
    total_params = sum(int(p.numel()) for p in matrix_params)
    cached_updates_flat = torch.empty(total_params, device=device, dtype=torch.bfloat16)

    zeropower = _zeropower_via_newtonschulz5
    zeropower_batched = _zeropower_via_newtonschulz5_batched
    if device.type == "cuda":
        zeropower = torch.compile(zeropower, dynamic=False)
        zeropower_batched = torch.compile(zeropower_batched, dynamic=False)

    def compute_update(idx: int, p: torch.nn.Parameter) -> torch.Tensor:
        g = p.grad
        if g is None:
            raise RuntimeError("Muon optimizer benchmark expected gradients")
        buf = momentum[idx]
        buf.mul_(0.95).add_(g)
        update = g.add(buf, alpha=0.95)
        if row_norm:
            update = F.rms_norm(update.float(), (update.size(-1),)).bfloat16()
        update = zeropower(update, steps=backend_steps)
        update *= max(1, update.size(0) / update.size(1)) ** 0.5
        return update

    def compute_updates_batched() -> list[torch.Tensor]:
        buckets: dict[tuple[tuple[int, ...], torch.dtype, torch.device], list[tuple[int, torch.nn.Parameter]]] = {}
        for idx, p in enumerate(matrix_params):
            if p.grad is None:
                raise RuntimeError("Muon optimizer benchmark expected gradients")
            buckets.setdefault((tuple(p.grad.shape), p.grad.dtype, p.grad.device), []).append((idx, p))

        if not buckets:
            return [compute_update(idx, p) for idx, p in enumerate(matrix_params)]

        updates: list[torch.Tensor | None] = [None] * len(matrix_params)
        for bucket in buckets.values():
            if len(bucket) < batched_min_bucket:
                for idx, p in bucket:
                    updates[idx] = compute_update(idx, p)
                continue
            grad_tensors = []
            for _, p in bucket:
                if p.grad is None:
                    raise RuntimeError("Muon optimizer benchmark expected gradients")
                grad_tensors.append(p.grad)
            bufs = [momentum[idx] for idx, _ in bucket]
            torch._foreach_mul_(bufs, 0.95)
            torch._foreach_add_(bufs, grad_tensors)
            update_inputs = torch._foreach_add(grad_tensors, bufs, alpha=0.95)
            if len(update_inputs) == 1:
                if row_norm:
                    update_inputs[0] = F.rms_norm(update_inputs[0].float(), (update_inputs[0].size(-1),)).bfloat16()
                update = zeropower(update_inputs[0], steps=backend_steps)
                update *= max(1, update.size(0) / update.size(1)) ** 0.5
                updates[bucket[0][0]] = update
                continue
            update_stack = torch.stack(update_inputs, dim=0)
            if row_norm:
                update_stack = F.rms_norm(update_stack.float(), (update_stack.size(-1),)).bfloat16()
            update_stack = zeropower_batched(update_stack, steps=backend_steps)
            update_stack *= max(1, update_stack.size(-2) / update_stack.size(-1)) ** 0.5
            for j, (idx, _) in enumerate(bucket):
                updates[idx] = update_stack[j]
        if any(update is None for update in updates):
            raise RuntimeError("Muon optimizer benchmark missed a matrix update")
        return [update for update in updates if update is not None]

    @torch.no_grad()
    def current_flat_alloc_step() -> None:
        updates_flat = torch.zeros(total_params, device=device, dtype=torch.bfloat16)
        curr = 0
        for idx, p in enumerate(matrix_params):
            update = compute_update(idx, p)
            updates_flat[curr : curr + p.numel()] = update.reshape(-1)
            curr += p.numel()
        curr = 0
        for p in matrix_params:
            update = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
            p.add_(update, alpha=-0.04)
            curr += p.numel()

    @torch.no_grad()
    def cached_flat_step() -> None:
        cached_updates_flat.zero_()
        curr = 0
        for idx, p in enumerate(matrix_params):
            update = compute_update(idx, p)
            cached_updates_flat[curr : curr + p.numel()] = update.reshape(-1)
            curr += p.numel()
        curr = 0
        for p in matrix_params:
            update = cached_updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
            p.add_(update, alpha=-0.04)
            curr += p.numel()

    @torch.no_grad()
    def direct_single_rank_step() -> None:
        for idx, p in enumerate(matrix_params):
            update = compute_update(idx, p)
            p.add_(update.to(dtype=p.dtype), alpha=-0.04)

    @torch.no_grad()
    def batched_direct_single_rank_step() -> None:
        updates = compute_updates_batched()
        buckets: dict[tuple[tuple[int, ...], torch.dtype, torch.device], list[tuple[torch.nn.Parameter, torch.Tensor]]] = {}
        for p, update in zip(matrix_params, updates, strict=True):
            buckets.setdefault((tuple(p.shape), p.dtype, p.device), []).append((p, update.to(dtype=p.dtype)))
        for bucket in buckets.values():
            torch._foreach_add_([p for p, _ in bucket], [update for _, update in bucket], alpha=-0.04)

    variants = []
    for name, fn in (
        ("current_flat_alloc", current_flat_alloc_step),
        ("cached_flat", cached_flat_step),
        ("direct_single_rank", direct_single_rank_step),
        ("batched_direct_single_rank", batched_direct_single_rank_step),
    ):
        variants.append(
            {
                "component": "muon_matrix_optimizer",
                "variant": name,
                "matrix_params": len(matrix_params),
                "total_params": int(total_params),
                "backend_steps": int(backend_steps),
                "optimizer_blocks": int(optimizer_blocks),
                "batched_min_bucket": int(batched_min_bucket),
                "row_norm": bool(row_norm),
                "step_ms": _time_median(fn, warmup=warmup, iters=iters, repeats=repeats, device=device),
            }
        )
    baseline = variants[0]["step_ms"]
    for item in variants[1:]:
        item["speedup_vs_current_flat_alloc"] = baseline / item["step_ms"]
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile PG block components for training-side bottlenecks.")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--qk-gain-init", type=float, default=1.5)
    parser.add_argument("--compile-module", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--muon-backend-steps", type=int, default=5)
    parser.add_argument("--optimizer-blocks", type=int, default=1)
    parser.add_argument("--muon-batched-min-bucket", type=int, default=2)
    parser.add_argument(
        "--muon-row-norm",
        action="store_true",
        default=os.environ.get("MODEL_STACK_MUON_ROW_NORM", "1").strip().lower()
        not in {"", "0", "false", "off", "no"},
    )
    parser.add_argument("--optimizer-only", action="store_true")
    parser.add_argument("--jsonl", action="store_true")
    args = parser.parse_args()

    dtype = _dtype(args.dtype)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(1337)

    x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device, dtype=dtype, requires_grad=True)
    x0 = torch.randn(args.batch_size, args.seq_len, args.dim, device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn_like(x)
    common = {
        "dim": args.dim,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "mlp_mult": args.mlp_mult,
        "rope_base": args.rope_base,
        "qk_gain_init": args.qk_gain_init,
        "dtype": dtype,
        "device": device,
        "bias": False,
    }
    attn = pg_bench._PGCausalSelfAttention(
        dim=args.dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        dtype=dtype,
        device=device,
        bias=False,
        fused_qkv=False,
    )
    mlp = pg_bench._PGMLP(dim=args.dim, mlp_mult=args.mlp_mult, dtype=dtype, device=device, bias=False)
    block = pg_bench._make_pg_block(fused_qkv=False, **common)
    modules = [
        ("attention", attn, (x,), grad),
        ("mlp", mlp, (x,), grad),
        ("block", block, (x, x0), grad),
    ]

    header = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(dtype),
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "tokens": int(args.batch_size) * int(args.seq_len),
        "dim": int(args.dim),
        "compile_module": bool(args.compile_module),
        "shape_gate": os.environ.get("MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE", ""),
        "grad_input_mode": os.environ.get("MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT", ""),
        "grad_weight_mode": os.environ.get("MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT", ""),
        "attention_repeat_kv": os.environ.get("MODEL_STACK_ATTENTION_REPEAT_KV", ""),
        "optimizer_blocks": int(args.optimizer_blocks),
        "muon_batched_min_bucket": int(args.muon_batched_min_bucket),
        "muon_row_norm": bool(args.muon_row_norm),
    }
    if args.jsonl:
        print(json.dumps({"header": header}))
    else:
        print(header)

    if not args.optimizer_only:
        for name, module, inputs, grad_out in modules:
            state = _clone_state_dict(module)
            dense = _bench_component(
                name,
                module,
                state_dict=state,
                args_tuple=inputs,
                grad_out=grad_out,
                mode="dense_ste",
                dtype=dtype,
                device=device,
                compile_module=args.compile_module,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
            )
            variants = [dense]
            for mode in pg_bench.BITNET_STE_MODES:
                bitnet = _bench_component(
                    name,
                    module,
                    state_dict=state,
                    args_tuple=inputs,
                    grad_out=grad_out,
                    mode=mode,
                    dtype=dtype,
                    device=device,
                    compile_module=args.compile_module,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
                bitnet["forward_speedup_vs_dense_ste"] = dense["forward_ms"] / bitnet["forward_ms"]
                bitnet["train_step_speedup_vs_dense_ste"] = dense["train_step_ms"] / bitnet["train_step_ms"]
                variants.append(bitnet)
            result = {"component": name, "variants": variants}
            print(json.dumps(result) if args.jsonl else result)

    optimizer_result = {
        "component": "muon_matrix_optimizer",
        "variants": _bench_muon_optimizer_variants(
            block,
            state_dict=_clone_state_dict(block),
            args_tuple=(x, x0),
            grad_out=grad,
            dtype=dtype,
            device=device,
            backend_steps=args.muon_backend_steps,
            optimizer_blocks=args.optimizer_blocks,
            batched_min_bucket=args.muon_batched_min_bucket,
            row_norm=args.muon_row_norm,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
        ),
    }
    print(json.dumps(optimizer_result) if args.jsonl else optimizer_result)


if __name__ == "__main__":
    main()
