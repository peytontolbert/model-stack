from __future__ import annotations

import argparse
import importlib
import json
from typing import Any, Dict, List

from .spaces import Choice, IntRange, LogUniform, SearchSpace, Uniform
from .study import Study, StudyConfig
from .algorithms.random import RandomSearch
from .algorithms.grid import GridSearch
from .algorithms.sobol import SobolSearch
from .algorithms.lhs import LatinHypercubeSearch
from .callbacks import VizLogger
from .bench.attn import benchmark_attention_backends, select_fastest_backend
from .bench.kernel import bench_attn as bench_attn_kernels, bench_rope as bench_rope_kernels, select_fastest as select_fastest_kernel, list_kernels
from .bench.mlp import bench_mlp
from .bench.norm import bench_norms
from .bench.kvcache import bench_kvcache
from .presets import from_specs_ops
from .schedulers.asha import ASHAScheduler
from .callbacks import EarlyStopOnNoImprovement


def _load_objective(spec: str):
    if ":" not in spec:
        raise ValueError("Objective must be in 'module:function' format")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return fn


def _parse_space(path: str) -> SearchSpace:
    data: Dict[str, Any] = json.loads(open(path, "r").read())
    params: Dict[str, Any] = {}
    for name, spec in data.items():
        t = spec.get("type")
        if t == "choice":
            params[name] = Choice(spec["options"])  # type: ignore[arg-type]
        elif t == "int":
            params[name] = IntRange(int(spec["start"]), int(spec["stop"]), int(spec.get("step", 1)))
        elif t == "uniform":
            params[name] = Uniform(float(spec["low"]), float(spec["high"]))
        elif t == "loguniform":
            params[name] = LogUniform(float(spec["low"]), float(spec["high"]), float(spec.get("base", 10)))
        else:
            raise ValueError(f"Unknown param type: {t}")
    return SearchSpace(params)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="autotune", description="Autotune utilities")
    sub = p.add_subparsers(dest="cmd")

    p_study = sub.add_parser("study", help="Run a hyperparameter study")
    p_study.add_argument("--objective", type=str, required=True, help="module:function objective(params, trial) -> score")
    p_study.add_argument("--space", type=str, required=True, help="Path to JSON search space definition")
    p_study.add_argument("--algo", type=str, default="random", choices=["random", "grid", "sobol", "lhs"]) 
    p_study.add_argument("--metric", type=str, default="objective")
    p_study.add_argument("--mode", type=str, default="min", choices=["min", "max"]) 
    p_study.add_argument("--max-trials", type=int, default=20)
    p_study.add_argument("--seed", type=int, default=1337)
    p_study.add_argument("--log-dir", type=str, default=".autotune")
    p_study.add_argument("--write-best-to", type=str, default=None)
    p_study.add_argument("--scheduler", type=str, default="none", choices=["none", "asha"], help="Optional multi-fidelity scheduler")
    p_study.add_argument("--asha-budgets", type=str, default=None, help="Comma-separated budgets for ASHA (e.g., 1,3,9)")
    p_study.add_argument("--asha-eta", type=int, default=3, help="ASHA reduction factor (eta)")
    p_study.add_argument("--concurrency", type=int, default=1, help="Max concurrent trials")
    p_study.add_argument("--timeout-s", type=float, default=None, help="Global timeout in seconds")
    p_study.add_argument("--budget-param", type=str, default="budget", help="Name of objective budget parameter")
    p_study.add_argument("--early-stop-patience", type=int, default=None, help="Enable early stop after N non-improving trials")

    p_bench = sub.add_parser("bench-attn", help="Benchmark attention backends")
    p_bench.add_argument("--seq", type=int, default=256)
    p_bench.add_argument("--heads", type=int, default=8)
    p_bench.add_argument("--d-k", type=int, default=64)
    p_bench.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]) 
    p_bench.add_argument("--save-fastest-to", type=str, default=None, help="Optional path to save fastest backend as JSON {backend: name}")

    p_kbench = sub.add_parser("bench-kernel", help="Benchmark registered kernels for a given op")
    p_kbench.add_argument("--op", type=str, required=True, choices=["attn", "rope"]) 
    p_kbench.add_argument("--seq", type=int, default=256)
    p_kbench.add_argument("--heads", type=int, default=8)
    p_kbench.add_argument("--d-k", type=int, default=64)
    p_kbench.add_argument("--base-theta", type=float, default=1e6)
    p_kbench.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]) 

    p_ops = sub.add_parser("space-from-ops", help="Emit a JSON search space from specs.ops categories")
    p_ops.add_argument("--categories", nargs="+", required=True)

    p_mlp = sub.add_parser("bench-mlp", help="Benchmark MLP activations")
    p_mlp.add_argument("--hidden", type=int, default=4096)
    p_mlp.add_argument("--ff", type=int, default=11008)
    p_mlp.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]) 

    p_norm = sub.add_parser("bench-norm", help="Benchmark normalization variants")
    p_norm.add_argument("--hidden", type=int, default=4096)
    p_norm.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]) 

    p_kv = sub.add_parser("bench-kvcache", help="Benchmark KV cache append/slice vs page size")
    p_kv.add_argument("--pages", type=str, default="256,512,1024,2048")
    p_kv.add_argument("--batch", type=int, default=1)
    p_kv.add_argument("--layers", type=int, default=1)
    p_kv.add_argument("--kv-heads", type=int, default=8)
    p_kv.add_argument("--head-dim", type=int, default=128)
    p_kv.add_argument("--seq", type=int, default=256)
    p_kv.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"]) 

    p_presets = sub.add_parser("space-from-presets", help="Emit merged JSON search space from named presets")
    p_presets.add_argument("--names", nargs="+", required=True, help="Preset function names in autotune.presets")

    args = p.parse_args(argv)

    if args.cmd == "bench-attn":
        import torch
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        res = benchmark_attention_backends(seq=args.seq, heads=args.heads, d_k=args.d_k, dtype=dtype_map[args.dtype])
        print("backend,avg_sec")
        for be, dt in res:
            print(f"{be},{dt:.6f}")
        fastest = select_fastest_backend(seq=args.seq, heads=args.heads, d_k=args.d_k, dtype=dtype_map[args.dtype])
        print("fastest:", fastest)
        if args.save_fastest_to:
            try:
                with open(args.save_fastest_to, "w") as f:
                    json.dump({"backend": fastest}, f, indent=2)
                print(f"saved: {args.save_fastest_to}")
            except Exception as e:
                print(f"warn: failed to save fastest to {args.save_fastest_to}: {e}")
        return

    if args.cmd == "bench-kernel":
        import torch
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        if args.op == "attn":
            res = bench_attn_kernels(args.seq, args.heads, args.d_k, dtype=dtype_map[args.dtype])
        else:
            res = bench_rope_kernels(args.seq, args.heads, args.d_k, base_theta=args.base_theta, dtype=dtype_map[args.dtype])
        print("kernel,avg_sec")
        for name, dt in res:
            print(f"{name},{dt:.6f}")
        best = select_fastest_kernel(res)
        if best is not None:
            print("fastest:", best)
        return

    if args.cmd == "bench-mlp":
        import torch
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        res = bench_mlp(args.hidden, args.ff, dtype=dtype_map[args.dtype])
        print("activation,avg_sec")
        for name, dt in res:
            print(f"{name},{dt:.6f}")
        return

    if args.cmd == "bench-norm":
        import torch
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        res = bench_norms(args.hidden, dtype=dtype_map[args.dtype])
        print("norm,avg_sec")
        for name, dt in res:
            print(f"{name},{dt:.6f}")
        return

    if args.cmd == "bench-kvcache":
        import torch
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        pages = [int(x) for x in str(args.pages).split(',') if x.strip()]
        res = bench_kvcache(pages, args.batch, args.layers, args.kv_heads, args.head_dim, args.seq, dtype=dtype_map[args.dtype])
        print("pagesize,avg_sec_per_token")
        for name, dt in res:
            print(f"{name.split('=')[1]},{dt:.6f}")
        return

    if args.cmd == "space-from-presets":
        # Import presets module and merge selected spaces by simple param union
        from . import presets as presets_mod
        spaces = []
        for nm in args.names:
            if not hasattr(presets_mod, nm):
                raise SystemExit(f"Unknown preset: {nm}")
            sp = getattr(presets_mod, nm)()
            spaces.append(sp)
        params: Dict[str, Any] = {}
        for sp in spaces:
            params.update(sp.parameters)
        data: Dict[str, Any] = {}
        for k, v in params.items():
            if hasattr(v, "options"):
                data[k] = {"type": "choice", "options": list(v.options)}  # type: ignore[attr-defined]
            elif hasattr(v, "low") and hasattr(v, "high") and not hasattr(v, "base"):
                data[k] = {"type": "uniform", "low": float(v.low), "high": float(v.high)}
            elif hasattr(v, "low") and hasattr(v, "high") and hasattr(v, "base"):
                data[k] = {"type": "loguniform", "low": float(v.low), "high": float(v.high), "base": float(v.base)}
            elif hasattr(v, "start") and hasattr(v, "stop"):
                data[k] = {"type": "int", "start": int(v.start), "stop": int(v.stop), "step": int(getattr(v, "step", 1))}
        print(json.dumps(data, indent=2))
        return

    if args.cmd == "study":
        objective = _load_objective(args.objective)
        space = _parse_space(args.space)
        if args.algo == "random":
            algo = RandomSearch(args.seed)
        elif args.algo == "grid":
            algo = GridSearch()
        elif args.algo == "sobol":
            algo = SobolSearch(seed=args.seed)
        else:
            algo = LatinHypercubeSearch(seed=args.seed)
        # Optional scheduler
        scheduler = None
        if args.scheduler == "asha":
            budgets: List[int]
            if args.asha_budgets:
                try:
                    budgets = [int(x) for x in str(args.asha_budgets).split(',') if x.strip()]
                except Exception as e:
                    raise SystemExit(f"Invalid --asha-budgets: {args.asha_budgets}")
            else:
                budgets = [1, 3, 9]
            scheduler = ASHAScheduler(budgets=budgets, eta=int(args.asha_eta))
        cfg = StudyConfig(metric=args.metric, mode=args.mode, max_trials=args.max_trials, seed=args.seed, log_dir=args.log_dir, write_best_to=args.write_best_to, concurrency=int(args.concurrency), timeout_s=args.timeout_s, budget_param=str(args.budget_param))
        callbacks = [VizLogger(args.log_dir, metric_key=f"autotune.{args.metric}")]
        # Early stopping callback
        if hasattr(args, "early_stop_patience") and args.early_stop_patience is not None:
            callbacks.append(EarlyStopOnNoImprovement(int(args.early_stop_patience), mode=args.mode))
        study = Study(cfg, space, algo, callbacks=callbacks, scheduler=scheduler)
        best = study.optimize(objective)
        print(json.dumps(best, indent=2))
        return

    if args.cmd == "space-from-ops":
        space = from_specs_ops(args.categories)
        # Emit as JSON with choice types
        data: Dict[str, Any] = {}
        for k, v in space.parameters.items():
            if hasattr(v, "options"):
                data[k] = {"type": "choice", "options": list(v.options)}  # type: ignore[attr-defined]
        print(json.dumps(data, indent=2))
        return

    p.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()


