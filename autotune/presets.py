from __future__ import annotations

from typing import Dict, Iterable

from .spaces import Choice, IntRange, LogUniform, SearchSpace, Uniform


def training_basics() -> SearchSpace:
    return SearchSpace({
        "precision": Choice(["fp32", "bf16", "fp16"]),
        "strategy": Choice(["DDP", "FSDP", "DeepSpeed"]),
        "lr": LogUniform(1e-5, 5e-3),
        "weight_decay": LogUniform(1e-6, 1e-2),
        "batch_size": Choice([8, 16, 32, 64]),
        "grad_clip": Uniform(0.0, 2.0),
        "attn_backend": Choice(["torch", "xformers", "flash2"]),
        "seq_len": IntRange(128, 1025, 128),
        "dropout": Uniform(0.0, 0.3),
    })


def from_specs_ops(categories: Iterable[str]) -> SearchSpace:
    # Build a space that chooses among curated ops per category. Caller should map chosen ops into model wiring.
    from specs import ops as spec_ops
    params: Dict[str, Choice] = {}
    for cat in categories:
        names = list(spec_ops.list_ops(cat).get(cat, ()))
        if not names:
            continue
        params[f"op.{cat}"] = Choice(names)
    return SearchSpace(params)


def attention_full() -> SearchSpace:
    return SearchSpace({
        "attn_backend": Choice(["torch", "xformers", "flash2", "triton"]),
        "n_kv_heads": Choice([1, 2, 4, 8]),
        "attn_dropout": Uniform(0.0, 0.2),
        "use_rope": Choice([True, False]),
        "rope_theta": LogUniform(1e5, 1e7),
        "use_alibi": Choice([False, True]),
        "window_size": IntRange(64, 1025, 64),
    })


def blocks_wiring() -> SearchSpace:
    return SearchSpace({
        "norm_policy": Choice(["prenorm", "postnorm"]),
        "norm_type": Choice(["rms", "layer"]),
        "activation": Choice(["gelu", "silu", "swiglu", "geglu", "reglu"]),
        "residual_scale": Uniform(0.5, 1.5),
        "checkpoint_forward": Choice([False, True]),
        "resid_dropout": Uniform(0.0, 0.2),
        "mlp_dropout": Uniform(0.0, 0.2),
        "attn_dropout": Uniform(0.0, 0.2),
    })


def parallelism(max_tp: int = 8) -> SearchSpace:
    # Caller is responsible for feasible values given world size
    tp_vals = [1, 2, 4, 8]
    tp_vals = [t for t in tp_vals if t <= int(max_tp)]
    return SearchSpace({
        "tp_size": Choice(tp_vals or [1]),
        "tp_apply_attn": Choice([True, False]),
        "tp_apply_mlp": Choice([True, False]),
        "tp_gather_output": Choice([True, False]),
    })


def numerics() -> SearchSpace:
    return SearchSpace({
        "matmul_precision": Choice(["high", "medium", "highest"]),
        "logits_fp32": Choice([True, False]),
    })


