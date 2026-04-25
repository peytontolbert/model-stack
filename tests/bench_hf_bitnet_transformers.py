from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compress.quantization import QuantizedLinearBitNet


@dataclass
class BenchResult:
    name: str
    shape: tuple[int, ...]
    hf_ms: float
    kernel_ms: float
    max_abs_err: float
    hf_runs_ms: list[float]
    kernel_runs_ms: list[float]


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported dtype: {name}")
    return mapping[normalized]


def _timeit(fn, *, warmup: int, iters: int, use_cuda_events: bool) -> float:
    for _ in range(int(warmup)):
        fn()
    if use_cuda_events:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(int(iters)):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / max(int(iters), 1)
    t0 = time.perf_counter()
    for _ in range(int(iters)):
        fn()
    return ((time.perf_counter() - t0) * 1000.0) / max(int(iters), 1)


def _resolve_target_modules(model, *, layer_idx: int, names: list[str]) -> dict[str, torch.nn.Module]:
    module_map = dict(model.named_modules())
    path_map = {
        "q_proj": f"model.layers.{layer_idx}.self_attn.q_proj",
        "k_proj": f"model.layers.{layer_idx}.self_attn.k_proj",
        "v_proj": f"model.layers.{layer_idx}.self_attn.v_proj",
        "o_proj": f"model.layers.{layer_idx}.self_attn.o_proj",
        "gate_proj": f"model.layers.{layer_idx}.mlp.gate_proj",
        "up_proj": f"model.layers.{layer_idx}.mlp.up_proj",
        "down_proj": f"model.layers.{layer_idx}.mlp.down_proj",
        "lm_head": "lm_head",
    }
    out: dict[str, torch.nn.Module] = {}
    for name in names:
        key = str(name).strip()
        if key not in path_map:
            raise KeyError(f"unknown module name: {key}")
        path = path_map[key]
        if path not in module_map:
            raise KeyError(f"module path not found: {path}")
        out[key] = module_map[path]
    return out


def _capture_inputs(
    model,
    tokenizer,
    *,
    prompt: str,
    batch_size: int,
    layer_idx: int,
    mode: str,
    target_names: list[str],
    max_prompt_tokens: int | None,
) -> dict[str, torch.Tensor]:
    modules = _resolve_target_modules(model, layer_idx=layer_idx, names=target_names)
    captured: dict[str, torch.Tensor] = {}
    encoded = tokenizer(prompt, return_tensors="pt")
    if max_prompt_tokens is not None and int(max_prompt_tokens) > 0:
        encoded = {
            key: value[:, -int(max_prompt_tokens) :].contiguous()
            for key, value in encoded.items()
        }
    if int(batch_size) > 1:
        encoded = {
            key: value.repeat(int(batch_size), *([1] * (value.dim() - 1))).contiguous()
            for key, value in encoded.items()
        }
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    def _run_with_hooks(**forward_kwargs):
        handles = []

        def _make_hook(name: str):
            def _hook(_module, inputs):
                if not inputs:
                    raise RuntimeError(f"module {name} received no inputs")
                x = inputs[0]
                if not torch.is_tensor(x):
                    raise TypeError(f"module {name} input is not a tensor: {type(x)}")
                captured[name] = x.detach().clone()

            return _hook

        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(_make_hook(name)))
        try:
            return model(**forward_kwargs)
        finally:
            for handle in handles:
                handle.remove()

    with torch.no_grad():
        if str(mode).strip().lower() == "prefill":
            _run_with_hooks(**encoded, use_cache=True)
        elif str(mode).strip().lower() == "decode":
            outputs = model(**encoded, use_cache=True)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            _run_with_hooks(input_ids=next_token, past_key_values=outputs.past_key_values, use_cache=True)
        else:
            raise ValueError(f"unsupported mode: {mode}")

    missing = [name for name in target_names if name not in captured]
    if missing:
        raise RuntimeError(f"failed to capture inputs for: {missing}")
    return captured


def _build_bitnet_kernel(module: torch.nn.Linear, *, activation_quant: str) -> QuantizedLinearBitNet:
    q = QuantizedLinearBitNet(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
    ).from_float(
        copy.deepcopy(module).cpu(),
        activation_quant=str(activation_quant),
        spin=False,
    )
    return q.to(device=module.weight.device)


def _bench_module_pair(
    name: str,
    hf_module: torch.nn.Linear,
    bitnet_module: QuantizedLinearBitNet,
    x: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    repeats: int,
    use_cuda_events: bool,
    backend: str | None,
) -> BenchResult:
    with torch.inference_mode():
        hf_out = hf_module(x)
        kernel_out = bitnet_module.runtime_linear(x, backend=backend)
    max_abs_err = float((hf_out - kernel_out).abs().max().item())
    repeat_count = max(int(repeats), 1)
    hf_runs = [
        float(_timeit(lambda: hf_module(x), warmup=warmup, iters=iters, use_cuda_events=use_cuda_events))
        for _ in range(repeat_count)
    ]
    kernel_runs = [
        float(
            _timeit(
                lambda: bitnet_module.runtime_linear(x, backend=backend),
                warmup=warmup,
                iters=iters,
                use_cuda_events=use_cuda_events,
            )
        )
        for _ in range(repeat_count)
    ]
    return BenchResult(
        name=name,
        shape=tuple(int(v) for v in x.shape),
        hf_ms=float(statistics.median(hf_runs)),
        kernel_ms=float(statistics.median(kernel_runs)),
        max_abs_err=max_abs_err,
        hf_runs_ms=hf_runs,
        kernel_runs_ms=kernel_runs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HF BitNet module projections against local BitNet kernels.")
    parser.add_argument("--model-id", type=str, default="microsoft/bitnet-b1.58-2B-4T")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--mode", type=str, choices=("prefill", "decode"), default="decode")
    parser.add_argument("--modules", type=str, default="q_proj,o_proj,gate_proj,down_proj,lm_head")
    parser.add_argument("--activation-quant", type=str, default="dynamic_int8")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Explain why RoPE scaling matters for long-context decode.")
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-cuda-events", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    module_names = [name.strip() for name in str(args.modules).split(",") if name.strip()]
    prompt = str(args.prompt) * max(int(args.prompt_repeat), 1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    targets = _resolve_target_modules(model, layer_idx=int(args.layer), names=module_names)
    inputs = _capture_inputs(
        model,
        tokenizer,
        prompt=prompt,
        batch_size=int(args.batch),
        layer_idx=int(args.layer),
        mode=str(args.mode),
        target_names=module_names,
        max_prompt_tokens=int(args.max_prompt_tokens) if args.max_prompt_tokens is not None else None,
    )

    results: list[BenchResult] = []

    for name in module_names:
        hf_module = targets[name]
        if not isinstance(hf_module, torch.nn.Linear):
            raise TypeError(f"expected nn.Linear for {name}, got {type(hf_module).__name__}")
        bitnet_module = _build_bitnet_kernel(hf_module, activation_quant=str(args.activation_quant))
        result = _bench_module_pair(
            name,
            hf_module,
            bitnet_module,
            inputs[name].to(device=device, dtype=next(hf_module.parameters()).dtype),
            warmup=int(args.warmup),
            iters=int(args.iters),
            repeats=int(args.repeats),
            use_cuda_events=not bool(args.no_cuda_events),
            backend=args.backend,
        )
        results.append(result)

    payload = {
        "model_id": str(args.model_id),
        "device": str(device),
        "dtype": str(dtype),
        "batch": int(args.batch),
        "mode": str(args.mode),
        "layer": int(args.layer),
        "activation_quant": str(args.activation_quant),
        "backend": args.backend,
        "prompt_repeat": max(int(args.prompt_repeat), 1),
        "max_prompt_tokens": int(args.max_prompt_tokens) if args.max_prompt_tokens is not None else None,
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "repeats": int(args.repeats),
        "modules": list(module_names),
        "results": [
            {
                "name": result.name,
                "shape": list(result.shape),
                "hf_ms": float(result.hf_ms),
                "kernel_ms": float(result.kernel_ms),
                "speedup_vs_hf": float(result.hf_ms / result.kernel_ms) if result.kernel_ms > 0 else float("inf"),
                "max_abs_err": float(result.max_abs_err),
                "hf_runs_ms": [float(v) for v in result.hf_runs_ms],
                "kernel_runs_ms": [float(v) for v in result.kernel_runs_ms],
                "hf_mean_ms": float(statistics.mean(result.hf_runs_ms)),
                "kernel_mean_ms": float(statistics.mean(result.kernel_runs_ms)),
            }
            for result in results
        ],
    }

    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print(f"model_id={args.model_id}")
    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"batch={int(args.batch)}")
    print(f"mode={args.mode}")
    print(f"layer={int(args.layer)}")
    print(f"activation_quant={args.activation_quant}")
    print(f"backend={args.backend}")
    print(f"prompt_repeat={max(int(args.prompt_repeat), 1)}")
    print(f"repeats={int(args.repeats)}")
    print(f"modules={','.join(module_names)}")

    for result in results:
        speedup = result.hf_ms / result.kernel_ms if result.kernel_ms > 0 else float("inf")
        print(f"[{result.name}] shape={result.shape}")
        print(f"  hf_ms={result.hf_ms:.3f}")
        print(f"  kernel_ms={result.kernel_ms:.3f}")
        print(f"  speedup_vs_hf={speedup:.3f}x")
        print(f"  max_abs_err={result.max_abs_err:.6f}")
        if int(args.repeats) > 1:
            print(f"  hf_runs_ms={','.join(f'{v:.3f}' for v in result.hf_runs_ms)}")
            print(f"  kernel_runs_ms={','.join(f'{v:.3f}' for v in result.kernel_runs_ms)}")


if __name__ == "__main__":
    main()
