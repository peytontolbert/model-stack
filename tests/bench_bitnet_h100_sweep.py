from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


def _split_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _split_strs(text: str) -> list[str]:
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _gpu_inventory() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    out: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        out.append(
            {
                "index": int(index),
                "name": str(props.name),
                "total_memory_bytes": int(props.total_memory),
                "major": int(props.major),
                "minor": int(props.minor),
                "multi_processor_count": int(props.multi_processor_count),
            }
        )
    return out


def _run_json(cmd: list[str]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.check_output(cmd, cwd=ROOT, text=True, env=env)
    return json.loads(out)


def _native_decode_case(
    *,
    python_exe: str,
    device: str,
    dtype: str,
    batch: int,
    prompt: int,
    layers: int,
    heads: int,
    head_dim: int,
    vocab_size: int,
    activation_quant: str,
    activation_quant_bits: int,
    packed_backend: str,
    warmup: int,
    iters: int,
    repeats: int,
) -> dict[str, Any]:
    cmd = [
        python_exe,
        "tests/bench_bitnet_decode.py",
        "--batch",
        str(batch),
        "--prompt",
        str(prompt),
        "--layers",
        str(layers),
        "--heads",
        str(heads),
        "--head-dim",
        str(head_dim),
        "--vocab-size",
        str(vocab_size),
        "--dtype",
        str(dtype),
        "--device",
        str(device),
        "--native-session",
        "--activation-quant",
        str(activation_quant),
        "--activation-quant-bits",
        str(activation_quant_bits),
        "--packed-backend",
        str(packed_backend),
        "--driver",
        "native_direct",
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--repeats",
        str(repeats),
        "--json",
    ]
    result = _run_json(cmd)
    result["command"] = cmd
    return result


def _hf_case(
    *,
    python_exe: str,
    model_id: str,
    device: str,
    dtype: str,
    layer: int,
    batch: int,
    mode: str,
    modules: str,
    activation_quant: str,
    max_prompt_tokens: int,
    warmup: int,
    iters: int,
    repeats: int,
    prompt: str,
    prompt_repeat: int,
) -> dict[str, Any]:
    cmd = [
        python_exe,
        "tests/bench_hf_bitnet_transformers.py",
        "--model-id",
        str(model_id),
        "--device",
        str(device),
        "--dtype",
        str(dtype),
        "--layer",
        str(layer),
        "--batch",
        str(batch),
        "--mode",
        str(mode),
        "--modules",
        str(modules),
        "--activation-quant",
        str(activation_quant),
        "--max-prompt-tokens",
        str(max_prompt_tokens),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--repeats",
        str(repeats),
        "--prompt",
        str(prompt),
        "--prompt-repeat",
        str(prompt_repeat),
        "--json",
    ]
    result = _run_json(cmd)
    result["command"] = cmd
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reproducible BitNet tuning sweep intended for H100-class boxes."
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--model-id", type=str, default="microsoft/bitnet-b1.58-2B-4T")
    parser.add_argument("--layers", type=str, default="0,15,29")
    parser.add_argument("--modules", type=str, default="q_proj,o_proj,gate_proj,down_proj,lm_head")
    parser.add_argument("--decode-batches", type=str, default="1,4,8")
    parser.add_argument("--decode-prompts", type=str, default="32,128")
    parser.add_argument("--native-decode-layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--activation-quant", type=str, default="dynamic_int8")
    parser.add_argument("--activation-quant-bits", type=int, default=8)
    parser.add_argument("--hf-decode-batch", type=int, default=4)
    parser.add_argument("--hf-prefill-batch", type=int, default=1)
    parser.add_argument("--hf-prefill-max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--hf-prompt", type=str, default="BitNet H100 tuning prompt. ")
    parser.add_argument("--hf-prompt-repeat", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-disabled-baseline", action="store_true")
    parser.add_argument("--skip-native-decode", action="store_true")
    parser.add_argument("--skip-hf-decode", action="store_true")
    parser.add_argument("--skip-hf-prefill", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "host": str(platform.node()),
        "platform": str(platform.platform()),
        "python_exe": str(args.python_exe),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "requested_device": str(args.device),
        "gpu_inventory": _gpu_inventory(),
        "model_id": str(args.model_id),
        "activation_quant": str(args.activation_quant),
        "activation_quant_bits": int(args.activation_quant_bits),
        "sections": {},
    }

    layers = _split_ints(args.layers)
    decode_batches = _split_ints(args.decode_batches)
    decode_prompts = _split_ints(args.decode_prompts)
    modules = ",".join(_split_strs(args.modules))

    if not bool(args.skip_native_decode):
        native_results: list[dict[str, Any]] = []
        backends = ["auto"]
        if bool(args.include_disabled_baseline):
            backends.append("disabled")
        for batch in decode_batches:
            for prompt in decode_prompts:
                for packed_backend in backends:
                    native_results.append(
                        _native_decode_case(
                            python_exe=str(args.python_exe),
                            device=str(args.device),
                            dtype=str(args.dtype),
                            batch=int(batch),
                            prompt=int(prompt),
                            layers=int(args.native_decode_layers),
                            heads=int(args.heads),
                            head_dim=int(args.head_dim),
                            vocab_size=int(args.vocab_size),
                            activation_quant=str(args.activation_quant),
                            activation_quant_bits=int(args.activation_quant_bits),
                            packed_backend=str(packed_backend),
                            warmup=int(args.warmup),
                            iters=int(args.iters),
                            repeats=int(args.repeats),
                        )
                    )
        payload["sections"]["native_decode"] = native_results

    if not bool(args.skip_hf_decode):
        hf_decode_results: list[dict[str, Any]] = []
        for layer in layers:
            hf_decode_results.append(
                _hf_case(
                    python_exe=str(args.python_exe),
                    model_id=str(args.model_id),
                    device=str(args.device),
                    dtype=str(args.dtype),
                    layer=int(layer),
                    batch=int(args.hf_decode_batch),
                    mode="decode",
                    modules=modules,
                    activation_quant=str(args.activation_quant),
                    max_prompt_tokens=max(decode_prompts),
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    repeats=int(args.repeats),
                    prompt=str(args.hf_prompt),
                    prompt_repeat=max(int(args.hf_prompt_repeat), 1),
                )
            )
        payload["sections"]["hf_decode"] = hf_decode_results

    if not bool(args.skip_hf_prefill):
        hf_prefill_results: list[dict[str, Any]] = []
        for layer in layers:
            hf_prefill_results.append(
                _hf_case(
                    python_exe=str(args.python_exe),
                    model_id=str(args.model_id),
                    device=str(args.device),
                    dtype=str(args.dtype),
                    layer=int(layer),
                    batch=int(args.hf_prefill_batch),
                    mode="prefill",
                    modules=modules,
                    activation_quant=str(args.activation_quant),
                    max_prompt_tokens=int(args.hf_prefill_max_prompt_tokens),
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                    repeats=int(args.repeats),
                    prompt=str(args.hf_prompt),
                    prompt_repeat=max(int(args.hf_prompt_repeat), 1),
                )
            )
        payload["sections"]["hf_prefill"] = hf_prefill_results

    text = json.dumps(payload, indent=2, sort_keys=True)
    if str(args.output).strip():
        output_path = Path(str(args.output))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote={output_path}")
    print(text)


if __name__ == "__main__":
    main()
