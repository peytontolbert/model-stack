from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

DEFAULT_REPO = Path("/data/repositories/ABot-World")
DEFAULT_MODEL = Path("/arxiv/models/acvlab--ABot-World-0-5B-LF")


@dataclass(frozen=True)
class ABotWorldProbe:
    model_id: str
    model_path: str
    repo_path: str
    status: str
    torch_version: str | None = None
    cuda_version: str | None = None
    attention_backend: str | None = None
    optional_kernels: dict[str, bool] | None = None
    load_seconds: float | None = None
    memory: dict[str, Any] | None = None
    patches: tuple[str, ...] = ()
    error: str | None = None


@contextmanager
def _repo_context(repo_path: Path) -> Iterator[None]:
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    os.chdir(repo_path)
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def _gpu_memory(torch: Any, device_index: int = 0) -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info(device_index)
    return {
        "used_mb": int((total - free) / 1024 / 1024),
        "free_mb": int(free / 1024 / 1024),
        "total_mb": int(total / 1024 / 1024),
        "reserved_mb": int(torch.cuda.memory_reserved(device_index) / 1024 / 1024),
        "allocated_mb": int(torch.cuda.memory_allocated(device_index) / 1024 / 1024),
    }


def _optional_kernel_flags() -> dict[str, bool]:
    return {
        "flash_attn": importlib.util.find_spec("flash_attn") is not None,
        "flash_attn_interface": importlib.util.find_spec("flash_attn_interface") is not None,
        "sageattention": importlib.util.find_spec("sageattention") is not None,
        "sageattn3": importlib.util.find_spec("sageattn3") is not None,
        "triton": importlib.util.find_spec("triton") is not None,
    }


def load_abot_generator_cuda(
    *,
    repo_path: str | Path = DEFAULT_REPO,
    model_path: str | Path = DEFAULT_MODEL,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
) -> tuple[Any, ABotWorldProbe]:
    """Load ABot's causal Wan generator with the model-stack CUDA bridge path.

    The upstream pipeline lets Diffusers materialize weights on CPU and later casts/moves
    the model. For the 9.8 GB ABot generator that can spend minutes in CPU paging and
    inflate host RSS. This bridge passes torch_dtype and a string device_map directly to
    Diffusers/Accelerate so safetensors dispatches BF16 weights onto CUDA.
    """
    repo = Path(repo_path)
    model = Path(model_path)
    started = time.perf_counter()
    patches = (
        "diffusers_linear_activation_fallback",
        "abot_generator_direct_cuda_device_map_bf16",
        "safetensors_device_map_string_cuda",
    )
    with _repo_context(repo):
        import torch
        from omegaconf import OmegaConf
        from utils.wan_wrapper import model_kwargs_with_relative_rope
        from wan.modules import attention as attn
        from wan.modules.causal_model import CausalWanModel

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        default_config = OmegaConf.load("configs/default_config.yaml")
        runtime_config = OmegaConf.load("configs/long_forcing_dmd.yaml")
        config = OmegaConf.merge(default_config, runtime_config)
        config.model_kwargs.model_name = str(model if model.is_absolute() else model.resolve())
        kwargs = model_kwargs_with_relative_rope(config)
        kwargs["local_attn_size"] = kwargs.get("local_attn_size", getattr(config, "local_attn_size", -1))
        kwargs["use_relative_rope"] = True
        local_attn_size = kwargs.pop("local_attn_size")
        sink_size = kwargs.pop("sink_size", 0)
        before = _gpu_memory(torch, 0)
        generator = CausalWanModel.from_pretrained(
            config.model_kwargs.model_name,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            model_type=config.model_type,
            num_frame_per_block=int(config.num_frame_per_block),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            device_map={"": device},
            **kwargs,
        )
        generator.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        probe = ABotWorldProbe(
            model_id="acvlab--ABot-World-0-5B-LF",
            model_path=str(model),
            repo_path=str(repo),
            status="generator_cuda_bf16_loaded",
            torch_version=str(torch.__version__),
            cuda_version=str(torch.version.cuda),
            attention_backend=str(getattr(attn, "DEFAULT_ATTN_BACKEND", "unknown")),
            optional_kernels=_optional_kernel_flags(),
            load_seconds=time.perf_counter() - started,
            memory={"before": before, "after_load": _gpu_memory(torch, 0)},
            patches=patches,
        )
        return generator, probe


def probe_as_dict(probe: ABotWorldProbe) -> dict[str, Any]:
    return asdict(probe)


def write_probe_json(probe: ABotWorldProbe, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(probe_as_dict(probe), indent=2, sort_keys=True) + "\n")
