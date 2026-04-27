from __future__ import annotations

import importlib.util
import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[2]
PG_SCRIPT = ROOT / "other_repos/parameter-golf/train_gpt.py"


def _load_muon():
    spec = importlib.util.spec_from_file_location("pg_train_gpt_for_muon_check", PG_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PG_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Muon


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_params(rank: int, device: torch.device) -> tuple[list[torch.nn.Parameter], list[torch.Tensor]]:
    torch.manual_seed(1234)
    shapes = [(8, 16), (8, 16), (16, 8), (16, 8), (12, 12), (12, 12), (4, 8)]
    params = [torch.nn.Parameter(torch.randn(shape, dtype=torch.float32, device=device).bfloat16()) for shape in shapes]
    torch.manual_seed(9000 + rank)
    grads = [torch.randn_like(p) for p in params]
    return params, grads


def _run_optimizer(
    Muon,
    params: list[torch.nn.Parameter],
    grads: list[torch.Tensor],
    *,
    parallel: bool,
) -> list[torch.Tensor]:
    os.environ["MODEL_STACK_MUON_BATCHED"] = "1"
    os.environ["MODEL_STACK_MUON_BATCHED_MIN_BUCKET"] = "2"
    os.environ["MODEL_STACK_MUON_COMPILE"] = "0"
    os.environ["MODEL_STACK_MUON_ROW_NORM"] = "1"
    os.environ["MODEL_STACK_MUON_DISTRIBUTED_EXCHANGE"] = "all_gather"
    os.environ["MODEL_STACK_MUON_DISTRIBUTED_SHARDING"] = "shape_bucket"
    os.environ["MODEL_STACK_MUON_PARALLEL"] = "1" if parallel else "0"

    for p, grad in zip(params, grads, strict=True):
        p.grad = grad.clone()
    opt = Muon(params, lr=0.04, momentum=0.95, backend_steps=5)
    if parallel:
        opt.launch_reduce_scatters()
    else:
        for p in params:
            p.grad = p.grad.bfloat16()
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
    opt.step()
    return [p.detach().clone() for p in params]


def _worker(rank: int, world_size: int, port: int) -> None:
    backend = os.environ.get("MUON_CHECK_BACKEND", "gloo").strip().lower()
    device_name = os.environ.get("MUON_CHECK_DEVICE", "cpu").strip().lower()
    if device_name == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        Muon = _load_muon()
        params_a, grads = _make_params(rank, device)
        params_b = [torch.nn.Parameter(p.detach().clone()) for p in params_a]
        baseline = _run_optimizer(Muon, params_a, grads, parallel=False)
        candidate = _run_optimizer(Muon, params_b, grads, parallel=True)
        max_diff = max((a - b).abs().max().item() for a, b in zip(baseline, candidate, strict=True))
        diff_tensor = torch.tensor(max_diff, device=device)
        dist.all_reduce(diff_tensor, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(f"backend:{backend} device:{device_name} max_diff:{diff_tensor.item():.6g}")
    finally:
        dist.destroy_process_group()


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE_CHECK", "2"))
    if os.environ.get("MUON_CHECK_DEVICE", "cpu").strip().lower() == "cuda" and torch.cuda.device_count() < world_size:
        raise RuntimeError(f"requested {world_size} CUDA ranks but only {torch.cuda.device_count()} devices are visible")
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
