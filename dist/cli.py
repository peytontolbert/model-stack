from __future__ import annotations

import argparse

from .launch import initialize_distributed, recommended_torchrun_cmd
from . import utils


def main() -> None:
    p = argparse.ArgumentParser("dist.cli", description="Distributed utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("print-cmd", help="Print a recommended torchrun command")
    pc.add_argument("--nnodes", type=int, default=1)
    pc.add_argument("--nproc-per-node", type=int, default=1)
    pc.add_argument("--master-addr", type=str, default="127.0.0.1")
    pc.add_argument("--master-port", type=int, default=29500)

    ini = sub.add_parser("init", help="Initialize process group for smoke testing")
    ini.add_argument("--backend", type=str, default="nccl")
    ini.add_argument("--seed", type=int, default=None)

    env = sub.add_parser("env", help="Print basic rank/world/local rank info")

    args = p.parse_args()
    if args.cmd == "print-cmd":
        print(
            recommended_torchrun_cmd(
                nnodes=int(args.nnodes),
                nproc_per_node=int(args.nproc_per_node),
                master_addr=str(args.master_addr),
                master_port=int(args.master_port),
            )
        )
        return
    if args.cmd == "init":
        initialize_distributed(backend=str(args.backend), seed=args.seed)
        print(f"Initialized: rank={utils.get_rank()} world={utils.get_world_size()} local_rank={utils.get_local_rank()}")
        return
    if args.cmd == "env":
        print(f"rank={utils.get_rank()} world={utils.get_world_size()} local_rank={utils.get_local_rank()}")
        return


if __name__ == "__main__":
    main()


