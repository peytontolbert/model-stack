from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _model_stack_native as native_module


def _time_ms(fn, iters: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def main() -> None:
    torch.manual_seed(0)
    shapes = [
        (65536, 1024, 2048),
        (65536, 2048, 1024),
    ]
    for m, k, n in shapes:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        q = torch.randint(-1, 2, (n, k), device="cuda", dtype=torch.int8)
        scale = torch.rand(n, device="cuda", dtype=torch.float32) * 0.02 + 0.001
        dense_w = (q.float() * scale[:, None]).to(torch.bfloat16).contiguous()

        pos, neg = native_module.bitnet_ternary_pack_masks_forward(q)
        y = native_module.bitnet_ternary_linear_forward(x, pos, neg, scale)
        ref = x.matmul(dense_w.t())
        torch.cuda.synchronize()
        diff = (y - ref).float().abs()

        pack_ms = _time_ms(lambda: native_module.bitnet_ternary_pack_masks_forward(q))
        linear_ms = _time_ms(lambda: native_module.bitnet_ternary_linear_forward(x, pos, neg, scale))

        def pack_and_linear() -> torch.Tensor:
            p, ng = native_module.bitnet_ternary_pack_masks_forward(q)
            return native_module.bitnet_ternary_linear_forward(x, p, ng, scale)

        fused_ms = _time_ms(pack_and_linear)
        dense_ms = _time_ms(lambda: x.matmul(dense_w.t()))
        print(
            f"shape m={m} k={k} n={n} "
            f"max_diff={float(diff.max()):.6g} mean_diff={float(diff.mean()):.6g} "
            f"pack_ms={pack_ms:.4f} linear_ms={linear_ms:.4f} "
            f"pack_linear_ms={fused_ms:.4f} dense_ms={dense_ms:.4f} "
            f"speedup_vs_dense={dense_ms / fused_ms:.4f}"
        )


if __name__ == "__main__":
    main()
