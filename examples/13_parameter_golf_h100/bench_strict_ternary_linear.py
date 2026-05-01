from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _model_stack_native as native_module


def _pack_int4_twos_complement(qweight: torch.Tensor) -> torch.Tensor:
    q = qweight.to(dtype=torch.int16)
    q = torch.where(q < 0, q + 16, q).to(dtype=torch.uint8)
    low = q[:, 0::2] & 0x0F
    high = (q[:, 1::2] & 0x0F) << 4
    return (low | high).contiguous()


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
        w_scale = torch.rand(n, device="cuda", dtype=torch.float32) * 0.02 + 0.001
        dense_w = (q.float() * w_scale[:, None]).to(torch.bfloat16).contiguous()

        w_pos, w_neg = native_module.bitnet_ternary_pack_masks_forward(q)
        w_pos64, w_neg64 = native_module.bitnet_ternary_pack_masks64_forward(q)
        x_pos, x_neg, x_scale = native_module.bitnet_ternary_quantize_activation_forward(x)
        x_pos64, x_neg64, x_scale64 = native_module.bitnet_ternary_quantize_activation64_forward(x)
        strict = native_module.bitnet_strict_ternary_linear_forward(
            x_pos,
            x_neg,
            x_scale,
            w_pos,
            w_neg,
            w_scale,
            torch.bfloat16,
        )
        strict64 = native_module.bitnet_strict_ternary_linear64_forward(
            x_pos64,
            x_neg64,
            x_scale64,
            w_pos64,
            w_neg64,
            w_scale,
            torch.bfloat16,
        )
        ternary_bf16 = native_module.bitnet_ternary_linear_forward(x, w_pos, w_neg, w_scale)
        dense = x.matmul(dense_w.t())

        packed_int4 = native_module.cutlass_int4_pack_shuffled_forward(q)
        cutlass_int4 = native_module.cutlass_int4_bf16_linear_forward(x, packed_int4, w_scale, None, True)
        torch.cuda.synchronize()

        strict_ref_w = q.float() * w_scale[:, None]
        x_ref_scale = x.float().abs().mean(dim=1).clamp_min(1.0e-8)
        x_ref_q = torch.round(x.float() / x_ref_scale[:, None]).clamp(-1, 1)
        strict_ref = (x_ref_q @ strict_ref_w.t()).mul(x_ref_scale[:, None]).to(torch.bfloat16)
        strict_diff = (strict - strict_ref).float().abs()
        strict64_diff = (strict64 - strict_ref).float().abs()
        bf16_diff = (ternary_bf16 - dense).float().abs()
        int4_diff = (cutlass_int4 - dense).float().abs()

        strict_quant_ms = _time_ms(lambda: native_module.bitnet_ternary_quantize_activation_forward(x))
        strict_linear_ms = _time_ms(
            lambda: native_module.bitnet_strict_ternary_linear_forward(
                x_pos, x_neg, x_scale, w_pos, w_neg, w_scale, torch.bfloat16
            )
        )
        strict64_quant_ms = _time_ms(lambda: native_module.bitnet_ternary_quantize_activation64_forward(x))
        strict64_linear_ms = _time_ms(
            lambda: native_module.bitnet_strict_ternary_linear64_forward(
                x_pos64, x_neg64, x_scale64, w_pos64, w_neg64, w_scale, torch.bfloat16
            )
        )

        def strict_full() -> torch.Tensor:
            xp, xn, xs = native_module.bitnet_ternary_quantize_activation_forward(x)
            return native_module.bitnet_strict_ternary_linear_forward(
                xp, xn, xs, w_pos, w_neg, w_scale, torch.bfloat16
            )

        def strict64_full() -> torch.Tensor:
            xp, xn, xs = native_module.bitnet_ternary_quantize_activation64_forward(x)
            return native_module.bitnet_strict_ternary_linear64_forward(
                xp, xn, xs, w_pos64, w_neg64, w_scale, torch.bfloat16
            )

        strict_full_ms = _time_ms(strict_full)
        strict64_full_ms = _time_ms(strict64_full)
        bf16_ternary_ms = _time_ms(lambda: native_module.bitnet_ternary_linear_forward(x, w_pos, w_neg, w_scale))
        dense_ms = _time_ms(lambda: x.matmul(dense_w.t()))
        int4_ms = _time_ms(lambda: native_module.cutlass_int4_bf16_linear_forward(x, packed_int4, w_scale, None, True))

        print(
            f"shape m={m} k={k} n={n} "
            f"strict_diff_max={float(strict_diff.max()):.6g} strict_diff_mean={float(strict_diff.mean()):.6g} "
            f"strict64_diff_max={float(strict64_diff.max()):.6g} strict64_diff_mean={float(strict64_diff.mean()):.6g} "
            f"bf16_ternary_diff_max={float(bf16_diff.max()):.6g} int4_diff_max={float(int4_diff.max()):.6g} "
            f"strict_quant_ms={strict_quant_ms:.4f} strict_linear_ms={strict_linear_ms:.4f} "
            f"strict_full_ms={strict_full_ms:.4f} strict64_quant_ms={strict64_quant_ms:.4f} "
            f"strict64_linear_ms={strict64_linear_ms:.4f} strict64_full_ms={strict64_full_ms:.4f} "
            f"bf16_ternary_ms={bf16_ternary_ms:.4f} "
            f"cutlass_int4_ms={int4_ms:.4f} dense_ms={dense_ms:.4f} "
            f"strict_speedup_vs_dense={dense_ms / strict_full_ms:.4f} "
            f"strict64_speedup_vs_dense={dense_ms / strict64_full_ms:.4f}"
        )


if __name__ == "__main__":
    main()
