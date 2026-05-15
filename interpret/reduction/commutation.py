from __future__ import annotations

import torch


def rg_commutation_error(
    teacher_group_output: torch.Tensor,
    student_block_output: torch.Tensor,
    *,
    projection_after: torch.Tensor | None = None,
    relative: bool = True,
) -> torch.Tensor:
    teacher = teacher_group_output
    if projection_after is not None:
        teacher = teacher @ projection_after.to(device=teacher.device, dtype=teacher.dtype)
    if teacher.shape != student_block_output.shape:
        raise ValueError(f"teacher/student outputs must match after projection, got {tuple(teacher.shape)} and {tuple(student_block_output.shape)}")
    err = (teacher.float() - student_block_output.float()).flatten(1).norm(dim=1)
    if relative:
        denom = teacher.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        err = err / denom
    return err.mean()


def block_grouping_dynamic_program(cost: torch.Tensor, n_student_blocks: int) -> tuple[float, list[tuple[int, int]]]:
    """Choose contiguous teacher block groups with minimum total cost.

    ``cost[i, j]`` is the cost of mapping teacher blocks ``[i, j)`` to one
    student block. Invalid groups can be set to ``inf``.
    """

    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("cost must be square with cost[i,j] for i < j")
    n = int(cost.shape[0])
    k = int(n_student_blocks)
    dp = torch.full((k + 1, n + 1), float("inf"), dtype=cost.dtype)
    parent = [[-1 for _ in range(n + 1)] for _ in range(k + 1)]
    dp[0, 0] = 0.0
    for blocks in range(1, k + 1):
        for end in range(1, n + 1):
            for start in range(blocks - 1, end):
                val = dp[blocks - 1, start] + cost[start, end - 1]
                if val < dp[blocks, end]:
                    dp[blocks, end] = val
                    parent[blocks][end] = start
    groups: list[tuple[int, int]] = []
    cur = n
    for blocks in range(k, 0, -1):
        start = parent[blocks][cur]
        if start < 0:
            return float("inf"), []
        groups.append((start, cur))
        cur = start
    groups.reverse()
    return float(dp[k, n].item()), groups
