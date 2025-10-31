from typing import Tuple
import torch
from tensor.numerics import router_topk_with_capacity
from tensor.metrics import moe_load_balance_loss as _moe_lb
from tensor.sparse import gather_combine


def topk_router(scores: torch.Tensor, k: int = 1, capacity_factor: float = 1.25, drop_policy: str = "dropless") -> Tuple[torch.Tensor, torch.Tensor]:
    return router_topk_with_capacity(scores, k=k, capacity_factor=capacity_factor, drop_policy=drop_policy)


def load_balance_loss(logits: torch.Tensor, routes: Tuple[torch.Tensor, torch.Tensor], num_experts: int) -> torch.Tensor:
    return _moe_lb(logits, num_experts=num_experts, dim=-1)


def combine_expert_outputs(expert_outputs, routes: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    # expert_outputs: list of E tensors (B,T,D) or a stacked tensor (B,T,E,D)
    # routes: (assignments (B,T,k), combine_weights (B,T,k))
    assignments, combine_weights = routes
    if isinstance(expert_outputs, (list, tuple)):
        outputs = torch.stack(expert_outputs, dim=2)  # (B,T,E,D)
    else:
        outputs = expert_outputs  # assume already (B,T,E,D)
    return gather_combine(outputs, assignments, combine_weights, gather_dim=2)


def expert_parallel_partition(params, world_size: int, policy: str = "block"):
    # Return view of params shards per rank; stubbed
    shards = [list(params) for _ in range(world_size)]
    return shards


