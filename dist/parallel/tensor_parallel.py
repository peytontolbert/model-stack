from __future__ import annotations

from typing import Tuple

import torch

from tensor.shard import allreduce_, allgather, shard_linear_weight
from tensor.init import kaiming_uniform_linear, xavier_uniform_linear
from dist.utils import get_rank


class TensorParallelLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, *, tp_size: int, bias: bool = True, axis: int = 0):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.axis = int(axis)
        self.tp_size = max(int(tp_size), 1)

        # Initialize via tensor.init on a temporary Linear, then shard
        temp = torch.nn.Linear(in_features, out_features, bias=False)
        kaiming_uniform_linear(temp, nonlinearity="relu")
        shards = shard_linear_weight(temp.weight.data, tp_size=self.tp_size, axis=self.axis)
        # Register this rank's shard; assume rank-aligned assignment by torchrun
        rank = get_rank(0)
        local_index = rank % self.tp_size
        self.weight = torch.nn.Parameter(shards[local_index].contiguous().to(torch.cuda.current_device()))
        self.use_bias = bool(bias)
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.weight.shape[0]))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.linear(x, self.weight, self.bias)
        # All-reduce partial outputs when partitioned along output dimension
        y = allreduce_(y, op="sum")
        return y


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, *, tp_size: int, bias: bool = True):
        super().__init__()
        if out_features % max(int(tp_size), 1) != 0:
            raise ValueError(f"out_features must be divisible by tp_size; got {out_features} vs {tp_size}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.tp_size = max(int(tp_size), 1)
        self.out_per = self.out_features // self.tp_size

        # Use tensor.init to initialize a full linear and then shard rows (axis=0)
        temp = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        kaiming_uniform_linear(temp, nonlinearity="relu")
        shards = shard_linear_weight(temp.weight.data, tp_size=self.tp_size, axis=0)
        local_index = get_rank(0) % self.tp_size
        local_w = shards[local_index].contiguous().to(torch.cuda.current_device())
        self.weight = torch.nn.Parameter(local_w)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_per))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_local = torch.nn.functional.linear(x, self.weight, self.bias)
        # Gather partial outputs along last dimension using tensor.shard
        return allgather(y_local, dim=-1)


class RowParallelLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, *, tp_size: int, bias: bool = True):
        super().__init__()
        if in_features % max(int(tp_size), 1) != 0:
            raise ValueError(f"in_features must be divisible by tp_size; got {in_features} vs {tp_size}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.tp_size = max(int(tp_size), 1)
        self.in_per = self.in_features // self.tp_size

        # Use tensor.init to initialize a full linear and then shard columns (axis=1)
        temp = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        kaiming_uniform_linear(temp, nonlinearity="relu")
        shards = shard_linear_weight(temp.weight.data, tp_size=self.tp_size, axis=1)
        local_index = get_rank(0) % self.tp_size
        local_w = shards[local_index].contiguous().to(torch.cuda.current_device())
        self.weight = torch.nn.Parameter(local_w)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Select local input slice that matches this rank's column shard
        local_index = get_rank(0) % self.tp_size
        start = local_index * self.in_per
        end = start + self.in_per
        x_local = x[..., start:end]
        y_local = torch.nn.functional.linear(x_local, self.weight, None)
        # Sum partial outputs across ranks using tensor.shard
        y = allreduce_(y_local, op="sum")
        if self.bias is not None:
            y = y + self.bias
        return y


def shard_mlp_linear(module: torch.nn.Linear, *, tp_size: int, axis: int = 0) -> Tuple[TensorParallelLinear, torch.nn.Linear]:
    tp = TensorParallelLinear(module.in_features, module.out_features, tp_size=int(tp_size), axis=int(axis), bias=module.bias is not None)
    return tp, module


def apply_tp_to_attention(attn_module: torch.nn.Module, tp_size: int) -> None:
    # Works with attention implementations that expose w_q, w_k, w_v, w_o
    for name in ("w_q", "w_k", "w_v", "w_o"):
        if not hasattr(attn_module, name):
            return
    d_model = attn_module.w_q.in_features
    kv_dim = attn_module.w_k.out_features
    # QKV projections: column-parallel and gather outputs
    attn_module.w_q = ColumnParallelLinear(d_model, d_model, tp_size=tp_size, bias=(attn_module.w_q.bias is not None))
    attn_module.w_k = ColumnParallelLinear(d_model, kv_dim, tp_size=tp_size, bias=(attn_module.w_k.bias is not None))
    attn_module.w_v = ColumnParallelLinear(d_model, kv_dim, tp_size=tp_size, bias=(attn_module.w_v.bias is not None))
    # Output projection: row-parallel and all-reduce sum
    attn_module.w_o = RowParallelLinear(d_model, d_model, tp_size=tp_size, bias=(attn_module.w_o.bias is not None))


def apply_tp_to_mlp(mlp: torch.nn.Module, tp_size: int) -> None:
    # MLP exposes w_in (hidden->ff* or 2*ff) and w_out (ff->hidden)
    if not hasattr(mlp, "w_in") or not hasattr(mlp, "w_out"):
        return
    hidden = mlp.w_out.out_features
    inter = mlp.w_out.in_features
    mlp.w_in = ColumnParallelLinear(hidden, mlp.w_in.out_features, tp_size=tp_size, bias=(mlp.w_in.bias is not None))
    mlp.w_out = RowParallelLinear(inter, hidden, tp_size=tp_size, bias=(mlp.w_out.bias is not None))


def apply_tensor_parallel(model: torch.nn.Module, tp_size: int) -> torch.nn.Module:
    if tp_size <= 1:
        return model
    for module in model.modules():
        # Attention modules
        if hasattr(module, "w_q") and hasattr(module, "w_o"):
            try:
                apply_tp_to_attention(module, tp_size)
            except Exception:
                pass
        # MLP modules
        if module.__class__.__name__ == "MLP" or (hasattr(module, "w_in") and hasattr(module, "w_out")):
            try:
                apply_tp_to_mlp(module, tp_size)
            except Exception:
                pass
    return model


