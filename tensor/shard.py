import math
import torch
import warnings


def tp_linear_partition(in_features: int, out_features: int, tp_degree: int, scheme: str = "row"):
    if tp_degree <= 0:
        raise ValueError("tp_degree must be > 0")
    if scheme == "row":
        return {
            "in_shards": [in_features // tp_degree] * tp_degree,
            "out_shards": [out_features] * tp_degree,
        }
    if scheme == "col":
        return {
            "in_shards": [in_features] * tp_degree,
            "out_shards": [out_features // tp_degree] * tp_degree,
        }
    raise ValueError("scheme must be 'row' or 'col'")


def kv_partition(num_heads: int, seq_len: int, tp_degree: int, gqa: bool = True):
    # returns ranges per shard for heads and sequence
    h_per = math.ceil(num_heads / tp_degree)
    t_per = math.ceil(seq_len / tp_degree)
    head_ranges = [(i * h_per, min((i + 1) * h_per, num_heads)) for i in range(tp_degree)]
    time_ranges = [(i * t_per, min((i + 1) * t_per, seq_len)) for i in range(tp_degree)]
    return head_ranges, time_ranges


def estimate_activation_bytes(B: int, T: int, D: int, dtype: str = "bf16") -> int:
    bytes_per = {"fp32": 4, "float32": 4, "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2}.get(dtype, 2)
    return int(B) * int(T) * int(D) * int(bytes_per)


def attn_flops(B: int, H: int, T: int, S: int, D: int) -> int:
    # QK^T: B*H*T*S*D, softmax ~ B*H*T*S, PV: B*H*T*S*D
    return int(B) * int(H) * (int(T) * int(S) * int(D) * 2 + int(T) * int(S))


def mlp_flops(B: int, T: int, D: int, expand: int) -> int:
    # two matmuls: D->expand*D and expand*D->D (ignoring bias/act)
    return int(B) * int(T) * (int(D) * int(expand) * int(D) * 2)


def tensor_bytes(x) -> int:
    import torch
    if not isinstance(x, torch.Tensor):
        return 0
    return x.numel() * x.element_size()


def estimate_latency_attn(T: int, H: int, D: int, bandwidth_GBs: float, flops_TFLOPs: float) -> float:
    # very rough estimate: time = max(compute_time, memory_time)
    compute = (attn_flops(B=1, H=H, T=T, S=T, D=D) / (flops_TFLOPs * 1e12))
    # bytes ~ read/write Q,K,V and probs: scale by 4 tensors (rough)
    bytes_io = 4 * T * H * D * 2  # read+write
    memory = bytes_io / (bandwidth_GBs * 1e9)
    return max(compute, memory)


# Distributed collectives with graceful fallbacks
def _get_op(op: str):
    try:
        import torch.distributed as dist
        mapping = {
            "sum": dist.ReduceOp.SUM,
            "avg": dist.ReduceOp.AVG,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
            "product": dist.ReduceOp.PRODUCT,
        }
        return mapping.get(op, dist.ReduceOp.SUM)
    except Exception:
        return None


def allreduce_(x: torch.Tensor, op: str = "sum") -> torch.Tensor:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, op=_get_op(op))
            return x
    except Exception:
        warnings.warn("allreduce fallback (no dist)")
    return x


def reduce_scatter(x: torch.Tensor, chunks: int, op: str = "sum") -> torch.Tensor:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            out = torch.empty_like(x, shape=(x.shape[0] // chunks,) + x.shape[1:])
            dist.reduce_scatter_tensor(out, x.contiguous(), op=_get_op(op))
            return out
    except Exception:
        warnings.warn("reduce_scatter fallback (no dist)")
    # Fallback: return the first chunk
    dim0 = x.shape[0]
    per = math.ceil(dim0 / max(chunks, 1))
    return x.narrow(0, 0, min(per, dim0)).contiguous()


def allgather(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            out_shape = list(x.shape)
            out_shape[dim] = out_shape[dim] * world
            out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            dist.all_gather_into_tensor(out, x.contiguous(), dim=dim)
            return out
    except Exception:
        warnings.warn("allgather fallback (no dist)")
    return x


def shard_linear_weight(weight: torch.Tensor, tp_size: int, axis: int = 0):
    if tp_size <= 1:
        return [weight]
    n = weight.shape[axis]
    per = math.ceil(n / tp_size)
    shards = []
    for i in range(tp_size):
        start = i * per
        end = min((i + 1) * per, n)
        if start >= end:
            shards.append(weight.narrow(axis, 0, 0))
        else:
            shards.append(weight.narrow(axis, start, end - start).contiguous())
    return shards


# Sequence parallel helpers
def seq_partition(x: torch.Tensor, tp_size: int, dim: int = 1):
    if tp_size <= 1:
        return [x], {"dim": dim, "sizes": [x.size(dim)]}
    n = x.size(dim)
    per = math.ceil(n / tp_size)
    parts = []
    sizes = []
    for i in range(tp_size):
        start = i * per
        end = min((i + 1) * per, n)
        sizes.append(max(end - start, 0))
        sl = [slice(None)] * x.ndim
        sl[dim] = slice(start, end)
        parts.append(x[tuple(sl)].contiguous())
    return parts, {"dim": dim, "sizes": sizes}


def seq_gather_restore(parts: list[torch.Tensor], meta: dict) -> torch.Tensor:
    dim = int(meta.get("dim", 1))
    return torch.cat(parts, dim=dim)


def seq_alltoall(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            chunks = list(x.chunk(world, dim=dim))
            out_chunks = [torch.empty_like(chunks[0]) for _ in range(world)]
            dist.all_to_all(out_chunks, chunks)
            return torch.cat(out_chunks, dim=dim)
    except Exception:
        warnings.warn("seq_alltoall fallback (no dist)")
    return x


# Expert/Tensor parallel extras
def shard_mlp_weight(weight: torch.Tensor, tp_size: int, axis: int = 0):
    return shard_linear_weight(weight, tp_size, axis=axis)


def expert_router_plan(gate_logits: torch.Tensor, capacity: int):
    # Top-1 gating plan: returns assignment indices and mask of overflows
    assign = gate_logits.argmax(dim=-1)
    # capacity control per expert (rough):
    B = assign.numel()
    overflow = torch.zeros_like(assign, dtype=torch.bool)
    return assign, overflow


# Activation partition helpers
def act_partition_plan(T: int, bytes_budget: int, bytes_per_token: int) -> list[int]:
    # Simple equal chunks that satisfy budget
    max_tokens = max(int(bytes_budget // max(bytes_per_token, 1)), 1)
    parts = []
    cur = 0
    while cur < T:
        parts.append(min(max_tokens, T - cur))
        cur += parts[-1]
    return parts


def reassemble_acts(chunks: list[torch.Tensor], dim: int = 1) -> torch.Tensor:
    return torch.cat(chunks, dim=dim)


# Per-token activation memory estimator (rough)
def estimate_activation_bytes_per_token(D: int, H: int, expand: int, dtype: str = "bf16") -> int:
    # Q,K,V,O ~ 4*D; MLP intermed ~ expand*D; residuals/others ~ ~D
    bytes_per = {"fp32": 4, "float32": 4, "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2}.get(dtype, 2)
    total = (4 * D) + (expand * D) + D
    return int(total) * int(bytes_per)

