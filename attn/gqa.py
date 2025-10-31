import torch
from tensor.shape import split_heads, merge_heads


def share_kv_heads(q_heads: int, kv_heads: int):
    # Map each query head to a kv head index
    import math
    factor = math.ceil(q_heads / kv_heads)
    return [min(i // factor, kv_heads - 1) for i in range(q_heads)]


def rearrange_qkv_for_mqa_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n_q: int, n_kv: int):
    # q: (B,T,Dq), k/v: (B,T,Dkv) -> (B,Hq,T,Dhq), (B,Hkv,T,Dhkv)
    qh = split_heads(q, n_q)
    kh = split_heads(k, n_kv)
    vh = split_heads(v, n_kv)
    return qh, kh, vh


