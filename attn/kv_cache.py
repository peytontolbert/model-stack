from typing import Optional
import torch


class PagedKVCache:
    def __init__(self, batch: int, n_layers: int, n_kv_heads: int, head_dim: int, pagesize: int, dtype: torch.dtype, device: torch.device):
        self.batch = batch
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.pagesize = pagesize
        self.dtype = dtype
        self.device = device
        # pages[layer][batch] -> list of pages (K/V tensors)
        self.K = [[[] for _ in range(batch)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(batch)] for _ in range(n_layers)]
        self.lengths = [[0 for _ in range(batch)] for _ in range(n_layers)]

    def append(self, layer_idx: int, b: int, k_chunk: torch.Tensor, v_chunk: torch.Tensor):
        # k_chunk/v_chunk: (Hk, T, Dh)
        self.K[layer_idx][b].append(k_chunk.to(device=self.device, dtype=self.dtype).contiguous())
        self.V[layer_idx][b].append(v_chunk.to(device=self.device, dtype=self.dtype).contiguous())
        self.lengths[layer_idx][b] += k_chunk.shape[1]

    def slice(self, layer_idx: int, b: int, start: int, end: int):
        # Collect across pages; simple linear gather for clarity
        t0 = 0
        ks, vs = [], []
        for pk, pv in zip(self.K[layer_idx][b], self.V[layer_idx][b]):
            t1 = t0 + pk.shape[1]
            s = max(start, t0)
            e = min(end, t1)
            if s < e:
                ks.append(pk[:, (s - t0):(e - t0), :])
                vs.append(pv[:, (s - t0):(e - t0), :])
            t0 = t1
            if t0 >= end:
                break
        return (torch.cat(ks, dim=1) if ks else None, torch.cat(vs, dim=1) if vs else None)

    def evict(self, max_tokens: int, policy: str = "fifo"):
        for l in range(self.n_layers):
            for b in range(self.batch):
                while self.lengths[l][b] > max_tokens and self.K[l][b]:
                    if policy in ("fifo", "sliding-window"):
                        pk = self.K[l][b].pop(0)
                        pv = self.V[l][b].pop(0)
                        self.lengths[l][b] -= pk.shape[1]
                    elif policy == "lru":
                        # no usage tracking here; treat same as fifo
                        pk = self.K[l][b].pop(0)
                        pv = self.V[l][b].pop(0)
                        self.lengths[l][b] -= pk.shape[1]

    # New: return a per-layer batched KV cache view implementing append/read/length
    def layer(self, layer_idx: int):
        return _LayerCacheView(self, layer_idx)


def init_kv_cache(batch: int, n_layers: int, n_kv_heads: int, head_dim: int, pagesize: int, dtype: torch.dtype, device: torch.device) -> PagedKVCache:
    return PagedKVCache(batch, n_layers, n_kv_heads, head_dim, pagesize, dtype, device)


def kv_cache_append(cache: PagedKVCache, layer_idx: int, k_chunk: torch.Tensor, v_chunk: torch.Tensor, block_ids: Optional[torch.Tensor] = None):
    # block_ids: (B,) which batch entries to append to; default all 0..B-1
    B = k_chunk.shape[0]
    if block_ids is None:
        block_ids = torch.arange(B, device=k_chunk.device)
    for i, b in enumerate(block_ids.tolist()):
        cache.append(layer_idx, b, k_chunk[i], v_chunk[i])


def kv_cache_slice(cache: PagedKVCache, layer_idx: int, start: int, end: int):
    Ks, Vs = [], []
    for b in range(cache.batch):
        k, v = cache.slice(layer_idx, b, start, end)
        Ks.append(k)
        Vs.append(v)
    return Ks, Vs


def kv_cache_evict(cache: PagedKVCache, max_tokens: int, policy: str = "fifo"):
    cache.evict(max_tokens, policy)



class _LayerCacheView:
    """A batched per-layer KV cache view exposing append/read/length.

    Expects/returns tensors shaped as (B, H, T, D) for K and V.
    """

    def __init__(self, parent: PagedKVCache, layer_idx: int) -> None:
        self.parent = parent
        self.layer_idx = int(layer_idx)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        # k, v: (B, H, T, D)
        assert k.dim() == 4 and v.dim() == 4, "KV must be (B,H,T,D)"
        assert k.shape == v.shape, "K and V shapes must match"
        B, H, T, D = k.shape
        for b in range(B):
            self.parent.append(self.layer_idx, b, k[b], v[b])

    def read(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        Ks: list[torch.Tensor] = []
        Vs: list[torch.Tensor] = []
        for b in range(self.parent.batch):
            k, v = self.parent.slice(self.layer_idx, b, int(start), int(end))
            if k is None:
                Ks.append(torch.empty(0, self.parent.n_kv_heads, self.parent.head_dim, dtype=self.parent.dtype, device=self.parent.device))
                Vs.append(torch.empty(0, self.parent.n_kv_heads, self.parent.head_dim, dtype=self.parent.dtype, device=self.parent.device))
            else:
                Ks.append(k)
                Vs.append(v)
        # Stack along batch: current layout is (H, T, D); make (B, H, T, D)
        if len(Ks) == 0 or Ks[0].numel() == 0:
            return (
                torch.empty(self.parent.batch, self.parent.n_kv_heads, 0, self.parent.head_dim, dtype=self.parent.dtype, device=self.parent.device),
                torch.empty(self.parent.batch, self.parent.n_kv_heads, 0, self.parent.head_dim, dtype=self.parent.dtype, device=self.parent.device),
            )
        Kb = torch.stack(Ks, dim=0)
        Vb = torch.stack(Vs, dim=0)
        return Kb, Vb

    def length(self) -> int:
        # Return the maximum sequence length across batch for this layer
        return max(self.parent.lengths[self.layer_idx]) if self.parent.lengths[self.layer_idx] else 0

