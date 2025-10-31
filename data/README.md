data: tokenization bridges, iterable datasets, sharding, mmap/streaming collation

Quickstart

```python
from data.loader import build_dataloader

dl = build_dataloader("/corpus/shards", batch_size=8, seq_len=1024)
for batch in dl:
    # batch.input_ids: (B, T) on CUDA if available
    # batch.attn_mask: None (model uses internal causal mask)
    ...
```

Streaming (iterable)

```python
from data import build_streaming_dataloader

dl = build_streaming_dataloader("/corpus/shards", batch_size=8, seq_len=1024)
```

Shard formats

- .bin: contiguous int32 tokens (memory-mapped)
- .npy: 1D NumPy array of ints (memory-mapped)
- .txt: space-separated ints per line (loaded into memory)

API

- data.Batch: dataclass with `input_ids` (B, T) and optional `attn_mask`
- data.build_dataloader(path, batch_size, seq_len, streaming=False, distributed=False, ...)
- data.build_streaming_dataloader(path, batch_size, seq_len, ...)
- data.ShardedTokenDataset: map-style dataset over shard windows
- data.get_tokenizer(name_or_path): HF or whitespace fallback

Notes

- Windows are non-overlapping of length T=seq_len. Targets are computed by shift in the trainer/model.
- Dataloader moves tensors to CUDA automatically if available; pass device=... to override.