## compile

Graph/runtime helpers for safe masked ops and CUDA/NCCL stream/seed guards.

### APIs
- `allow_in_graph`, `masked_fill_where`, `infer_attn_shapes`
- Seed/stream guards: `graph_safe_seed`, `cuda_graph_seed_scope`, `record_stream_guard`, `nccl_stream_guard`
- Graph utilities: `overlap_copy_compute`, `cuda_graph_warmup`, `graph_replay_step`


