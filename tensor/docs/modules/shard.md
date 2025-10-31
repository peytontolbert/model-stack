## shard

Tensor/model sharding utilities and FLOPs/bytes estimators.

### Partitioning and collectives
- `tp_linear_partition`, `kv_partition`, `shard_linear_weight`
- Collectives: `allreduce_`, `reduce_scatter`, `allgather`

### Sequence parallel helpers
- `seq_alltoall`, `seq_partition`, `seq_gather_restore`

### Estimators
- `attn_flops`, `mlp_flops`, `tensor_bytes`, `estimate_latency_attn`, `estimate_activation_bytes`, `estimate_activation_bytes_per_token`

### Expert/moe helpers
- `expert_router_plan`, `act_partition_plan`, `reassemble_acts`, `shard_mlp_weight`


