import torch
from tensor import accum_steps, microbatch_plan, bucket_sizes, tensor_shards, estimate_bytes, activation_bytes, roofline, flops_linear, params_count_linear, time_op


def test_schedule_memory_profile():
    assert accum_steps(1024, 128) == 8
    plan = microbatch_plan(100, memory_bytes=1024, bytes_per_token=16)
    assert len(plan) >= 1
    buckets = bucket_sizes(1000, max_bytes=100, item_bytes=4)
    assert sum(buckets) == 1000
    shards = tensor_shards((2, 8, 1024, 64), bytes_budget=10_000)
    assert "axis" in shards and "parts" in shards
    eb = estimate_bytes((10, 10), torch.float32)
    ab = activation_bytes([(10, 10)], [torch.float32])
    r = roofline(1e9, 1e8, {"peak_flops": 1e12, "peak_bw": 1e11})
    f = flops_linear(128, 256)
    p = params_count_linear(128, 256)
    to = time_op(lambda x: x + 1, torch.tensor(1.0))
    assert eb > 0 and ab > 0 and f > 0 and p > 0 and "mean" in to

