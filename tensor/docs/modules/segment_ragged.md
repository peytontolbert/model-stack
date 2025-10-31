## Segment and ragged operations

APIs: `segment_sum/mean/max/min`, `ragged_gather`, `ragged_scatter`.

### Notes
- Segment ops reduce over the leading dimension keyed by `segments` and optionally accept `n_segments` for preallocation.
- Ragged gather/scatter operate on concatenated sequences with `offsets` and `lengths`.


