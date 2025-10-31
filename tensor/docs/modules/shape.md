## shape

Shape transforms, padding, and assertions.

### Splits and merges
- `split_heads`, `merge_heads`, `split_qkv`, `merge_qkv`, `split_gqa_heads`, `merge_gqa_heads`

### Padding and layouts
- `center_pad`, `pad_to_multiple`, `right_trim_to`, `ensure_contiguous_lastdim`, `reorder_to_channels_last_2d`

### Assertions and introspection
- `assert_mask_shape`, `assert_boolean_mask`, `assert_broadcastable`, `expect_shape`, `same_shape`, `enforce_static_shape`, `trace_shape`, `expect_memory_format`, `stride_equal`, `is_view_of`


