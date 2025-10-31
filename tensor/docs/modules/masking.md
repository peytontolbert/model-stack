## masking

Mask creation and manipulation helpers for attention.

### Core masks
- `build_causal_mask`, `build_sliding_window_causal_mask`, `build_prefix_lm_mask`, `build_padding_mask`

### Ops and utils
- `apply_mask`, `broadcast_mask`, `attention_mask_from_lengths`, `lengths_from_attention_mask`
- `invert_mask`, `as_bool_mask`, `window_pattern_from_spans`

### Windowed masking (submodule)
- `build_block_causal_mask`, `build_dilated_causal_mask` (from `masking.windows`)


