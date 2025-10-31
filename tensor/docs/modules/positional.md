## positional

Rotary embeddings, ALiBi, and relative position bias utilities.

### RoPE
- `build_rope_cache`, `apply_rotary`, `apply_rotary_scaled`
- 2D: `build_rope_cache_2d`, `apply_rotary_2d`

### ALiBi / Relative positions
- `build_alibi_bias`, `alibi_slopes`, `fit_alibi_slopes`
- `build_relative_position_indices`, `relative_position_bias_from_table`, `relative_position_bucket`

### Other
- `build_sinusoidal_cache`, `rope_ntk_scaling`, `rope_yarn_factors`, `rescale_positions`


