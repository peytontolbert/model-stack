## sampling

Token sampling constraints and penalties for decoding.

### Penalties and temperatures
- `apply_temperature`, `apply_repetition_penalty`, `apply_presence_frequency_penalty`

### Top-k/p/typical/min-p/TFS
- `apply_topk_mask`, `apply_topp_mask`, `apply_typical_mask`, `apply_min_p_mask`, `apply_min_tokens_to_keep_mask`, `apply_tfs_mask`, `apply_eta_mask`

### Constraints and Gumbel tools
- `build_regex_constraint_mask`, `apply_no_repeat_ngram_mask`, `apply_stop_phrases_mask`, `json_schema_mask`, `cfgrammar_mask`
- `sample_gumbel`, `gumbel_topk`, `gumbel_softmax`


