## Transformers Llama model stack – parity checklist

Use this checklist to verify our local stack matches Hugging Face Transformers’ Llama implementation for inference.

### Configuration (LlamaConfig)
- [ ] hidden_size: model width (D)
- [ ] intermediate_size: MLP hidden (FF)
- [ ] num_hidden_layers: number of decoder layers (L)
- [ ] num_attention_heads: attention heads (H)
- [ ] num_key_value_heads: GQA/MQA heads (Hk)
- [ ] head_dim (optional): per-head dimension; defaults to hidden_size // num_attention_heads
- [ ] rms_norm_eps: epsilon for RMSNorm
- [ ] rope_parameters: { rope_theta, scaling params }
  - [ ] rope_theta
  - [ ] rope_scaling (type, factor)
  - [ ] original_max_position_embeddings
  - [ ] low_freq_factor / high_freq_factor (when applicable)
- [ ] attention_bias (default False)
- [ ] attention_dropout (default 0.0)
- [ ] mlp_bias (default False)
- [ ] use_cache (default True)
- [ ] tie_word_embeddings (default False)
- [ ] bos_token_id / eos_token_id / pad_token_id

### Core architecture (LlamaModel)
- [ ] Token embeddings: `nn.Embedding(vocab_size, hidden_size)`
- [ ] Decoder layers (repeat L):
  - [ ] Input RMSNorm: `LlamaRMSNorm(hidden_size, eps=rms_norm_eps)`
  - [ ] Self-attention: `LlamaAttention`
    - [ ] Linear projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`
    - [ ] Shapes: Q (H*Dh), K/V (Hk*Dh), O (H*Dh)
    - [ ] GQA fan-out: repeat K/V across groups (H / Hk)
    - [ ] Scale: `1/sqrt(head_dim)`
    - [ ] Dropout: `attention_dropout` (disabled in eval)
    - [ ] Bias flags respected via `attention_bias`
    - [ ] Rotary: `apply_rotary_pos_emb(query, key, cos, sin)`
  - [ ] Post-attn RMSNorm
  - [ ] MLP: gated SwiGLU (`gate_proj`, `up_proj`) + `down_proj`
    - [ ] Gate/Up fused along rows for `w_in`
    - [ ] Optional biases if `mlp_bias=True`
- [ ] Final RMSNorm: `LlamaRMSNorm`
- [ ] Rotary embedding provider: `LlamaRotaryEmbedding(config)` -> `(cos, sin)`

### Masking and positions
- [ ] Additive causal mask shape: (B, 1, T, S)
- [ ] Combines causal and padding masks
- [ ] `position_ids` default to `cache_position` (when cache is used) or range
- [ ] `cache_position` advances with past length

### KV cache
- [ ] `past_key_values` via `DynamicCache`
- [ ] Cache update in attention with RoPE-aware kwargs
- [ ] Works with grouped KV heads

### Attention backends
- [ ] `_attn_implementation` routing
  - [ ] Eager / SDPA / Flash / Flex supported by Transformers
- [ ] Scaling and masks forwarded to backend

### LM head and tying (LlamaForCausalLM)
- [ ] Base model + `lm_head: Linear(hidden_size, vocab_size, bias=False)`
- [ ] `_tied_weights_keys = ["lm_head.weight"]` (tying controlled by config)
- [ ] Forward slices logits with `logits_to_keep` for efficiency

### Generation (GenerationMixin)
- [ ] Greedy and sampling (do_sample)
- [ ] temperature, top_k, top_p
- [ ] repetition_penalty, no_repeat_ngram_size
- [ ] bad_words_ids / suppress tokens / forced tokens (optional)
- [ ] min/max length, early stopping, eos handling
- [ ] Beam search and diverse decoding (optional)

### Loader expectations (HF checkpoints)
- [ ] Embeddings and final RMSNorm weights
- [ ] Attention q/k/v/o weights (+ optional biases)
- [ ] MLP gate/up/down weights (gate+up fused for `w_in`) (+ optional biases)
- [ ] Shapes reflect GQA: K/V use `num_key_value_heads`

---

### Local stack parity quick-map

- Config: hidden_size→`d_model`, intermediate_size→`d_ff`, num_hidden_layers→`n_layers`, num_attention_heads→`n_heads`, num_key_value_heads→`n_kv_heads`, head_dim, `rms_norm_eps`, rope parameters, `attention_bias`, `attention_dropout`, `mlp_bias`.
- Blocks: RMSNorm, Llama attention (Q/K/V/O with GQA), SwiGLU MLP (gate+up fused), final RMSNorm.
- RoPE: base theta; scaling (linear, yarn), low/high frequency factors; returns `(cos, sin)`; applied to Q/K.
- Masking: additive causal (B,1,T,S), padding integrated, honors `position_ids` and `cache_position`.
- KV cache: paged cache; repeat-interleave K/V for GQA; sliding-window eviction (optional).
- Backends: routed to SDPA/Flash/XFormers/Torch with explicit scaling.
- Head: `lm_head` present; weight-tying supported by config.
- Generation: greedy + sampling with penalties; eos; optional sliding-window.
- Loader: HF safetensors mapped for embeddings, norms, attn Q/K/V/O, MLP gate/up/down; biases loaded when present; GQA shapes honored.

Use this as a pre-flight checklist whenever adding a new Llama checkpoint or tweaking parity-sensitive code.

### HF-specific support and integration details (from Transformers)
- LlamaPreTrainedModel support flags
  - supports_gradient_checkpointing
  - _no_split_modules = ["LlamaDecoderLayer"]
  - _skip_keys_device_placement = ["past_key_values"]
  - _supports_flash_attn, _supports_sdpa, _supports_flex_attn
  - _supports_attention_backend, _can_compile_fullgraph
- Config inference hints and parallel plans
  - keys_to_ignore_at_inference = ["past_key_values"]
  - base_model_tp_plan: colwise/rowwise splits for q/k/v/o and mlp
  - base_model_pp_plan: stage IO mapping for embed/layers/norm
  - LlamaForCausalLM: _tied_weights_keys = ["lm_head.weight"], _tp_plan={"lm_head":"colwise_rep"}, _pp_plan
- Rotary embedding selection
  - LlamaRotaryEmbedding can choose implementation by `rope_type` via ROPE_INIT_FUNCTIONS; default uses `rope_parameters["rope_theta"]`.


