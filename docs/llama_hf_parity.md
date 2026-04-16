## LLaMA HF → Local Stack Parity Plan

Goal: Mirror Hugging Face Transformers LLaMA behavior 1:1 in our local model stack (`model/*`, `blocks/*`, `attn/*`, `tensor/*`). This document lists references, mapping, invariants, and a verification checklist.

### Primary HF References (read-only)
- `cloned-repos/transformers/src/transformers/models/llama/modeling_llama.py`
  - Classes: `LlamaRMSNorm`, `LlamaRotaryEmbedding`/`apply_rotary_pos_emb`, `LlamaAttention` (GQA + repeat_kv), `LlamaMLP` (gate/up/down), `LlamaDecoderLayer`, `LlamaModel`, `LlamaForCausalLM`
  - Cache semantics, dtype casts, scaling constants
- `cloned-repos/transformers/src/transformers/models/llama/configuration_llama.py`
- `cloned-repos/transformers/src/transformers/generation/utils.py` and `generation/logits_process.py`
- `cloned-repos/transformers/src/transformers/modeling_outputs.py` (e.g., `CausalLMOutputWithPast`)
- `cloned-repos/transformers/src/transformers/cache_utils.py` (Cache API expectations)

### Local Stack Counterparts (implementation sites)
- Blocks and attention
  - `blocks/llama_block.py` (inherits `TransformerBlock`), `blocks/transformer_block.py`
  - `attn/eager.py` (implements QKV, RoPE, SDPA path), `attn/interfaces.py`
  - `tensor/positional.py` (RoPE), `tensor/masking.py` (masks), `tensor/norms.py` (RMS)
  - `tensor/mlp.py` (SwiGLU-style MLP with `w_in` gated split and `w_out`)
- Model composition
  - `model/causal.py:CausalLM` (embed → blocks → norm → `lm_head`)
  - `model/factory.py`, `specs/config.py` (model config)
  - Optional: `attn/kv_cache.py` (paged cache), `compress/*` (quant/LoRA if needed)

### Config Field Mapping (HF → Local)
- hidden_size → `ModelConfig.d_model`
- num_hidden_layers → `ModelConfig.n_layers`
- num_attention_heads → `ModelConfig.n_heads`
- num_key_value_heads → (support in attention impl; GQA via repeat_kv)
- intermediate_size → `ModelConfig.d_ff`
- vocab_size → `ModelConfig.vocab_size`
- rope_theta/rope_scaling → `ModelConfig.rope_theta` (+ scaling behavior)
- rms_norm_eps → used in RMSNorm (match epsilon)

### Weight Key Mapping (HF → Local)
- `model.embed_tokens.weight` → `embed.weight`
- Attention (per layer `i`):
  - `model.layers.{i}.self_attn.q_proj.weight` → `blocks.{i}.attn.w_q.weight`
  - `model.layers.{i}.self_attn.k_proj.weight` → `blocks.{i}.attn.w_k.weight`
  - `model.layers.{i}.self_attn.v_proj.weight` → `blocks.{i}.attn.w_v.weight`
  - `model.layers.{i}.self_attn.o_proj.weight` → `blocks.{i}.attn.w_o.weight`
- MLP (SwiGLU):
  - `model.layers.{i}.mlp.gate_proj.weight` and `up_proj.weight` → concatenate along out_dim → `blocks.{i}.mlp.w_in.weight = [gate; up]`
  - `model.layers.{i}.mlp.down_proj.weight` → `blocks.{i}.mlp.w_out.weight`
- Norms:
  - `model.layers.{i}.input_layernorm.weight` → `blocks.{i}.n1.weight`
  - `model.layers.{i}.post_attention_layernorm.weight` → `blocks.{i}.n2.weight`
- Final:
  - `model.norm.weight` → `norm.weight`
  - `lm_head.weight` (tied) → `lm_head.weight` (tie to `embed.weight` if applicable)

### Behavioral Invariants to Match
- Attention
  - SDPA call shape and scaling by `1/sqrt(head_dim)`
  - GQA: repeat KV heads `repeat = H / Hk`
  - RoPE: identical sin/cos cache, base theta, per-position application
  - Causal/attention mask semantics (boolean vs additive)
- MLP
  - SwiGLU gating: `x = silu(a) * b`, with split order consistent with HF
- Norm & dtypes
  - RMSNorm epsilon value matches HF
  - Casting behavior for stability (float32 in key matmuls where HF does)
- Outputs
  - Provide `last_hidden_state`, `logits`; cache optional (later phase)
- Generation
  - Temperature, TopP, repetition penalty formula parity
  - EOS may be int or list; identical stop condition

### Loader: HF safetensors → Local State Dict
Plan:
1) Read `model.safetensors.index.json` to gather shard files
2) Load tensors, map keys via the table above
3) For MLP `w_in`, vertically stack `gate_proj` and `up_proj` along out_dim
4) Assign to local model state dict; assert shape matches for each tensor
5) Optionally verify embedding/head tying

### API Alignment (Local Model)
- `model/causal.py:CausalLM.forward` accepts `attention_mask` as an alias of `attn_mask` and supports `cache=None`
- Return an object/dict exposing `.logits` and `.last_hidden_state` when needed by runners

### Verification Checklist
- [ ] RMSNorm eps equals HF configuration
- [ ] RoPE embedding identical (spot-check a few positions)
- [ ] Attention outputs match HF for a fixed seed/input (no cache)
- [ ] MLP outputs match HF for a fixed input
- [ ] One full layer output parity
- [ ] End-to-end forward parity (L layers) on a short sequence (no cache)
- [ ] Logits parity within numerical tolerance (report max abs diff)
- [ ] Generation (greedy) same tokens for N steps
- [ ] Generation (sampling) parity under fixed seed/logits processors
- [ ] Cache-enabled decoding parity (phase 2)

### Numerics & Tolerances
- Record tolerances per component: attention, MLP, residual stream, logits
- Typical targets: `~1e-5` fp32, `~1e-3` bf16; document any deviations

### Test Plan
- Deterministic input: fixed token ids, masks, dtype, and seed
- Fixture to run HF vs Local side-by-side
- Report:
  - Per-layer hidden-state L2/∞ norms (delta)
  - Final logits top-1/top-k differences
  - Generation token-by-token match ratios

### Known Gaps / Phase 2
- HF Cache API compatibility (optional); current local path supports sliding-window KV or in-repo paged cache
- Exact HF memory/layout quirks (e.g., offload/sharding) not required for numeric parity

### Implementation Notes
- Our runner already supports a local backend: `--backend local` builds `CausalLM` and applies adapters to local module names
- Manual head projection path is implemented with `--head-device auto|cpu|same` to avoid GPU OOM deterministically

### Action Items
- [x] Implement safetensors weight loader (HF → Local) with key mapping and `w_in` stacking
- [x] Add `attention_mask` compatibility shim in `CausalLM.forward`
- [x] Build parity harness comparing HF vs Local on the same config/checkpoint
- [ ] Document any deviations and wire fixes in `blocks/*`, `attn/*`, `tensor/*`

