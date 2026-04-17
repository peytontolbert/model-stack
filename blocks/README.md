Transformer blocks (norms, MLPs, residuals, pre/post-norm wiring)

Features
- Prenorm/Postnorm policies
- RMSNorm/LayerNorm
- MHA/GQA via `n_kv_heads`
- SwiGLU/GELU/SiLU MLPs with dropout
- RoPE and ALiBi integration
- Residual and stochastic-depth drop paths
- Checkpoint-friendly forward

Quickstart
```python
from specs.config import ModelConfig
from blocks.examples.example_lm import ExampleTransformerLM

cfg = ModelConfig(d_model=512, n_heads=8, n_layers=4, d_ff=2048, vocab_size=32000)
model = ExampleTransformerLM(cfg, block="llama").cuda()
```

Blocks
- `runtime.block_modules.TransformerBlock`: generic wiring
- `runtime.block_modules.LlamaBlock`: RMSNorm + SwiGLU + RoPE (prenorm)
- `runtime.block_modules.GPTBlock`: LayerNorm + GELU (postnorm by default)
- `runtime.block_modules.ParallelTransformerBlock`: parallel residual wiring
- `runtime.local_attn_block.LocalAttentionBlock`: sliding-window local attention
- `runtime.prefix_lm_block.PrefixLMBlock`: PrefixLM causal/prefix masking
- `runtime.banded_attn_block.BandedAttentionBlock`: Banded attention by |i-j| <= bandwidth
- `runtime.dilated_local_attn_block.DilatedLocalAttentionBlock`: Dilated local attention
- `runtime.block_sparse_attn_block.BlockSparseAttentionBlock`: Pattern-driven block-sparse attention
- `runtime.segment_bidir_attn_block.SegmentBidirAttentionBlock`: bidirectional attention within segments
- `runtime.window_pattern_attn_block.WindowPatternAttentionBlock`: custom window-span attention
- `runtime.strided_attn_block.StridedAttentionBlock`: strided sparse attention
- `runtime.block_modules.CrossAttentionBlock`: decoder cross-attention block
- `runtime.block_modules.MoEBlock`: MoE MLP variant with top-k routing

Initialization
```python
from runtime.block_init import init_transformer_stack
init_transformer_stack(model.blocks, recipe="llama")
```

Factory
```python
from runtime.block_factory import build_block_stack
stack = build_block_stack(cfg, variant="llama", drop_path_max=0.1, init_recipe="llama", residual_policy="deepnet")
# other variants: "gpt", "parallel", "local", "dilated", "blocksparse", "prefix", "banded", "cross", "moe", "encoder", "decoder"
```

Adapters
```python
from runtime.block_adapters import attach_adapters_to_block
blk = stack[0]
attach_adapters_to_block(blk, bottleneck=32, ia3=True, where="mlp")
```

Public API
```python
from blocks import build_block_stack, LlamaBlock, GPTBlock, ParallelTransformerBlock
```

Notes
- `blocks.factory` is a compatibility alias over `runtime.block_factory`.
- `blocks.transformer_block`, `blocks.llama_block`, `blocks.gpt_block`, `blocks.parallel_block`, `blocks.cross_attn_block`, and `blocks.moe_block` are compatibility aliases over `runtime.block_modules`.
- `blocks.shared`, `blocks.local_attn_block`, `blocks.prefix_lm_block`, `blocks.banded_attn_block`, `blocks.dilated_local_attn_block`, `blocks.block_sparse_attn_block`, `blocks.segment_bidir_attn_block`, `blocks.window_pattern_attn_block`, and `blocks.strided_attn_block` are compatibility aliases over their corresponding `runtime.*` modules.
- `blocks.config`, `blocks.init`, `blocks.schedules`, `blocks.policies`, and `blocks.targets` are compatibility aliases over `runtime.block_config`, `runtime.block_init`, `runtime.block_schedules`, `runtime.block_policies`, and `runtime.block_targets`.
- `blocks.stack`, `blocks.adapters`, and `blocks.utils` are compatibility aliases over `runtime.block_stack`, `runtime.block_adapters`, and `runtime.block_utils`.
