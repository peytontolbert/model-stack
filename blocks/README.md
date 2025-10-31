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
- `blocks.transformer_block.TransformerBlock`: generic wiring
- `blocks.llama_block.LlamaBlock`: RMSNorm + SwiGLU + RoPE (prenorm)
- `blocks.gpt_block.GPTBlock`: LayerNorm + GELU (postnorm by default)
 - `blocks.parallel_block.ParallelTransformerBlock`: parallel residual wiring
 - `blocks.local_attn_block.LocalAttentionBlock`: sliding-window local attention
- `blocks.prefix_lm_block.PrefixLMBlock`: PrefixLM causal/prefix masking
- `blocks.banded_attn_block.BandedAttentionBlock`: Banded attention by |i-j| <= bandwidth
 - `blocks.dilated_local_attn_block.DilatedLocalAttentionBlock`: Dilated local attention
 - `blocks.block_sparse_attn_block.BlockSparseAttentionBlock`: Pattern-driven block-sparse attention
 - `blocks.cross_attn_block.CrossAttentionBlock`: decoder cross-attention block
 - `blocks.moe_block.MoEBlock`: MoE MLP variant with top-k routing

Initialization
```python
from blocks.init import init_transformer_stack
init_transformer_stack(model.blocks, recipe="llama")
```

Factory
```python
from blocks.factory import build_block_stack
stack = build_block_stack(cfg, variant="llama", drop_path_max=0.1, init_recipe="llama", residual_policy="deepnet")
# other variants: "gpt", "parallel", "local", "dilated", "blocksparse", "prefix", "banded", "cross", "moe", "encoder", "decoder"
```

Adapters
```python
from blocks.adapters import attach_adapters_to_block
blk = stack[0]
attach_adapters_to_block(blk, bottleneck=32, ia3=True, where="mlp")
```

Public API
```python
from blocks import build_block_stack, LlamaBlock, GPTBlock, ParallelTransformerBlock
```