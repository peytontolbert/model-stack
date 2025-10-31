E2E demos & reproducible scripts

Tiny, small, base LMs; instruction-tuning, RLHF/SFT recipes; domain adapters; notebooks.

CI-verified smoke tests that exercise the graph.

examples/
  00_tiny_lm/
    - Train a tiny LM on shards (or synthetic fallback)
    - Run: python examples/00_tiny_lm/run.py
  01_sft_dialog/
    - Minimal supervised fine-tuning on toy dialog pairs
    - Run: python examples/01_sft_dialog/run.py
  02_int8_export/
    - Save a tiny model and export ONNX with int8 weight-only quant
    - Run: python examples/02_int8_export/run.py
  03_fsdp_8gpu/
    - Multi-GPU launcher instructions (torchrun + FSDP)
    - See: examples/03_fsdp_8gpu/README.md
  04_eval_coding/
    - Evaluate LM next-token metrics on shards (or synthetic fallback)
    - Run: python examples/04_eval_coding/run.py
  05_repo_adapters/
    - Generate repository-conditioned LoRA-style adapters for this repo
    - Run: python examples/05_repo_adapters/run.py
  06_repo_fast_weights/
    - Derive fast-weight hyperparameters conditioned on this repo
    - Run: python examples/06_repo_fast_weights/run.py
  07_tensor_numerics/
    - Use tensor metrics utilities (e.g., token accuracy)
    - Run: python examples/07_tensor_numerics/run.py
  08_model_generate/
    - Build a model and sample tokens with repo generate helpers
    - Run: python examples/08_model_generate/run.py
  09_interpret_logit_lens/
    - Inspect representations with logit-lens on selected layers
    - Run: python examples/09_interpret_logit_lens/run.py
  10_autotune_search/
    - Simple latency-based attention implementation selection demo
    - Run: python examples/10_autotune_search/run.py
  11_compress_quantize/
    - Apply weight-only int8 quantization via compress.apply
    - Run: python examples/11_compress_quantize/run.py
  12_data_tokenize_shard/
    - Tokenize and decode with the default tokenizer
    - Run: python examples/12_data_tokenize_shard/run.py
  13_repo_grounded_adapters/
    - Repo-conditioned adapters + on-the-fly subgraph adapters; grounded prompt with token diagnostics
    - Run: python examples/13_repo_grounded_adapters/run.py

See also: examples/ON_THE_FLY_ADAPTERS.md
