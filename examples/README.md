## Examples

End-to-end demos and reproducible scripts for training, evaluation, export, interpretation, compression, and adapter workflows.

### Scripted Examples

- `00_tiny_lm/`: tiny LM training smoke
  Run: `python examples/00_tiny_lm/run.py`
- `01_sft_dialog/`: minimal supervised fine-tuning on toy dialogs
  Run: `python examples/01_sft_dialog/run.py`
- `02_int8_export/`: tiny-model export and quantization smoke
  Run: `python examples/02_int8_export/run.py`
- `03_fsdp_8gpu/`: multi-GPU launcher notes
  See: [03_fsdp_8gpu/README.md](03_fsdp_8gpu/README.md)
- `04_eval_coding/`: next-token evaluation smoke
  Run: `python examples/04_eval_coding/run.py`
- `05_repo_adapters/`: repository-conditioned adapter demo
  Run: `python examples/05_repo_adapters/run.py`
- `06_repo_fast_weights/`: repository-conditioned fast-weight demo
  Run: `python examples/06_repo_fast_weights/run.py`
- `07_tensor_numerics/`: tensor metrics and numerics helpers
  Run: `python examples/07_tensor_numerics/run.py`
- `08_model_generate/`: model build and generation helpers
  Run: `python examples/08_model_generate/run.py`
- `09_interpret_logit_lens/`: logit-lens inspection
  Run: `python examples/09_interpret_logit_lens/run.py`
- `10_autotune_search/`: latency-based attention search demo
  Run: `python examples/10_autotune_search/run.py`
- `11_compress_quantize/`: weight-only INT8 quantization demo
  Run: `python examples/11_compress_quantize/run.py`
- `12_data_tokenize_shard/`: tokenizer encode/decode smoke
  Run: `python examples/12_data_tokenize_shard/run.py`
- `13_parameter_golf_h100/`: H100 Parameter Golf ternary train/export recipe
  See: [13_parameter_golf_h100/README.md](13_parameter_golf_h100/README.md)

### Larger Example Trees

- [repo_grounded_adapters/README.md](repo_grounded_adapters/README.md): repository-grounded adapter workflow
- [program_conditioned_adapter/README.md](program_conditioned_adapter/README.md): generalized program-conditioned adapter workflow

The legacy `13_repo_grounded_adapters/` directory is not the maintained entry point; use `examples/repo_grounded_adapters/` instead.
