Multi-GPU FSDP training (launcher)

Requires multiple GPUs and torchrun. Runs the project training entrypoint with FSDP and bf16.

Commands

torchrun --nproc_per_node=8 -m train.run \
  --config cfgs/llm_small.yaml \
  --dist.strategy FSDP --dist.precision bf16 --dist.grad_ckpt true \
  --viz.backend csv --viz.profile_every_n_steps 500

Notes

- Adjust --nproc_per_node to your GPU count.
- For CPU-only testing, switch --dist.backend to gloo and --dist.precision to fp32.


