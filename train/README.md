Trainer and run helpers for model training

- Core features: trainer loops, optim/lr schedulers, AMP, DDP/FSDP/DeepSpeed via `dist`, gradient checkpointing
- New: validation hooks, early stopping, EMA weights, resume, WD-exempt param groups, viz logging of lr/metrics

Quick start

```python
from specs.config import ModelConfig
from specs.dist import DistConfig
from model.factory import build_model
from train.run import run_training

cfg = ModelConfig(d_model=512, n_heads=8, n_layers=6, d_ff=2048, vocab_size=32000)
dist = DistConfig(strategy="FSDP", precision="bf16", grad_ckpt=True)
model = build_model(cfg, task="causal-lm", block="llama")

# Your dataset should yield objects with .input_ids and optional .attn_mask
dataset = ...

run_training(model=model, dataset=dataset, dist_cfg=dist, batch_size=32, max_epochs=1)
```

Notes
- AMP: `precision` in `DistConfig` controls autocast and scaler (fp16 uses GradScaler, bf16 does not).
- DDP/FSDP/DeepSpeed: handled by `dist.DistributedEngine.wrap_model` and `wrap_loader`.
- Gradient checkpointing: set `grad_ckpt=True` in `DistConfig` (per-block checkpointing in `Trainer`).

Validation & early stopping

```python
from data.loader import build_dataloader
from eval.loop import evaluate_lm_next_token  # optional

val_dataset = ...
run_training(
  model=model,
  dataset=dataset,
  val_dataset=val_dataset,
  dist_cfg=dist,
  batch_size=32,
  validate_every_steps=500,
)
```

Trainer will run periodic validation (ppl/acc via `eval`) and log `val/ppl` to viz.

Resume & EMA

```python
tc = TrainConfig(resume_from="ckpts/last/trainer_state.pt", ema_decay=0.999)
trainer = Trainer(model, dist_cfg=dist, train_cfg=tc)
```
