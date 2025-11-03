Autotune: Hyperparameter and Systems Tuning

This module provides a lightweight but comprehensive framework for automated tuning across model, runtime, and kernel-level knobs. It integrates with existing components like `viz`, `attn`, and `dist` while remaining framework-agnostic for your objective function.

Highlights
- Random, grid, Sobol quasi-random, and Latin hypercube search (pluggable)
- Simple, composable search spaces (choice, int ranges, uniform, log-uniform)
- Trial orchestration with persistent local storage
- Viz logging integration for trial metrics
- Microbenchmark utility for attention backend selection
- CLI for running studies and microbenches

Quick Start (Python API)
```python
from autotune import Study, StudyConfig
from autotune.spaces import SearchSpace, Choice, IntRange, LogUniform
from autotune.algorithms.random import RandomSearch

space = SearchSpace({
    "lr": LogUniform(1e-5, 5e-3),
    "batch_size": Choice([8, 16, 32]),
    "attn_backend": Choice(["torch", "xformers", "flash2"]),
    "seq_len": IntRange(128, 513, 128),
})

def objective(params, trial):
    # Train/eval or run inference; return scalar to minimize/maximize
    # Example (pseudo): loss = train_eval_model(params)
    # return loss
    return 1.0  # placeholder

cfg = StudyConfig(metric="loss", mode="min", max_trials=20, log_dir=".autotune")
study = Study(cfg, space, RandomSearch(seed=1337))
best = study.optimize(objective)
print("best params:", best)
```

CLI Usage
```bash
# Define search space in JSON
cat > space.json << 'JSON'
{
  "lr": {"type": "loguniform", "low": 1e-5, "high": 5e-3},
  "batch_size": {"type": "choice", "options": [8, 16, 32]},
  "attn_backend": {"type": "choice", "options": ["torch", "xformers", "flash2"]}
}
JSON

# Run a study (objective is a Python callable: module:function)
python -m autotune.cli study \
  --objective mypkg.objectives:train_eval \
  --space space.json \
  --algo random \
  --metric loss --mode min \
  --max-trials 25 \
  --log-dir .autotune \
  --write-best-to .autotune/best.json

# Benchmark attention backends for your environment
python -m autotune.cli bench-attn --seq 512 --heads 16 --d-k 64 --dtype bf16
```

Available algorithms for `--algo`: `random`, `grid`, `sobol`, `lhs`.

Example using Sobol quasi-random search:
```bash
python -m autotune.cli study \
  --objective mypkg.objectives:train_eval \
  --space space.json \
  --algo sobol \
  --metric loss --mode min \
  --max-trials 25
```

Integrations and Tips
- Viz: trial results are logged as `autotune.<metric>` in `.autotune/scalars.csv` (or via `viz` if available).
- Attention backend selection: use `autotune.bench.attn.select_fastest_backend(...)` to pick `torch`/`xformers`/`flash2` for your shapes.
- Distributed & precision: include knobs like `precision` (fp16/bf16/fp32), `strategy` (DDP/FSDP/DeepSpeed), and use them inside your objective to configure `dist.engine.DistributedEngine`.
- Export: after tuning, pass best params to your export config or write to a JSON via `write_best_to`.

Extending
- Implement new searchers by adding a class with `reset(space, seed)` and `suggestions(space, max_trials)`.
- Add callbacks by providing a `callable(Trial) -> None` and passing to `Study(..., callbacks=[...])`.

Notes
- This module does not assume a specific training loop—your `objective(params, trial)` is responsible for running the workload and returning a scalar score.
- The default storage writes each trial to `.autotune/trials/trial_XXXXX.json` and the current best to `.autotune/best.json`.