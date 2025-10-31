RL: Reinforcement Learning (HF-style scaffolding)

This package provides minimal scaffolding for RL fine-tuning:
- Config schemas in `config.py`
- Algorithms in `algorithms/` (PPO and DPO minimal reference)
- A simple `trainer.py` to run loops
- A CLI in `cli.py` for quick runs

Example:
```bash
python -m rl.cli ppo --steps 1000
```

