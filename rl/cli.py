import argparse
import torch

from .config import RLConfig
from .trainer import Trainer
from specs.config import ModelConfig
from model.causal import CausalLM
from attn.backends import select_attention_backend


def make_batches(algo: str, device: torch.device, steps: int = 1000, seq_len: int = 16, vocab: int = 128):
    for _ in range(steps):
        if algo == "ppo":
            inp = torch.randint(0, vocab, (8, seq_len), device=device)
            act = torch.randint(0, vocab, (8, seq_len), device=device)
            old_lp = torch.zeros_like(inp, dtype=torch.float32, device=device)
            adv = torch.randn_like(old_lp)
            yield {
                "input_ids": inp,
                "actions": act,
                "old_logprobs": old_lp,
                "advantages": adv,
            }
        else:
            inp_p = torch.randint(0, vocab, (8, seq_len), device=device)
            inp_r = torch.randint(0, vocab, (8, seq_len), device=device)
            act_p = torch.randint(0, vocab, (8, seq_len), device=device)
            act_r = torch.randint(0, vocab, (8, seq_len), device=device)
            yield {
                "input_ids_pref": inp_p,
                "input_ids_rej": inp_r,
                "actions_pref": act_p,
                "actions_rej": act_r,
            }


def main():
    p = argparse.ArgumentParser("rl")
    sub = p.add_subparsers(dest="cmd", required=True)

    ppo = sub.add_parser("ppo", help="run a toy PPO loop")
    ppo.add_argument("--steps", type=int, default=1000)

    dpo = sub.add_parser("dpo", help="run a toy DPO loop")
    dpo.add_argument("--steps", type=int, default=1000)

    args = p.parse_args()
    algo = "ppo" if args.cmd == "ppo" else "dpo"
    cfg = RLConfig(algo=algo, steps=args.steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build a small CausalLM from existing model stack
    mcfg = ModelConfig(
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=1024,
        vocab_size=256,
        dtype="bfloat16" if torch.cuda.is_available() else "float32",
    )
    # Heuristically select attention backend and map to ModelConfig.attn_impl
    backend = select_attention_backend(is_causal=True, dtype=torch.bfloat16 if mcfg.dtype == "bfloat16" else torch.float32, seq=64, heads=mcfg.n_heads, device=device)
    mcfg.attn_impl = "flash" if backend == "flash2" else ("triton" if backend == "triton" else "eager")
    model = CausalLM(mcfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    trainer = Trainer(model, optim, cfg)
    metrics = trainer.train_steps(make_batches(cfg.algo, device, steps=cfg.steps), steps=cfg.steps)
    print({k: round(v, 6) for k, v in metrics.items()})


if __name__ == "__main__":
    main()


