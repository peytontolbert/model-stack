# train/trainer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional
import os
from contextlib import nullcontext

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from specs.dist import DistConfig
from tensor.optim import (
    clip_by_policy_,
    schedule_cosine_with_warmup,
    schedule_linear_with_warmup,
)


@dataclass
class TrainConfig:
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    # Scheduling
    scheduler: str = "cosine"  # "cosine" | "linear" | "none"
    warmup_steps: int = 1000
    total_steps: int = 100000
    min_lr_ratio: float = 0.0  # only for cosine
    # Training loop
    accumulate_steps: int = 1
    clip_grad_norm: Optional[float] = 1.0
    log_interval: int = 10
    # Checkpointing
    save_every_steps: int = 0  # 0 disables periodic saving
    save_dir: Optional[str] = None
    save_best: bool = True
    resume_from: Optional[str] = None
    # Validation
    validate_every_steps: int = 0  # 0 disables periodic validation
    early_stop_patience: int = 0   # 0 disables early stopping
    early_stop_mode: str = "min"   # "min" (for ppl/loss) or "max"
    # EMA
    ema_decay: Optional[float] = None  # e.g., 0.999
    # Optim param groups
    weight_decay_exempt: bool = True   # exclude biases and LayerNorm weights from weight decay


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        dist_cfg: DistConfig,
        train_cfg: TrainConfig,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        viz_session: Optional["VizSession"] = None,
    ) -> None:
        self.model = model
        self.dist_cfg = dist_cfg
        self.cfg = train_cfg
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        self.viz = viz_session

        # AMP scaler only for fp16; bf16 uses autocast without scaling
        use_fp16 = (dist_cfg.precision == "fp16")
        self.scaler = GradScaler(enabled=use_fp16)

        # Optimizer with optional param grouping
        if self.cfg.weight_decay_exempt:
            decay, no_decay = [], []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.endswith(".bias") or ("norm" in name.lower()):
                    no_decay.append(p)
                else:
                    decay.append(p)
            param_groups = [
                {"params": decay, "weight_decay": float(self.cfg.weight_decay)},
                {"params": no_decay, "weight_decay": 0.0},
            ]
        else:
            param_groups = list(model.parameters())
        self.opt = optim.AdamW(
            param_groups,
            lr=self.cfg.learning_rate,
            betas=self.cfg.betas,
            eps=self.cfg.eps,
        )

        # Step counters
        self.global_step: int = 0
        self._accum_step: int = 0
        self._best_metric: Optional[float] = None
        self._stale: int = 0

        # Exponential Moving Average (EMA) of parameters
        self._ema_decay = float(self.cfg.ema_decay) if self.cfg.ema_decay is not None else None
        self._ema_state: Optional[dict[str, torch.Tensor]] = None
        if self._ema_decay is not None:
            self._ema_state = {k: p.detach().clone() for k, p in self.model.state_dict().items()}

        # Optionally resume
        if self.cfg.resume_from is not None:
            self._maybe_resume(self.cfg.resume_from)

    # -------------------------
    # LR scheduling (manual per-step factor)
    # -------------------------
    def _lr_scale(self, step: int) -> float:
        if self.cfg.scheduler == "cosine":
            return schedule_cosine_with_warmup(
                step, self.cfg.warmup_steps, self.cfg.total_steps, self.cfg.min_lr_ratio
            )
        if self.cfg.scheduler == "linear":
            return schedule_linear_with_warmup(step, self.cfg.warmup_steps, self.cfg.total_steps)
        return 1.0

    def _apply_lr(self, base_lr: float, scale: float) -> None:
        val = float(base_lr) * float(scale)
        for g in self.opt.param_groups:
            g["lr"] = val

    def _lr_current(self) -> float:
        try:
            return float(self.opt.param_groups[0]["lr"])  # type: ignore[index]
        except Exception:
            return float(self.cfg.learning_rate)

    # -------------------------
    # Forward helpers (supports activation checkpointing)
    # -------------------------
    def _forward_logits(self, batch) -> torch.Tensor:
        # Expected batch has .input_ids and optional .attn_mask
        if getattr(self.dist_cfg, "grad_ckpt", False) and hasattr(self.model, "blocks"):
            # Per-block checkpointing preserves additional args
            from torch.utils.checkpoint import checkpoint
            x = self.model.embed(batch.input_ids)
            attn_mask = getattr(batch, "attn_mask", None)
            for blk in self.model.blocks:
                def _wrap(inp, module=blk, mask=attn_mask):
                    return module(inp, mask, None)
                x = checkpoint(_wrap, x)
            x = self.model.norm(x)
            logits = self.model.lm_head(x)
            return logits
        # Fallback to model forward
        return self.model(batch.input_ids, getattr(batch, "attn_mask", None))

    # -------------------------
    # One optimization step (supports grad accumulation)
    # -------------------------
    def step(self, batch) -> float:
        if self._accum_step == 0:
            self.opt.zero_grad(set_to_none=True)

        # Select AMP autocast context by precision
        if self.dist_cfg.precision == "bf16":
            amp_ctx = autocast(dtype=torch.bfloat16)
        elif self.dist_cfg.precision == "fp16":
            amp_ctx = autocast(dtype=torch.float16)
        else:
            amp_ctx = nullcontext()

        # For DDP, avoid gradient synchronization during accumulation micro-steps
        should_accum = (self._accum_step + 1) < max(1, int(self.cfg.accumulate_steps))
        no_sync = getattr(self.model, "no_sync", None)
        sync_ctx = no_sync() if (callable(no_sync) and should_accum) else nullcontext()

        with amp_ctx:
            logits = self._forward_logits(batch)
            # Language modeling loss (shifted)
            vocab = logits.size(-1)
            loss = self.loss_fn(
                logits[..., :-1, :].contiguous().view(-1, vocab),
                batch.input_ids[..., 1:].contiguous().view(-1),
            )

        # Scale if needed and backward
        with sync_ctx:
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        self._accum_step += 1
        took_step = False
        if self._accum_step >= max(1, int(self.cfg.accumulate_steps)):
            # Unscale for clipping when using scaler
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.opt)
            # Clip by global norm if configured
            if self.cfg.clip_grad_norm and self.cfg.clip_grad_norm > 0:
                clip_by_policy_(self.model.parameters(), max_norm=float(self.cfg.clip_grad_norm))

            # LR schedule update
            scale = self._lr_scale(self.global_step)
            self._apply_lr(self.cfg.learning_rate, scale)

            # Optimizer step
            if self.scaler.is_enabled():
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                self.opt.step()

            # EMA update
            if self._ema_decay is not None and self._ema_state is not None:
                d = self._ema_decay
                with torch.no_grad():
                    for k, v in self.model.state_dict().items():
                        self._ema_state[k].mul_(d).add_(v, alpha=(1.0 - d))

            self.global_step += 1
            self._accum_step = 0
            took_step = True

        loss_val = float(loss.detach().item())
        if self.viz is not None and (self.global_step % max(1, int(self.cfg.log_interval)) == 0) and took_step:
            self.viz.log_scalar(self.global_step, "train/loss", loss_val)
            self.viz.log_scalar(self.global_step, "train/lr", self._lr_current())
        if took_step:
            self._maybe_save(loss_val)
        return loss_val

    # -------------------------
    # Checkpointing helpers
    # -------------------------
    def _is_rank0(self) -> bool:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
        except Exception:
            pass
        # Fallback to env var used by launcher
        return int(os.getenv("RANK", "0")) == 0

    def _maybe_save(self, loss_val: float) -> None:
        if not self.cfg.save_dir:
            return
        if self.cfg.save_every_steps and self.cfg.save_every_steps > 0:
            if self.global_step % int(self.cfg.save_every_steps) == 0:
                self._save_checkpoint(f"step{self.global_step:07d}")
        if self.cfg.save_best:
            best = getattr(self, "_best_loss", None)
            if best is None or loss_val < float(best):
                setattr(self, "_best_loss", float(loss_val))
                self._save_checkpoint("best")

    def _unwrap_model(self) -> torch.nn.Module:
        m = self.model
        # unwrap .module repeatedly (DDP/FSDP/DeepSpeed)
        while hasattr(m, "module"):
            try:
                m = m.module  # type: ignore[attr-defined]
            except Exception:
                break
        return m

    def _save_checkpoint(self, tag: str) -> None:
        if not self._is_rank0():
            return
        try:
            from pathlib import Path
            from model.checkpoint import save_pretrained
            m = self._unwrap_model()
            cfg = getattr(m, "cfg", None)
            out_dir = Path(self.cfg.save_dir) / str(tag)
            if cfg is not None:
                save_pretrained(m, cfg, str(out_dir))
            else:
                # fallback to plain state_dict
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(m.state_dict(), out_dir / "model.pt")
            # Save trainer state (optimizer/scaler/step/ema)
            state = {
                "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
                "global_step": self.global_step,
                "ema": self._ema_state,
            }
            torch.save(state, out_dir / "trainer_state.pt")
            if self.viz is not None:
                try:
                    self.viz.log_artifact(out_dir, name=f"ckpt_{tag}")
                except Exception:
                    pass
        except Exception:
            # best-effort only; ignore save errors to not break training
            pass

    def _maybe_resume(self, path: str) -> None:
        try:
            from pathlib import Path
            p = Path(path)
            if p.is_dir():
                state_path = p / "trainer_state.pt"
            else:
                state_path = p
            if state_path.exists():
                state = torch.load(state_path, map_location="cpu")
                self.opt.load_state_dict(state.get("opt", {}))
                sc = state.get("scaler")
                if sc is not None and self.scaler.is_enabled():
                    self.scaler.load_state_dict(sc)
                self.global_step = int(state.get("global_step", 0))
                ema = state.get("ema")
                if ema is not None and self._ema_state is not None:
                    self._ema_state = {k: v.to(next(self.model.parameters()).device) for k, v in ema.items()}
        except Exception:
            pass

    # -------------------------
    # Epoch/Steps loops
    # -------------------------
    def train_steps(self, loader, max_steps: int, *, val_loader=None) -> None:
        for _ in range(max_steps):
            try:
                batch = next(self._it)
            except AttributeError:
                self._it = iter(loader)
                batch = next(self._it)
            except StopIteration:
                self._it = iter(loader)
                batch = next(self._it)
            self.step(batch)
            # Periodic validation
            if val_loader is not None and self.cfg.validate_every_steps and (self.global_step % int(self.cfg.validate_every_steps) == 0):
                self._run_validation(val_loader)
                if self._should_early_stop():
                    break

    def train_epochs(self, loader, max_epochs: int, *, val_loader=None) -> None:
        for _ in range(max_epochs):
            for batch in loader:
                self.step(batch)
                if val_loader is not None and self.cfg.validate_every_steps and (self.global_step % int(self.cfg.validate_every_steps) == 0):
                    self._run_validation(val_loader)
                    if self._should_early_stop():
                        return

    # -------------------------
    # Validation & Early stopping
    # -------------------------
    def _run_validation(self, val_loader) -> None:
        try:
            from eval.loop import evaluate_lm_next_token  # type: ignore
        except Exception:
            return
        # Swap to EMA weights for evaluation if available
        ctx = nullcontext()
        if self._ema_state is not None:
            ctx = _swap_state_dict(self.model, self._ema_state)
        with ctx:
            res = evaluate_lm_next_token(self.model, val_loader, device=next(self.model.parameters()).device, max_batches=None, report_accuracy=True)
        metric = float(res.ppl)
        better = (metric < (self._best_metric if self._best_metric is not None else float("inf"))) if self.cfg.early_stop_mode == "min" else (metric > (self._best_metric if self._best_metric is not None else float("-inf")))
        if better:
            self._best_metric = metric
            self._stale = 0
            # Save best validation checkpoint
            if self.cfg.save_dir and self.cfg.save_best:
                self._save_checkpoint("best_val")
        else:
            self._stale += 1
        if self.viz is not None:
            self.viz.log_scalar(self.global_step, "val/ppl", float(res.ppl))
            if res.acc is not None:
                self.viz.log_scalar(self.global_step, "val/acc", float(res.acc))

    def _should_early_stop(self) -> bool:
        return bool(self.cfg.early_stop_patience and self._stale >= int(self.cfg.early_stop_patience))


# Utility: swap model state_dict temporarily
from contextlib import contextmanager

@contextmanager
def _swap_state_dict(model: torch.nn.Module, state: dict[str, torch.Tensor]):
    orig = {k: v.clone() for k, v in model.state_dict().items()}
    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(orig, strict=False)
