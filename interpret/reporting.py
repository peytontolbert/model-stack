from __future__ import annotations

from typing import Iterable

import torch

from interpret.features.sae import SAEConfig
from interpret.probes.dataset import ProbeDataset, summarize_probe_dataset


def summarize_patch_sweep(
    names: Iterable[str],
    scores: torch.Tensor,
    *,
    topk: int = 10,
    unit_label: str = "unit_index",
    time_label: str = "time_index",
) -> dict[str, object]:
    names = list(names)
    if scores.ndim not in {2, 3}:
        raise ValueError("scores must have shape [N,T] or [N,U,T]")
    if scores.shape[0] != len(names):
        raise ValueError("names length must match scores.shape[0]")

    flat_scores = scores.reshape(scores.shape[0], -1)
    per_name_max, _ = flat_scores.max(dim=-1)
    ordered = torch.argsort(per_name_max, descending=True)
    top_entries: list[dict[str, object]] = []
    flat = scores.reshape(-1)
    kk = min(int(topk), int(flat.numel()))
    if kk > 0:
        topv, topi = torch.topk(flat, k=kk)
        for value, index in zip(topv.tolist(), topi.tolist()):
            idx = torch.tensor(index, dtype=torch.long)
            coords = [int(v) for v in torch.unravel_index(idx, scores.shape)]
            entry = {"name": names[coords[0]], "score": float(value)}
            if scores.ndim == 2:
                entry[time_label] = coords[1]
            else:
                entry[unit_label] = coords[1]
                entry[time_label] = coords[2]
            top_entries.append(entry)

    return {
        "shape": list(scores.shape),
        "global_max": float(scores.max().item()) if scores.numel() > 0 else 0.0,
        "best_by_name": [
            {"name": names[int(i)], "score": float(per_name_max[int(i)].item())}
            for i in ordered.tolist()
        ],
        "top_entries": top_entries,
    }


def summarize_path_patch_sweep(result: dict[str, object], *, topk: int = 10) -> dict[str, object]:
    source_modules = [str(x) for x in result["source_modules"]]
    receiver_modules = [str(x) for x in result["receiver_modules"]]
    target_restore = result["target_restore"]
    receiver_restore = result["receiver_restore"]
    if not isinstance(target_restore, torch.Tensor) or target_restore.ndim != 2:
        raise ValueError("target_restore must be a [S,R] tensor")
    if not isinstance(receiver_restore, torch.Tensor) or receiver_restore.shape != target_restore.shape:
        raise ValueError("receiver_restore must match target_restore shape")

    top_paths: list[dict[str, object]] = []
    flat = target_restore.reshape(-1)
    kk = min(int(topk), int(flat.numel()))
    if kk > 0:
        topv, topi = torch.topk(flat, k=kk)
        for value, index in zip(topv.tolist(), topi.tolist()):
            idx = torch.tensor(index, dtype=torch.long)
            source_idx, receiver_idx = [int(v) for v in torch.unravel_index(idx, target_restore.shape)]
            top_paths.append(
                {
                    "source_module": source_modules[source_idx],
                    "receiver_module": receiver_modules[receiver_idx],
                    "target_logit_restore_fraction": float(value),
                    "receiver_restore_fraction": float(receiver_restore[source_idx, receiver_idx].item()),
                }
            )

    return {
        "shape": list(target_restore.shape),
        "top_paths": top_paths,
        "best_source_modules": [
            {"source_module": source_modules[i], "score": float(target_restore[i].max().item())}
            for i in range(len(source_modules))
        ],
        "best_receiver_modules": [
            {"receiver_module": receiver_modules[j], "score": float(target_restore[:, j].max().item())}
            for j in range(len(receiver_modules))
        ],
    }


def summarize_sae_training(info: dict[str, object], *, cfg: SAEConfig | None = None) -> dict[str, object]:
    epochs_run = int(info.get("epochs_run", 0))
    requested_epochs = int(cfg.epochs) if cfg is not None else None
    summary = {
        "loss": float(info.get("loss", 0.0)),
        "reconstruction_mse": float(info.get("reconstruction_mse", 0.0)),
        "avg_code_l1": float(info.get("avg_code_l1", 0.0)),
        "max_code_activation": float(info.get("max_code_activation", 0.0)),
        "best_epoch": int(info.get("best_epoch", -1)),
        "epochs_run": epochs_run,
        "stopped_early": bool(info.get("stopped_early", False)),
    }
    if requested_epochs is not None:
        summary["requested_epochs"] = requested_epochs
    history = info.get("loss_history", [])
    if isinstance(history, list) and history:
        summary["last_loss"] = float(history[-1])
    return summary


def summarize_probe_training_split(
    dataset: ProbeDataset,
    *,
    val_fraction: float,
    train_rows: int,
    val_rows: int,
) -> dict[str, object]:
    dataset_summary = summarize_probe_dataset(dataset)
    return {
        "dataset": dataset_summary,
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "val_fraction": float(val_fraction),
    }


__all__ = [
    "summarize_patch_sweep",
    "summarize_path_patch_sweep",
    "summarize_probe_training_split",
    "summarize_sae_training",
]
