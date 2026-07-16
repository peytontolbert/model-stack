from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

import torch
import torch.nn.functional as F
class InstructionDataGuardNet(torch.nn.Module):
    def __init__(self, input_dim: int = 4096, dropout: float = 0.7) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.dropout = torch.nn.Dropout(float(dropout))
        self.sigmoid = torch.nn.Sigmoid()
        self.input_layer = torch.nn.Linear(self.input_dim, self.input_dim)
        self.hidden_layer_0 = torch.nn.Linear(self.input_dim, 2000)
        self.hidden_layer_1 = torch.nn.Linear(2000, 500)
        self.hidden_layer_2 = torch.nn.Linear(500, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.normalize(x, dim=-1)
        x = self.dropout(x)
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer_0(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer_1(x))
        x = self.dropout(x)
        x = self.hidden_layer_2(x)
        return self.sigmoid(x)


@dataclass(frozen=True)
class InstructionDataGuardArtifacts:
    model: InstructionDataGuardNet
    model_path: str
    device: torch.device
    dtype: torch.dtype
    input_dim: int
    dropout: float
    parameter_count: int

    def score_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[-1] != self.input_dim:
            raise ValueError(f"expected embeddings with last dimension {self.input_dim}, got {tuple(embeddings.shape)}")
        with torch.inference_mode():
            return self.model(embeddings.to(device=self.device, dtype=self.dtype))


def resolve_instruction_data_guard_dtype(dtype: str | torch.dtype | None = None) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype or "float32").lower()
    return {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }.get(name, torch.float32)


def load_instruction_data_guard(
    model_path: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype | None = None,
) -> InstructionDataGuardArtifacts:
    root = Path(model_path)
    config_path = root / "config.json"
    weights_path = root / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json under {root}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"missing model.safetensors under {root}")
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    input_dim = int(config.get("input_dim", 4096))
    dropout = float(config.get("dropout", 0.7))
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = resolve_instruction_data_guard_dtype(dtype)

    model = InstructionDataGuardNet(input_dim=input_dim, dropout=dropout)
    try:
        from safetensors.torch import load_file
    except Exception as exc:
        raise RuntimeError("safetensors.torch is required to load instruction-data-guard weights") from exc
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device=torch_device, dtype=torch_dtype)
    model.eval()
    return InstructionDataGuardArtifacts(
        model=model,
        model_path=str(root),
        device=torch_device,
        dtype=torch_dtype,
        input_dim=input_dim,
        dropout=dropout,
        parameter_count=int(sum(param.numel() for param in model.parameters())),
    )
