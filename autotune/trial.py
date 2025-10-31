from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class TrialStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class Trial:
    id: int
    params: Dict[str, Any]
    status: TrialStatus = TrialStatus.PENDING
    result: Optional[float] = None
    error: Optional[str] = None
    step: int = 0
    user_data: Dict[str, Any] = field(default_factory=dict)

    def set_running(self) -> None:
        self.status = TrialStatus.RUNNING

    def set_result(self, value: float) -> None:
        self.result = float(value)
        self.status = TrialStatus.COMPLETED

    def set_failed(self, error: str) -> None:
        self.error = error
        self.status = TrialStatus.FAILED

    def set_pruned(self) -> None:
        self.status = TrialStatus.PRUNED


