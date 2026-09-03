"""Data models for persistent optimization experience."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class OperatorSignature:
    semantic_type: str
    dtypes: tuple[str, ...] = ()
    shapes: tuple[tuple[int, ...], ...] = ()
    tokens: tuple[str, ...] = ()
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperienceRecord:
    signature: OperatorSignature
    platform: str
    kernel_backend: str
    outcome: str
    kernel_code: str
    verified: bool
    id: str = field(default_factory=lambda: uuid4().hex)
    action: str = ""
    bottleneck: str = ""
    lesson: str = ""
    error_message: str = ""
    baseline_time_ms: float | None = None
    kernel_time_ms: float | None = None
    improvement_pct: float | None = None
    profiler_metrics: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    source_hash: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_reusable_success(self) -> bool:
        return self.verified and self.outcome in {"generated", "improved", "success"}
