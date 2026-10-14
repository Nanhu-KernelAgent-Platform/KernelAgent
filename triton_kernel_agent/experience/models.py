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
    components: tuple[str, ...] = ()
    contract_digest: str = ""
    graph_version: str = ""
    graph_digest: str = ""
    graph_nodes: tuple[dict[str, Any], ...] = ()
    interface: dict[str, Any] = field(default_factory=dict)
    graph_tensor_specs: dict[str, Any] = field(default_factory=dict)

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
    lesson_claims: tuple[dict[str, Any], ...] = ()
    incumbent_time_ms: float | None = None
    selected_by_worker: bool | None = None
    claim_type: str = "unclassified"
    claim_conditions: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    baseline_time_ms: float | None = None
    kernel_time_ms: float | None = None
    kernel_samples_ms: tuple[float, ...] = ()
    baseline_samples_ms: tuple[float, ...] = ()
    case_results: tuple[dict[str, Any], ...] = ()
    case_measurements: tuple[dict[str, Any], ...] = ()
    comparison: dict[str, Any] = field(default_factory=dict)
    improvement_pct: float | None = None
    profiler_metrics: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    source_hash: str = ""
    parent_ref: str = ""
    config_changes: dict[str, str] = field(default_factory=dict)
    hardware: str = ""
    layout: str = ""
    direction: str = "unknown"
    kind: str = "case"
    scope: str = "family"
    evidence_count: int = 1
    retrieval_match: str = "whole"
    graph_match_evidence: dict[str, Any] = field(default_factory=dict)
    cross_direction: bool = False
    applicable_directions: tuple[str, ...] = ()
    observed_directions: tuple[str, ...] = ()
    retrieval_reason: str = ""
    candidate_match: str = ""
    candidate_text_match: bool = False
    candidate_graph_match: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_reusable_success(self) -> bool:
        return (
            self.verified
            and self.outcome in {"generated", "improved", "success"}
            and (self.improvement_pct is None or self.improvement_pct >= 0)
        )

    @property
    def text_evidence_status(self) -> str:
        """Text classification is not the historical outcome or a code gate."""
        if not self.verified:
            return "correctness_failed" if self.outcome == "failed" else "unverified"
        observation = self.environment.get("attempt_observation", {})
        coverage = observation.get("case_coverage", {})
        if coverage.get("source") == "explicit_case_suite" and not coverage.get("all_passed"):
            return "unverified"
        noise = observation.get("noise_aware_comparison", {})
        if noise.get("rule") == "interleaved_mad_v1":
            return {
                "faster": "robust_gain", "slower": "robust_regression",
            }.get(noise.get("status"), "inconclusive")
        if self.outcome == "generated":
            return "correctness_only"
        return "median_only" if self.kernel_time_ms is not None or self.improvement_pct is not None else "unmeasured"
