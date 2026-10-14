"""Attempt facts and typed claims; model prose never decides evidence polarity."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from statistics import median
from typing import Any

from .models import ExperienceRecord
from .conditions import check, observed, evidence_matches


CLAIM_TYPES = {
    "performance_recommendation", "performance_warning",
    "correctness_constraint", "unclassified",
}
SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_claims (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    predicate TEXT NOT NULL,
    conditions_json TEXT NOT NULL,
    scope TEXT NOT NULL CHECK(scope IN ('family', 'general')),
    semantic_type TEXT NOT NULL,
    platform TEXT NOT NULL,
    kernel_backend TEXT NOT NULL,
    applicable_directions_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claim_domain
ON experiment_claims(platform, kernel_backend, semantic_type);
CREATE TABLE IF NOT EXISTS experiment_claim_evidence (
    claim_id TEXT NOT NULL REFERENCES experiment_claims(id),
    attempt_id TEXT NOT NULL REFERENCES experiment_attempts(id),
    measurement_id TEXT REFERENCES experiment_measurements(id),
    case_key TEXT NOT NULL DEFAULT '',
    relation TEXT NOT NULL CHECK(relation IN ('supports', 'contradicts', 'inconclusive')),
    reason TEXT NOT NULL,
    PRIMARY KEY(claim_id, attempt_id, case_key)
);
"""


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def _measured(summary: Any, samples: tuple[float, ...]) -> float | None:
    if samples and all(_positive(value) is not None for value in samples):
        return median(samples)
    return _positive(summary)


def _gain(before: float | None, after: float | None) -> float | None:
    return (before - after) / before * 100 if before and after else None


def observe_attempt(record: ExperienceRecord) -> dict[str, Any]:
    """Keep parent comparison, incumbent comparison, and selection separate."""
    parent = _measured(record.baseline_time_ms, record.baseline_samples_ms)
    candidate = _measured(record.kernel_time_ms, record.kernel_samples_ms)
    incumbent = _positive(record.incumbent_time_ms)
    if not record.verified:
        comparison = "unverified"
    elif parent is None or candidate is None:
        comparison = "unmeasured"
    else:
        comparison = "faster" if candidate < parent else (
            "slower" if candidate > parent else "equal"
        )
    result = {
        "version": "attempt_observation_v1",
        "comparison_rule": "median_only_v1",
        "correctness": "passed" if record.verified else (
            "failed" if record.outcome == "failed" else "unknown"
        ),
        "parent_comparison": comparison,
        "parent_ms": parent,
        "candidate_ms": candidate if record.verified else None,
        "parent_gain_pct": _gain(parent, candidate) if record.verified else None,
        "incumbent_ms": incumbent,
        "incumbent_gain_pct": _gain(incumbent, candidate) if record.verified else None,
        "selected_by_worker": record.selected_by_worker,
        "reported_outcome": record.outcome,
        "reported_improvement_pct": record.improvement_pct
            if isinstance(record.improvement_pct, (int, float))
            and math.isfinite(record.improvement_pct) else None,
        "has_paired_samples": len(record.baseline_samples_ms) >= 3
            and len(record.kernel_samples_ms) >= 3,
    }
    if record.comparison:
        from triton_kernel_agent.measurement_policy import compare, valid_schedule
        robust = compare(record.baseline_samples_ms, record.kernel_samples_ms,
                         interleaved=record.verified and valid_schedule(record.comparison.get("measurement_schedule")))
        result["comparison_rule"] = robust["rule"]
        result["noise_aware_comparison"] = {**robust,
            "policy_adoption": record.comparison.get("adopt"),
            "policy_reason": record.comparison.get("reason"),
            "measurement_schedule": record.comparison.get("measurement_schedule", [])}
    return result


def normalize_claims(value: Any) -> list[dict[str, Any]]:
    """Tolerate malformed optional model fields without granting them evidence."""
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value[:4]:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        kind = item.get("claim_type")
        conditions = item.get("conditions", {})
        if (not isinstance(text, str) or not text.strip()
                or not isinstance(kind, str) or kind not in CLAIM_TYPES
                or not isinstance(conditions, dict)):
            continue
        predicate = item.get("predicate", "")
        if not isinstance(predicate, str):
            continue
        defaults = {
            "performance_recommendation": "candidate_improves_parent",
            "performance_warning": "candidate_does_not_improve_parent",
            "unclassified": "unknown",
        }
        predicate = predicate or defaults.get(kind, "")
        if predicate not in {
            defaults.get(kind),
            *(('candidate_passes_correctness', 'candidate_fails_correctness')
              if kind == "correctness_constraint" else ()),
        } or not predicate:
            continue
        normalized.append({
            "text": text.strip(), "claim_type": kind,
            "predicate": predicate, "conditions": conditions,
        })
    return normalized


def claim_relation(claim: dict[str, Any], observation: dict[str, Any]) -> tuple[str, str]:
    predicate = claim["predicate"]
    comparison = observation.get("parent_comparison")
    robust = observation.get("noise_aware_comparison")
    if isinstance(robust, dict):
        comparison = robust.get("status", "inconclusive")
    if predicate == "unknown":
        return "inconclusive", "unclassified_reflection"
    if predicate in {"candidate_passes_correctness", "candidate_fails_correctness"}:
        actual = observation.get("correctness")
        if actual not in {"passed", "failed"}:
            return "inconclusive", "correctness_not_observed"
        expected = "passed" if predicate == "candidate_passes_correctness" else "failed"
        return ("supports" if actual == expected else "contradicts"), "correctness_observation"
    if comparison not in {"faster", "slower", "equal"}:
        return "inconclusive", "noise_or_insufficient_evidence" if robust else "verified_parent_comparison_missing"
    improves = comparison == "faster"
    expected = predicate == "candidate_improves_parent"
    return ("supports" if improves == expected else "contradicts"), (
        "noise_aware_parent_comparison" if robust else "parent_median_comparison")


def record_claims(
    connection: sqlite3.Connection, run_id: str,
    claims: list[dict[str, Any]], *, semantic_type: str,
    platform: str, backend: str, created_at: str,
) -> None:
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    run = connection.execute(
        "SELECT r.hardware, r.environment_json, w.contract_json FROM experiment_runs r "
        "JOIN experiment_workloads w ON w.id=r.workload_id WHERE r.id=?", (run_id,),
    ).fetchone()
    for item in claims:
        normalized = normalize_claims([item])
        if not normalized:
            raise ValueError("claim must declare valid text, claim_type, predicate and conditions")
        claim = normalized[0]
        scope = item.get("scope", "family")
        if scope not in {"family", "general"}:
            raise ValueError("claim scope must be family or general")
        directions = item.get("applicable_directions", [])
        if (not isinstance(directions, list)
                or any(not isinstance(value, str) or
                       value not in {"forward", "backward", "joint"} for value in directions)
                or len(directions) != len(set(directions))
                or (len(directions) > 1 and (
                    scope != "general" or set(directions) != {"forward", "backward"}))):
            raise ValueError("invalid claim applicable_directions")
        family = semantic_type if scope == "family" else "*"
        values = (
            claim["text"], claim["claim_type"], claim["predicate"],
            json.dumps(claim["conditions"], sort_keys=True, ensure_ascii=False),
            scope, family, platform, backend, json.dumps(sorted(directions)),
        )
        claim_id = hashlib.sha256(json.dumps(values).encode()).hexdigest()
        connection.execute(
            "INSERT OR IGNORE INTO experiment_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (claim_id, *values, created_at),
        )
        evidence_list = item.get("evidence", [])
        if not isinstance(evidence_list, list):
            raise ValueError("claim evidence must be a list")
        for evidence in evidence_list:
            if not isinstance(evidence, dict):
                raise ValueError("claim evidence must be an object")
            attempt_id = f"{run_id}:{evidence.get('attempt_id')}"
            attempt = connection.execute(
                "SELECT decision_json, verified, outcome FROM experiment_attempts "
                "WHERE id = ? AND run_id = ?",
                (attempt_id, run_id),
            ).fetchone()
            if attempt is None:
                raise ValueError("claim evidence must cite an attempt in this run")
            measurement_id = None
            facts = observed(json.loads(run[2]), hardware=run[0], environment=json.loads(run[1]))
            observation = json.loads(attempt[0])
            observation["correctness"] = "passed" if attempt[1] else (
                "failed" if attempt[2] == "failed" else "unknown"
            )
            observation["parent_comparison"] = "unmeasured"
            if evidence.get("case_key") is not None:
                measurement = connection.execute(
                    "SELECT id, verified, baseline_ms, kernel_ms, benchmark_json, baseline_samples_json, samples_json, inputs_json "
                    "FROM experiment_measurements WHERE attempt_id = ? AND case_key = ?",
                    (attempt_id, evidence["case_key"]),
                ).fetchone()
                case = connection.execute(
                    "SELECT status FROM experiment_case_results WHERE attempt_id=? AND case_key=?",
                    (attempt_id, evidence["case_key"]),
                ).fetchone()
                if measurement is None and case is None:
                    raise ValueError("claim evidence must cite an existing measurement")
                if case is not None:
                    observation["correctness"] = case[0] if case[0] != "missing" else "unknown"
                elif measurement:
                    observation["correctness"] = "passed" if measurement[1] else "failed"
                measurement_id = measurement[0] if measurement else None
                if measurement:
                    facts = observed(json.loads(run[2]), hardware=run[0],
                                     environment=json.loads(run[1]), inputs=json.loads(measurement[7]))
                    benchmark = json.loads(measurement[4])
                    robust = benchmark.get("comparison")
                    # Per-case propositions cannot borrow the aggregate decision.
                    observation.pop("noise_aware_comparison", None)
                    if isinstance(robust, dict) and robust:
                        from triton_kernel_agent.measurement_policy import compare, valid_schedule
                        observation["noise_aware_comparison"] = compare(
                            json.loads(measurement[5]), json.loads(measurement[6]),
                            interleaved=bool(attempt[1] and measurement[1]) and
                                valid_schedule(benchmark.get("measurement_schedule") or robust.get("measurement_schedule")))
                if measurement and attempt[1] and measurement[1] and measurement[2] and measurement[3]:
                    observation["parent_comparison"] = (
                        "faster" if measurement[3] < measurement[2] else
                        "slower" if measurement[3] > measurement[2] else "equal"
                    )
            relation, reason = claim_relation(claim, observation)
            applicability = check(claim["conditions"], facts)
            if applicability != "compatible":
                relation, reason = "inconclusive", f"conditions_{applicability}"
            # The model's proposed relation is deliberately ignored.
            connection.execute(
                "INSERT OR IGNORE INTO experiment_claim_evidence VALUES (?, ?, ?, ?, ?, ?)",
                (claim_id, attempt_id, measurement_id,
                 evidence.get("case_key") or "", relation, reason),
            )


def claim_report(
    connection: sqlite3.Connection, limit: int, *, semantic_type: str = "",
    platform: str = "", kernel_backend: str = "", validated_only: bool = False,
    context: dict | None = None,
) -> list[dict[str, Any]]:
    filters, parameters = [], []
    for column, value in (("semantic_type", semantic_type), ("platform", platform),
                          ("kernel_backend", kernel_backend)):
        if value:
            filters.append("(l.scope = 'general' OR l.semantic_type = ?)" if
                           column == "semantic_type" else f"l.{column} = ?")
            parameters.append(value)
    filters.append("""NOT EXISTS (SELECT 1 FROM experience_quarantine q
        WHERE q.kind = 'case' AND
        (q.id = a.id OR a.id = (r.id || ':' || q.id)))""")
    filters.append("""NOT EXISTS (SELECT 1 FROM experience_quarantine q
        WHERE q.kind = 'lesson' AND q.id = l.id)""")
    where = "WHERE " + " AND ".join(filters)
    rows = connection.execute(
        f"""SELECT l.*, e.relation, e.reason, a.id AS attempt_id,
            a.verified, a.error_message, r.id AS run_id, r.session_id, r.provenance,
            w.id AS workload_id, w.semantic_type AS observed_family,
            json_extract(w.contract_json, '$.direction') AS direction,
            m.samples_json, m.baseline_samples_json, m.inputs_json,
            r.hardware, r.environment_json, w.contract_json
        FROM experiment_claims l
        LEFT JOIN experiment_claim_evidence e ON e.claim_id = l.id
        LEFT JOIN experiment_attempts a ON a.id = e.attempt_id
        LEFT JOIN experiment_runs r ON r.id = a.run_id
        LEFT JOIN experiment_workloads w ON w.id = r.workload_id
        LEFT JOIN experiment_measurements m ON m.id = e.measurement_id
        {where}""", parameters,
    ).fetchall()
    from .consolidation import roots
    aliases = roots(connection)
    quarantined = {row[0] for row in connection.execute(
        "SELECT id FROM experience_quarantine WHERE kind='lesson'")}
    canonical = {r['id']: r for r in connection.execute('SELECT * FROM experiment_claims')}
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        root = aliases.get(row['id'], row['id'])
        if root in quarantined:
            continue
        report = grouped.setdefault(root, {
            key: canonical[root][key] for key in (
                "id", "text", "scope", "semantic_type", "platform", "kernel_backend",
                "applicable_directions_json", "claim_type", "predicate",
                "conditions_json", "created_at",
            )
        })
        report.setdefault('member_claim_ids', [])
        report["validation_context"] = context
        if row['id'] not in report['member_claim_ids']:
            report['member_claim_ids'].append(row['id'])
        for field in ("supporting_runs", "eligible_runs", "eligible_workloads",
                      "eligible_families", "contrary_runs", "inconclusive_runs",
                      "eligible_directions", "observed_runs"):
            report.setdefault(field, set())
        if row["attempt_id"] is None:
            continue
        session = row["session_id"] or row["run_id"]
        report["observed_runs"].add(session)
        facts = observed(json.loads(row["contract_json"]), hardware=row["hardware"],
                         environment=json.loads(row["environment_json"]),
                         inputs=json.loads(row["inputs_json"]) if row["inputs_json"] else None)
        conditions = json.loads(row["conditions_json"])
        applicability = check(conditions, facts)
        # Re-evaluate old evidence on reading; do not rewrite historical rows.
        relation = row["relation"] if applicability == "compatible" else "inconclusive"
        if context is not None and (check(conditions, context) != "compatible"
                                    or not evidence_matches(facts, context)):
            continue
        field = {"supports": "supporting_runs", "contradicts": "contrary_runs",
                 "inconclusive": "inconclusive_runs"}[relation]
        report[field].add(session)
        # Finite test passes are bounded correctness observations, not proof
        # of a transferable correctness rule; keep them auditable candidates.
        if (relation == "supports" and row["verified"]
                and all(facts.get(key) for key in ("hardware", "dtype", "layout"))
                and row["claim_type"].startswith("performance_")
                and row["reason"] == "noise_aware_parent_comparison"
                and row["provenance"] not in {"legacy", "synthetic-demo"}
                and len(json.loads(row["samples_json"] or "[]")) >= 3
                and len(json.loads(row["baseline_samples_json"] or "[]")) >= 3):
            report["eligible_runs"].add(session)
            report["eligible_workloads"].add(row["workload_id"])
            report["eligible_families"].add(row["observed_family"])
            if row["direction"] in {"forward", "backward", "joint"}:
                report["eligible_directions"].add(row["direction"])
    reports = []
    for item in grouped.values():
        directions = item.pop("eligible_directions")
        item["eligible_directions"] = tuple(
            value for value in ("forward", "backward", "joint") if value in directions
        )
        for key, value in list(item.items()):
            if isinstance(value, set):
                item[key] = len(value)
        item["status"] = "contested" if item["contrary_runs"] else (
            "validated" if item["eligible_runs"] >= 2 and item["eligible_workloads"] >= 2
            and (item["scope"] == "family" or item["eligible_families"] >= 2) else "candidate"
        )
        if not validated_only or item["status"] == "validated":
            reports.append(item)
    reports.sort(key=lambda row: (row["supporting_runs"], -row["contrary_runs"],
                                 row["created_at"]), reverse=True)
    return reports[:limit]
