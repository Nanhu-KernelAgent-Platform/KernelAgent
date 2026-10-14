"""Immutable kernel experiments and evidence-backed typed claims."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from statistics import median
from typing import Any
from uuid import uuid4

from .evidence import SCHEMA as CLAIM_SCHEMA, claim_report, record_claims


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_workloads (
    id TEXT PRIMARY KEY,
    semantic_type TEXT NOT NULL,
    platform TEXT NOT NULL,
    kernel_backend TEXT NOT NULL,
    contract_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_workload_domain
ON experiment_workloads(platform, kernel_backend, semantic_type);
CREATE TABLE IF NOT EXISTS experiment_runs (
    id TEXT PRIMARY KEY,
    workload_id TEXT NOT NULL REFERENCES experiment_workloads(id),
    model TEXT NOT NULL,
    hardware TEXT NOT NULL,
    environment_json TEXT NOT NULL,
    provenance TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_workload ON experiment_runs(workload_id);
CREATE TABLE IF NOT EXISTS experiment_attempts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_runs(id),
    parent_id TEXT REFERENCES experiment_attempts(id),
    parent_ref TEXT NOT NULL DEFAULT '',
    source_hash TEXT NOT NULL,
    kernel_code TEXT NOT NULL,
    action TEXT NOT NULL,
    config_changes_json TEXT NOT NULL,
    outcome TEXT NOT NULL,
    verified INTEGER NOT NULL CHECK(verified IN (0, 1)),
    error_message TEXT NOT NULL,
    created_at TEXT NOT NULL,
    decision_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_attempt_run ON experiment_attempts(run_id);
CREATE TABLE IF NOT EXISTS experiment_measurements (
    id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL REFERENCES experiment_attempts(id),
    case_key TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    baseline_ms REAL CHECK(baseline_ms > 0),
    baseline_samples_json TEXT NOT NULL DEFAULT '[]',
    kernel_ms REAL CHECK(kernel_ms > 0),
    samples_json TEXT NOT NULL,
    verified INTEGER NOT NULL CHECK(verified IN (0, 1)),
    benchmark_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(attempt_id, case_key)
);
CREATE INDEX IF NOT EXISTS idx_measurement_attempt
ON experiment_measurements(attempt_id);
CREATE TABLE IF NOT EXISTS experiment_case_results (
    attempt_id TEXT NOT NULL REFERENCES experiment_attempts(id),
    case_key TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('passed', 'failed', 'missing')),
    contract_json TEXT NOT NULL,
    error_message TEXT NOT NULL,
    PRIMARY KEY(attempt_id, case_key)
);
"""




def initialize(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA)
    connection.executescript(CLAIM_SCHEMA)
    from .consolidation import SCHEMA as ALIAS_SCHEMA
    connection.executescript(ALIAS_SCHEMA)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value.strip()


def _workload_id(workload: dict[str, Any]) -> str:
    contract = workload["contract"]
    if (contract.get("graph_version") and contract.get("graph_digest")
            and contract.get("interface")):
        # New workloads use their traced computation and runtime interface,
        # not a prose-derived family label. Legacy identities stay untouched.
        identity = {
            "platform": workload["platform"],
            "kernel_backend": workload["kernel_backend"],
            "direction": contract.get("direction"),
            "graph_version": contract["graph_version"],
            "graph_digest": contract["graph_digest"],
            "interface": contract["interface"],
        }
    else:
        identity = {
            key: workload[key]
            for key in ("semantic_type", "platform", "kernel_backend", "contract")
        }
    return hashlib.sha256(_json(identity).encode()).hexdigest()[:24]


def record_bundle(connection: sqlite3.Connection, bundle: dict[str, Any]) -> str:
    """Atomically record a run, its attempts, test cases, and cited lessons.

    The caller owns the connection transaction. All attempt IDs in a bundle are
    local to that run; a parent must already have appeared in the bundle.
    """
    if "lessons" in bundle:
        raise ValueError("Use typed claims; untyped lessons are not supported")
    workload = bundle["workload"]
    run = bundle["run"]
    if not isinstance(workload, dict) or not isinstance(run, dict):
        raise ValueError("workload and run must be objects")
    semantic_type = _required_text(workload, "semantic_type")
    platform = _required_text(workload, "platform")
    backend = _required_text(workload, "kernel_backend")
    contract = workload.get("contract", {})
    if not isinstance(contract, dict):
        raise ValueError("contract must be an object")
    workload_id = workload.get("id") or _workload_id(workload)
    run_id = run.get("id") or uuid4().hex
    if not isinstance(workload_id, str) or not workload_id:
        raise ValueError("workload.id must be a nonempty string")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run.id must be a nonempty string")
    created_at = _now()
    existing = connection.execute(
        "SELECT semantic_type, platform, kernel_backend, contract_json "
        "FROM experiment_workloads WHERE id = ?", (workload_id,)
    ).fetchone()
    expected = (semantic_type, platform, backend, _json(contract))
    if existing and tuple(existing) != expected:
        prior_contract = json.loads(existing[3])
        prior = {
            "semantic_type": existing[0], "platform": existing[1],
            "kernel_backend": existing[2], "contract": prior_contract,
        }
        if not (contract.get("graph_digest") and prior_contract.get("graph_digest") and
                _workload_id(prior) == _workload_id(workload)):
            raise ValueError(f"workload ID {workload_id!r} has a different contract")
    connection.execute(
        """INSERT OR IGNORE INTO experiment_workloads VALUES (?, ?, ?, ?, ?, ?)""",
        (workload_id, semantic_type, platform, backend, _json(contract), created_at),
    )
    connection.execute(
        """INSERT INTO experiment_runs (
            id, workload_id, model, hardware, environment_json,
            provenance, session_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (run_id, workload_id, str(run.get("model") or ""),
         str(run.get("hardware") or ""), _json(run.get("environment") or {}),
         str(run.get("provenance") or "manual"),
         str(run.get("session_id") or ""), created_at),
    )

    seen: set[str] = set()
    measurements: dict[tuple[str, str], str] = {}
    for attempt in bundle.get("attempts", []):
        if not isinstance(attempt, dict):
            raise ValueError("each attempt must be an object")
        local_id = _required_text(attempt, "id")
        parent_local = attempt.get("parent_id")
        if local_id in seen or (parent_local and parent_local not in seen):
            raise ValueError("duplicate attempt or parent not in this run")
        seen.add(local_id)
        attempt_id = f"{run_id}:{local_id}"
        parent_id = f"{run_id}:{parent_local}" if parent_local else None
        code = str(attempt.get("kernel_code") or "")
        if not isinstance(attempt.get("decision", {}), dict):
            raise ValueError("attempt decision must be an object")
        connection.execute(
            """INSERT INTO experiment_attempts (
                id, run_id, parent_id, parent_ref, source_hash, kernel_code,
                action, config_changes_json, outcome, verified, error_message,
                created_at, decision_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (attempt_id, run_id, parent_id, str(attempt.get("parent_ref") or ""),
             hashlib.sha256(code.encode()).hexdigest(),
             code, str(attempt.get("action") or ""),
             _json(attempt.get("config_changes") or {}),
             str(attempt.get("outcome") or "unknown"), int(bool(attempt.get("verified"))),
             str(attempt.get("error_message") or ""), created_at,
             _json(attempt.get("decision") or {})),
        )
        case_results = attempt.get("case_results", [])
        if not isinstance(case_results, list):
            raise ValueError("case_results must be a list")
        for case in case_results:
            case_key = _required_text(case, "case_key")
            status = case.get("status")
            if status not in {"passed", "failed", "missing"}:
                raise ValueError("case status must be passed, failed, or missing")
            if attempt.get("verified") and status != "passed":
                raise ValueError("verified attempt cannot contain failed or missing cases")
            connection.execute(
                "INSERT INTO experiment_case_results VALUES (?, ?, ?, ?, ?)",
                (attempt_id, case_key, status, _json(case.get("contract") or {}),
                 str(case.get("error") or "")),
            )
        for measurement in attempt.get("measurements", []):
            if not isinstance(measurement, dict):
                raise ValueError("each measurement must be an object")
            case_key = _required_text(measurement, "case_key")
            if measurement.get("verified") and not attempt.get("verified"):
                raise ValueError("verified measurement requires a verified attempt")
            samples = measurement.get("samples_ms") or []
            baseline_samples = measurement.get("baseline_samples_ms") or []
            if not isinstance(samples, list) or not isinstance(baseline_samples, list):
                raise ValueError("sample series must be lists")
            samples = [_positive(item, "samples_ms item") for item in samples]
            baseline_samples = [
                _positive(item, "baseline_samples_ms item")
                for item in baseline_samples
            ]
            baseline_ms = _positive(measurement.get("baseline_ms"), "baseline_ms")
            if baseline_samples:
                baseline_median = median(baseline_samples)
                if baseline_ms is not None and not math.isclose(
                    baseline_ms, baseline_median, rel_tol=0.05
                ):
                    raise ValueError("baseline_ms must match sample median within 5%")
                baseline_ms = baseline_median
            kernel_ms = _positive(measurement.get("kernel_ms"), "kernel_ms")
            if samples:
                measured_median = median(samples)
                if kernel_ms is not None and not math.isclose(
                    kernel_ms, measured_median, rel_tol=0.05
                ):
                    raise ValueError("kernel_ms must match the sample median within 5%")
                kernel_ms = measured_median
            measurement_id = uuid4().hex
            connection.execute(
                """INSERT INTO experiment_measurements VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (measurement_id, attempt_id, case_key,
                 _json(measurement.get("inputs") or {}),
                 baseline_ms, _json(baseline_samples), kernel_ms, _json(samples),
                 int(bool(measurement.get("verified"))),
                 _json(measurement.get("benchmark") or {}), created_at),
            )
            measurements[(local_id, case_key)] = measurement_id

    record_claims(
        connection, run_id, bundle.get("claims", []),
        semantic_type=semantic_type, platform=platform,
        backend=backend, created_at=created_at,
    )
    return run_id






def lesson_report(
    connection: sqlite3.Connection, limit: int = 20, *, semantic_type: str = "",
    platform: str = "", kernel_backend: str = "", validated_only: bool = False,
    context: dict | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return claim_report(connection, limit, semantic_type=semantic_type,
                        platform=platform, kernel_backend=kernel_backend,
                        validated_only=validated_only, context=context)


def run_report(connection: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    run = connection.execute(
        """SELECT r.*, w.semantic_type, w.platform, w.kernel_backend,
        w.contract_json FROM experiment_runs r JOIN experiment_workloads w
        ON w.id = r.workload_id WHERE r.id = ?""", (run_id,)
    ).fetchone()
    if run is None:
        return None
    result = dict(run)
    result["contract"] = json.loads(result.pop("contract_json"))
    result["environment"] = json.loads(result.pop("environment_json"))
    result["attempts"] = []
    for attempt in connection.execute(
        "SELECT * FROM experiment_attempts WHERE run_id = ? ORDER BY created_at, rowid",
        (run_id,),
    ):
        item = dict(attempt)
        item["decision"] = json.loads(item.pop("decision_json"))
        item["config_changes"] = json.loads(item.pop("config_changes_json"))
        item["measurements"] = []
        item["case_results"] = []
        for case in connection.execute(
            "SELECT * FROM experiment_case_results WHERE attempt_id = ? ORDER BY case_key", (item["id"],)
        ):
            case = dict(case)
            case["contract"] = json.loads(case.pop("contract_json"))
            item["case_results"].append(case)
        for measurement in connection.execute(
            "SELECT * FROM experiment_measurements WHERE attempt_id = ? ORDER BY case_key",
            (item["id"],),
        ):
            point = dict(measurement)
            point["inputs"] = json.loads(point.pop("inputs_json"))
            point["samples_ms"] = json.loads(point.pop("samples_json"))
            point["baseline_samples_ms"] = json.loads(
                point.pop("baseline_samples_json")
            )
            point["benchmark"] = json.loads(point.pop("benchmark_json"))
            item["measurements"].append(point)
        item["claims"] = [dict(row) for row in connection.execute(
            """SELECT l.id, l.text, l.claim_type, l.predicate, l.conditions_json,
                      e.relation, e.reason, e.measurement_id, e.case_key
               FROM experiment_claim_evidence e JOIN experiment_claims l ON l.id=e.claim_id
               WHERE e.attempt_id = ? ORDER BY l.id""", (item["id"],)
        )]
        for claim in item["claims"]:
            claim["conditions"] = json.loads(claim.pop("conditions_json"))
        result["attempts"].append(item)
    return result
