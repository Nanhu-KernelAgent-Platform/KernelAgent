"""Operator-independent interleaved measurements and conservative adoption.

The MAD envelope is a robustness rule, not a statistical confidence interval.
Old/non-interleaved samples cannot establish a noise-aware win.
"""
from __future__ import annotations

from statistics import median
from .case_suite import valid_samples

RULE = "interleaved_mad_v1"
BLOCKS = 5
WINDOWS = 3
MIN_GAIN_PCT = 2.0
MIN_GAIN_MS = 0.0001


def valid_schedule(schedule):
    expected = [{"block": block, "role": role} for block in range(BLOCKS)
                for role in (("parent", "candidate") if block % 2 == 0 else ("candidate", "parent"))]
    return schedule == expected


def compare(parent, candidate, *, interleaved=False):
    result = {"rule": RULE, "status": "inconclusive", "reason": "insufficient_interleaved_evidence",
              "adopt": False, "min_gain_pct": MIN_GAIN_PCT, "min_gain_ms": MIN_GAIN_MS,
              "blocks": BLOCKS, "windows_per_block": WINDOWS}
    if not interleaved or not all(valid_samples(values, BLOCKS * WINDOWS) for values in (parent, candidate)):
        return result
    before = [median(parent[i:i + WINDOWS]) for i in range(0, len(parent), WINDOWS)]
    after = [median(candidate[i:i + WINDOWS]) for i in range(0, len(candidate), WINDOWS)]
    savings = [a - b for a, b in zip(before, after)]
    gains = [100 * (a - b) / a for a, b in zip(before, after)]
    def bounds(values):
        center = median(values)
        radius = 2 * 1.4826 * median(abs(x - center) for x in values)
        return [center - radius, center + radius]
    gain_bounds, saving_bounds = bounds(gains), bounds(savings)
    # Stable block medians alone must not hide broad within-block jitter.
    raw_gain_bounds = bounds([100 * (a - b) / a for a, b in zip(parent, candidate)])
    raw_saving_bounds = bounds([a - b for a, b in zip(parent, candidate)])
    gain_bounds = [min(gain_bounds[0], raw_gain_bounds[0]), max(gain_bounds[1], raw_gain_bounds[1])]
    saving_bounds = [min(saving_bounds[0], raw_saving_bounds[0]), max(saving_bounds[1], raw_saving_bounds[1])]
    wins = sum(x > 0 for x in savings)
    if gain_bounds[0] >= MIN_GAIN_PCT and saving_bounds[0] >= MIN_GAIN_MS and wins >= 4:
        status, reason = "faster", "robust_practical_gain"
    elif gain_bounds[1] <= -MIN_GAIN_PCT and saving_bounds[1] <= -MIN_GAIN_MS and wins <= 1:
        status, reason = "slower", "robust_practical_regression"
    else:
        status, reason = "inconclusive", "noise_or_gain_below_threshold"
    return {**result, "status": status, "reason": reason, "adopt": status == "faster",
            "gain_pct": median(gains), "gain_envelope_pct": gain_bounds,
            "saving_envelope_ms": saving_bounds, "winning_blocks": wins,
            "parent_block_medians_ms": before, "candidate_block_medians_ms": after}


def interleaved_pair(measure, parent, candidate):
    """Caller holds the shared device lock across the entire AB/BA schedule."""
    history = {"parent": [], "candidate": []}
    schedule = []
    for block in range(BLOCKS):
        order = ("parent", "candidate") if block % 2 == 0 else ("candidate", "parent")
        for role in order:
            result = measure(parent if role == "parent" else candidate)
            history[role].append(result)
            schedule.append({"block": block, "role": role})
    def merge(records):
        last = records[-1]
        samples = [x for record in records for x in record.get("samples_ms", [])]
        keys = [row["case_key"] for row in records[0].get("case_measurements", [])]
        rows = []
        valid = all(valid_samples(record.get("samples_ms", []), WINDOWS) for record in records)
        for key in keys:
            points = [next((r for r in record.get("case_measurements", []) if r["case_key"] == key), None)
                      for record in records]
            if any(r is None for r in points) or any(r["inputs"] != points[0]["inputs"] for r in points if r):
                valid = False
                continue
            values = [x for r in points for x in r["samples_ms"]]
            if not all(valid_samples(r["samples_ms"], WINDOWS) for r in points):
                valid = False
                continue
            rows.append({**points[-1], "samples_ms": values, "time_ms": median(values)})
        if any([r["case_key"] for r in record.get("case_measurements", [])] != keys for record in records):
            valid = False
        return {**last, "samples_ms": samples if valid else [],
                "time_ms": median(samples) if valid and samples else float("inf"),
                "case_measurements": rows if valid else [], "interleaved": valid,
                "measurement_schedule": schedule, "comparison_rule": RULE}
    before, after = merge(history["parent"]), merge(history["candidate"])
    decision = compare(before["samples_ms"], after["samples_ms"], interleaved=before["interleaved"] and after["interleaved"])
    prior_cases = {r["case_key"]: r for r in before["case_measurements"]}
    per_case = {}
    for row in after["case_measurements"]:
        prior = prior_cases.get(row["case_key"])
        valid = prior is not None and prior["inputs"] == row["inputs"]
        per_case[row["case_key"]] = compare(prior["samples_ms"] if prior else [], row["samples_ms"],
                                            interleaved=valid and before["interleaved"] and after["interleaved"])
    if set(per_case) != set(prior_cases) or any(row["status"] == "slower" for row in per_case.values()):
        decision = {**decision, "adopt": False, "reason": "case_coverage_or_material_regression"}
    after["comparison"] = {**decision, "cases": per_case, "measurement_schedule": schedule}
    return before, after
