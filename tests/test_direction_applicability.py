"""Direction applicability needs explicit declarations and real paired evidence."""

from __future__ import annotations

import copy
import json

import pytest

from triton_kernel_agent.experience import OperatorSignature, SQLiteExperienceStore, build_operator_signature
from triton_kernel_agent.experience.experiment_db import (
    lesson_report,
    record_bundle,
)
from triton_kernel_agent.experience.store import format_experience_context


_LESSON = "Avoid unnecessary intermediate traffic when launch cost dominates"


def _bundle(direction: str, family: str, session_id: str) -> dict:
    from tests.test_experiment_db import _bundle as experiment_bundle
    bundle = experiment_bundle()
    bundle["run"].update(provenance="manual", session_id=session_id)
    bundle["workload"]["semantic_type"] = family
    bundle["workload"]["contract"]["operation"] = f"{family}(x)"
    bundle["workload"]["contract"]["direction"] = direction
    bundle["claims"][0].update(
        text=_LESSON,
        scope="general",
        applicable_directions=["forward", "backward"],
    )
    return bundle


def test_bidirectional_principle_requires_evidence_in_both_directions(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = OperatorSignature("sigmoid", dtypes=("bfloat16",))
    with store._connect() as connection:
        record_bundle(connection, _bundle("forward", "rmsnorm", "session-one"))
        record_bundle(connection, _bundle("forward", "sigmoid", "session-two"))
        report = lesson_report(connection)[0]
        assert report["status"] == "validated"
        assert json.loads(report["applicable_directions_json"]) == [
            "backward", "forward"
        ]
        assert report["eligible_directions"] == ("forward",)
    assert store.retrieve(signature, "musa", "musa", direction="backward") == []

    with store._connect() as connection:
        record_bundle(connection, _bundle("backward", "gelu", "session-three"))
        report = lesson_report(connection)[0]
        assert report["eligible_directions"] == (
            "forward", "backward"
        )
    hits = store.retrieve(
        signature, "musa", "musa", stage="optimization", direction="backward",
        hardware="example-gpu", layout="contiguous",
    )
    assert len(hits) == 1
    assert hits[0].kind == "principle"
    assert hits[0].kernel_code == ""
    assert hits[0].applicable_directions == ("backward", "forward")
    assert hits[0].observed_directions == ("forward", "backward")
    assert hits[0].retrieval_reason == "validated_bidirectional_principle"
    assert "validated on both execution directions" in format_experience_context(hits)

    event_id = store.log_retrieval(
        session_id="audit", stage="optimization", signature=signature,
        records=hits, direction="backward",
    )
    with store._connect() as connection:
        row = connection.execute(
            "SELECT selection_json FROM experience_retrieval_events WHERE id = ?",
            (event_id,),
        ).fetchone()
    selection = json.loads(row["selection_json"])
    assert selection[0]["reason"] == "validated_bidirectional_principle"
    assert selection[0]["observed_directions"] == ["forward", "backward"]
    assert selection[0]["applicable_directions"] == ["backward", "forward"]


@pytest.mark.parametrize("declared", [
    ["forward", "forward"],
    ["forward", "unknown"],
    [{"direction": "forward"}],
    ["forward", "joint"],
])
def test_invalid_direction_declarations_roll_back(tmp_path, declared):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    bundle = _bundle("forward", "rmsnorm", "bad")
    bundle["claims"][0]["applicable_directions"] = declared
    with pytest.raises(ValueError, match="applicable_directions"):
        with store._connect() as connection:
            record_bundle(connection, bundle)
    with store._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0


def test_multiple_directions_require_general_scope(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    bundle = _bundle("forward", "rmsnorm", "bad-scope")
    bundle["claims"][0]["scope"] = "family"
    with pytest.raises(ValueError, match="applicable_directions"):
        with store._connect() as connection:
            record_bundle(connection, bundle)

def test_bidirectional_principles_use_one_prompt_slot(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    for direction, family, session in (
        ("forward", "rmsnorm", "one"),
        ("forward", "sigmoid", "two"),
        ("backward", "gelu", "three"),
    ):
        bundle = _bundle(direction, family, session)
        second = copy.deepcopy(bundle["claims"][0])
        second["text"] = "Minimize redundant launches for very short kernels"
        bundle["claims"].append(second)
        with store._connect() as connection:
            record_bundle(connection, bundle)
    signature = OperatorSignature("sigmoid", dtypes=("bfloat16",))
    hits = store.retrieve(
        signature, "musa", "musa", stage="optimization",
        direction="backward", limit=4, hardware="example-gpu", layout="contiguous",
    )
    assert len(hits) == 1
    assert hits[0].retrieval_reason == "validated_bidirectional_principle"
