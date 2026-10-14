import copy
import json
import sqlite3
from pathlib import Path

import pytest

from scripts.experience_db import main
from triton_kernel_agent.experience import (
    ExperienceRecord,
    SQLiteExperienceStore,
    build_operator_signature,
)
from triton_kernel_agent.experience.experiment_db import (
    lesson_report,
    record_bundle,
    run_report,
)


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/experience_bundle.json"


def test_new_schema_has_one_ledger_and_a_read_only_case_view(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    SQLiteExperienceStore(store.path)  # Reopening does not migrate anything.
    with store._connect() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 9
        objects = dict(connection.execute("SELECT name, type FROM sqlite_master"))
        assert objects["experience_cases"] == "view"
        assert not {"experiences", "experience_lessons", "experiment_lessons",
                    "experiment_lesson_evidence"} & objects.keys()
        with pytest.raises(sqlite3.OperationalError, match="view"):
            connection.execute("DELETE FROM experience_cases")


@pytest.mark.parametrize("version", [0, 8, 10])
def test_nonempty_unsupported_database_is_not_modified(tmp_path, version):
    path = tmp_path / "unsupported.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE original (content TEXT)")
        connection.execute("INSERT INTO original VALUES ('preserved')")
        connection.execute(f"PRAGMA user_version={version}")
    with pytest.raises(ValueError, match="Only experience schema 9"):
        SQLiteExperienceStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == version
        assert connection.execute("SELECT * FROM original").fetchall() == [("preserved",)]
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("original",)]


def test_ledger_projection_preserves_complete_graph_context_and_duplicate_sources(tmp_path):
    from dataclasses import replace
    from triton_kernel_agent.experience import OperatorSignature
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    signature = OperatorSignature("arbitrary", tokens=("opaque",), digest="digest",
        graph_version="graph-v1", graph_digest="graph-digest",
        graph_nodes=({"id": "0", "target": "aten.sin.default"},),
        graph_tensor_specs={"0": {"shape": [8], "stride": [1], "dtype": "float32"}},
        interface={"inputs": [{"shape": [8], "stride": [1], "dtype": "float32"}]})
    first = ExperienceRecord(signature, "musa", "musa", "generated", "same source", True,
                             direction="forward", profiler_metrics={"occupancy": .5})
    second = replace(first, id="independent", direction="backward")
    store.add_online(first, session_id="one")
    store.add_online(second, session_id="two")
    with store._connect() as connection:
        rows = connection.execute("SELECT * FROM experience_cases ORDER BY direction").fetchall()
        assert len(rows) == 2
        for row in rows:
            restored = store._from_row(row)
            assert restored.signature == signature
            assert restored.profiler_metrics == {"occupancy": .5}
        assert {row["id"] for row in rows} == {first.id, second.id}
        assert {row["direction"] for row in rows} == {"forward", "backward"}




def _bundle():
    bundle = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    from triton_kernel_agent.measurement_policy import interleaved_pair
    for attempt in bundle["attempts"]:
        for point in attempt["measurements"]:
            parent, candidate = interleaved_pair(lambda role: {
                "samples_ms": [point["baseline_ms"] if role == "parent"
                               else point["samples_ms"][1]] * 3,
            }, "parent", "candidate")
            point["baseline_samples_ms"] = parent["samples_ms"]
            point["samples_ms"] = candidate["samples_ms"]
            point["benchmark"]["comparison"] = candidate["comparison"]
    return bundle


def test_bundle_records_per_shape_lineage_and_evidence(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    with store._connect() as connection:
        run_id = record_bundle(connection, _bundle())
        report = run_report(connection, run_id)
        lessons = lesson_report(connection)
    assert len(report["attempts"]) == 2
    assert report["attempts"][1]["parent_id"] == f"{run_id}:first"
    assert len(report["attempts"][1]["measurements"]) == 2
    assert lessons[0]["supporting_runs"] == 1
    assert lessons[0]["status"] == "candidate"




def test_repeated_workload_alone_cannot_validate_family_lesson(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    bundle = _bundle()
    bundle["run"]["provenance"] = "manual"
    with store._connect() as connection:
        first = record_bundle(connection, bundle)
        second = record_bundle(connection, copy.deepcopy(bundle))
        lessons = lesson_report(connection)
        assert run_report(connection, first) is not None
        assert run_report(connection, second) is not None
    assert first != second
    assert lessons[0]["eligible_runs"] == 2
    assert lessons[0]["eligible_workloads"] == 1
    assert lessons[0]["status"] == "candidate"


def test_family_lesson_needs_distinct_workloads_and_sessions(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    first = _bundle()
    first["run"].update(provenance="agent", session_id="session-one")
    repeated = copy.deepcopy(first)
    second = copy.deepcopy(first)
    second["run"]["session_id"] = "session-two"
    second["workload"]["contract"]["shape_variant"] = "larger"
    with store._connect() as connection:
        record_bundle(connection, first)
        record_bundle(connection, repeated)
        assert lesson_report(connection)[0]["supporting_runs"] == 1
        record_bundle(connection, second)
        report = lesson_report(connection)[0]
    assert report["eligible_runs"] == 2
    assert report["eligible_workloads"] == 2
    assert report["status"] == "validated"


def test_online_evidence_keeps_repeat_attempts_with_latest_case_projection(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    record = ExperienceRecord(
        signature=build_operator_signature("def forward(x): return torch.sigmoid(x)"),
        platform="musa", kernel_backend="musa", outcome="improved",
        kernel_code="same source", verified=True,
        parent_ref="program-parent-7",
        baseline_time_ms=0.2, kernel_time_ms=0.1,
        lesson="Fuse adjacent steps", improvement_pct=50,
    )
    store.add_online(record, session_id="one")
    store.add_online(copy.deepcopy(record), session_id="two")
    with store._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experience_cases").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM experiment_attempts").fetchone()[0] == 2
        report = lesson_report(connection)[0]
        lineage = connection.execute(
            "SELECT DISTINCT parent_ref FROM experiment_attempts"
        ).fetchall()
    assert [row[0] for row in lineage] == ["program-parent-7"]
    assert report["supporting_runs"] == 0
    assert report["inconclusive_runs"] == 2
    assert report["claim_type"] == "unclassified"
    assert report["eligible_runs"] == 0  # Online aggregate has no sample series.
    assert report["status"] == "candidate"
    assert all(item.kind != "principle" for item in store.retrieve(
        record.signature, "musa", "musa"
    ))


def test_measured_implementations_use_best_run_and_strict_direction(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(
        "def get_inputs(): return [torch.randn(64, 1568, dtype=torch.float32)]\n"
        "def forward(x, weight): return torch.relu(torch.matmul(x, weight.T))"
    )
    for source, direction, latency in (
        ("same source", "forward", 0.12),
        ("same source", "forward", 0.08),
        ("other source", "forward", 0.10),
        ("backward source", "backward", 0.03),
    ):
        store.add_online(ExperienceRecord(
            signature=signature, platform="musa", kernel_backend="musa",
            direction=direction, outcome="improved", verified=True,
            kernel_code=source, baseline_time_ms=0.2,
            kernel_time_ms=latency, improvement_pct=40,
            hardware="MTT S5000",
        ), session_id=f"{source}-{latency}")
    selected = store.search_measured_implementations(
        signature, "musa", "musa", direction="forward",
        hardware="MTT S5000",
    )
    assert [(item.kernel_code, item.kernel_time_ms) for item in selected] == [
        ("same source", 0.08), ("other source", 0.10),
    ]
    assert store.search_measured_implementations(
        signature, "musa", "musa", direction="forward", hardware="other GPU"
    ) == []
    assert [item.kernel_code for item in store.search_measured_implementations(
        signature, "musa", "musa", direction="backward"
    )] == ["backward source"]
    incompatible = build_operator_signature(
        "def get_inputs(): return [torch.randn(128, 1568, dtype=torch.float32)]\n"
        "def forward(x, weight): return torch.relu(torch.matmul(x, weight.T))"
    )
    assert store.search_measured_implementations(
        incompatible, "musa", "musa", direction="forward"
    ) == []


def test_verified_start_without_improvement_remains_measured_candidate(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(
        "def get_inputs(): return [torch.randn(64, 1568)]\n"
        "def forward(x, weight): return torch.relu(torch.matmul(x, weight.T))"
    )
    store.add_online(ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        direction="backward", outcome="success", verified=True,
        kernel_code="verified start", kernel_time_ms=0.07,
        kernel_samples_ms=(0.06, 0.07, 0.08),
        lesson="Correct starting implementation",
    ), session_id="backward-no-gain")
    candidates = store.search_measured_implementations(
        signature, "musa", "musa", direction="backward"
    )
    assert len(candidates) == 1
    assert candidates[0].kernel_time_ms == 0.07
    with store._connect() as connection:
        samples = connection.execute(
            "SELECT samples_json FROM experiment_measurements"
        ).fetchone()[0]
    assert json.loads(samples) == [0.06, 0.07, 0.08]


@pytest.mark.parametrize("operator", ["sigmoid", "rms_norm", "softmax"])
def test_provisional_candidate_is_operator_independent(tmp_path, operator):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    call = (f"torch.{operator}(x, (256,))" if operator == "rms_norm"
            else "torch.softmax(x, dim=1)" if operator == "softmax"
            else "torch.sigmoid(x)")
    common = (
        "def get_inputs(): return [torch.randn(16, 256, dtype=torch.float32)]\n"
        f"def forward(x): return {call}\n"
    )
    original = build_operator_signature("# first description\n" + common)
    reworded = build_operator_signature("# same computation, revised wording\n" + common)
    assert original.contract_digest != reworded.contract_digest
    store.add_online(ExperienceRecord(
        signature=original, platform="musa", kernel_backend="musa",
        direction="forward", outcome="improved", verified=True,
        kernel_code=f"{operator} implementation", baseline_time_ms=0.2,
        kernel_time_ms=0.1,
    ), session_id="first")
    hits = store.search_measured_implementations(
        reworded, "musa", "musa", direction="forward"
    )
    assert len(hits) == 1
    assert hits[0].candidate_match == "provisional_contract"
    assert store.search_measured_implementations(
        reworded, "musa", "musa", direction="backward"
    ) == []
    changed_dtype = build_operator_signature(
        "# different dtype\n" + common.replace("float32", "float16")
    )
    assert store.search_measured_implementations(
        changed_dtype, "musa", "musa", direction="forward"
    ) == []


def test_online_workload_identity_ignores_prompt_wording_and_test_hash(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    for description, test_hash in (
        ("Generate sigmoid: def forward(x): return torch.sigmoid(x)", "test-a"),
        ("Optimize sigmoid: def forward(x): return torch.sigmoid(x)", "test-b"),
    ):
        store.add_online(ExperienceRecord(
            signature=build_operator_signature(description),
            platform="musa", kernel_backend="musa", outcome="generated",
            kernel_code="source", verified=True,
        ), session_id=test_hash, test_suite_hash=test_hash)
    with store._connect() as connection:
        workloads = connection.execute(
            "SELECT COUNT(*) FROM experiment_workloads"
        ).fetchone()[0]
    assert workloads == 1


def test_retrieve_promotes_only_independently_validated_lesson(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    first = _bundle()
    first["run"]["provenance"] = "manual"
    second = copy.deepcopy(first)
    second["workload"]["contract"]["shape_variant"] = "larger"
    signature = build_operator_signature("def forward(x): return rmsnorm(x.to(torch.bfloat16))")
    with store._connect() as connection:
        record_bundle(connection, first)
    assert store.retrieve(signature, "musa", "musa") == []
    with store._connect() as connection:
        record_bundle(connection, second)
    selected = store.retrieve(signature, "musa", "musa", hardware="example-gpu", layout="contiguous")
    assert len(selected) == 1
    assert selected[0].kind == "principle"
    assert selected[0].evidence_count == 2


def test_retrieve_cli_previews_selected_items_without_source(tmp_path, capsys):
    db = tmp_path / "experiences.sqlite3"
    store = SQLiteExperienceStore(db)
    problem = tmp_path / "problem.py"
    problem.write_text("def forward(x): return torch.sigmoid(x)")
    signature = build_operator_signature(problem.read_text())
    store.add_online(ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        outcome="generated", kernel_code="secret source", verified=True,
    ), session_id="session")
    assert main([
        "--db", str(db), "retrieve", str(problem),
        "--platform", "musa", "--kernel-backend", "musa",
    ]) == 0
    output = capsys.readouterr().out
    assert "secret source" not in output
    # Correctness alone has no textual lesson; source reuse has a separate gate.
    assert json.loads(output)["selected"] == []


def test_retrieval_audit_records_selected_ids_and_outcome(tmp_path, capsys):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature("def forward(x): return torch.sigmoid(x)")
    record = ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        outcome="generated", kernel_code="verified", verified=True,
    )
    event_id = store.log_retrieval(
        session_id="session-a", stage="generation", signature=signature,
        records=[record], seeds=[],
    )
    store.set_retrieval_outcome(event_id, {"success": True})
    with store._connect() as connection:
        row = connection.execute(
            "SELECT * FROM experience_retrieval_events WHERE id = ?", (event_id,)
        ).fetchone()
    assert json.loads(row["case_ids_json"]) == [record.id]
    assert json.loads(row["selection_json"])[0]["observed_direction"] == "unknown"
    assert json.loads(row["outcome_json"]) == {"success": True}
    assert main([
        "--db", str(store.path), "retrievals", "--session-id", "session-a"
    ]) == 0
    assert json.loads(capsys.readouterr().out)[0]["case_ids"] == [record.id]


def test_general_lesson_needs_two_families_and_contrary_evidence_blocks_it(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    first = _bundle()
    first["run"]["provenance"] = "manual"
    first["claims"][0]["scope"] = "general"
    second = copy.deepcopy(first)
    second["workload"]["semantic_type"] = "sigmoid"
    second["workload"]["contract"]["operation"] = "sigmoid(x)"
    with store._connect() as connection:
        record_bundle(connection, first)
        record_bundle(connection, second)
        assert lesson_report(connection)[0]["status"] == "validated"
        assert len(lesson_report(
            connection, semantic_type="softmax", platform="musa",
            kernel_backend="musa",
        )) == 1
        contrary = copy.deepcopy(second)
        for point in contrary["attempts"][1]["measurements"]:
            point["samples_ms"] = [point["baseline_ms"] * 2] * 15
        record_bundle(connection, contrary)
        assert lesson_report(connection)[0]["status"] == "contested"


def test_invalid_evidence_rolls_back_whole_bundle(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    bundle = _bundle()
    bundle["claims"][0]["evidence"][0]["case_key"] = "missing"
    with pytest.raises(ValueError, match="must cite an existing measurement"):
        with store._connect() as connection:
            record_bundle(connection, bundle)
    with store._connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM experiment_runs"
        ).fetchone()[0] == 0


def test_existing_writer_syncs_without_inventing_shape_samples(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    record = ExperienceRecord(
        signature=build_operator_signature("def forward(x): return torch.sigmoid(x)"),
        platform="musa", kernel_backend="musa", outcome="improved",
        kernel_code="source", verified=True,
        baseline_time_ms=0.2, kernel_time_ms=0.1,
        lesson="Avoid extra loads", improvement_pct=50,
    )
    assert store.add(record)
    assert not store.add(record)
    with store._connect() as connection:
        report = run_report(connection, connection.execute("SELECT id FROM experiment_runs").fetchone()[0])
        lessons = lesson_report(connection)
    point = report["attempts"][0]["measurements"][0]
    assert point["case_key"] == "aggregate-unspecified"
    assert point["inputs"] == record.signature.interface
    assert point["samples_ms"] == []
    assert lessons[0]["supporting_runs"] == 0
    assert lessons[0]["eligible_runs"] == 0


def test_cli_records_example_and_reports_stats(tmp_path, capsys):
    db = tmp_path / "experience.sqlite3"
    assert main(["--db", str(db), "record", str(EXAMPLE)]) == 0
    run_id = json.loads(capsys.readouterr().out)["run_id"]
    assert main(["--db", str(db), "show", run_id]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == run_id
    assert main(["--db", str(db), "stats"]) == 0
    assert json.loads(capsys.readouterr().out)["measurements"] == 3




def test_quarantine_preserves_rows_but_excludes_retrieval_and_evidence(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experience.sqlite3")
    signature = build_operator_signature("def forward(x): return torch.sigmoid(x)")
    record = ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        outcome="improved", kernel_code="source", verified=True,
        baseline_time_ms=0.2, kernel_time_ms=0.1,
        improvement_pct=50, lesson="Avoid extra loads",
    )
    store.add(record)
    with store._connect() as connection:
        lesson_id = lesson_report(connection)[0]["id"]
    assert store.quarantine("case", record.id, "incorrect source pairing")
    assert store.quarantine("lesson", lesson_id, "derived from suspect case")
    assert not store.quarantine("case", record.id, "already marked")
    assert store.search(signature, "musa", "musa") == []
    assert store.search_verified_seeds(signature, "musa", "musa") == []
    with store._connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM experience_cases WHERE id = ?", (record.id,)
        ).fetchone()[0] == 1
        assert lesson_report(connection) == []
