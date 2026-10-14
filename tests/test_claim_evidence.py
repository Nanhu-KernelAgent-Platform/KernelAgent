"""Regression checks for observed facts, negative lessons, and model distrust."""

import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from triton_kernel_agent.experience import ExperienceRecord, OperatorSignature, SQLiteExperienceStore
from triton_kernel_agent.experience.evidence import normalize_claims
from triton_kernel_agent.experience.conditions import check, context
from triton_kernel_agent.experience.experiment_db import lesson_report, record_bundle, run_report
from triton_kernel_agent.experience.store import format_experience_context
from triton_kernel_agent.opt_manager import OptimizationManager
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import (
    OptimizationAttempt, OptimizationOrchestrator,
)
from triton_kernel_agent.prompt_manager import PromptManager


def _record(*, shape=128, candidate=2.0, verified=True, claims=()):
    return ExperienceRecord(
        signature=OperatorSignature("elementwise", dtypes=("float32",), shapes=((shape,),)),
        platform="musa", kernel_backend="musa", direction="forward",
        outcome="regressed" if verified else "failed", verified=verified,
        kernel_code=f"candidate-{shape}-{candidate}-{verified}",
        baseline_time_ms=1.0, kernel_time_ms=candidate if verified else None,
        incumbent_time_ms=1.0, selected_by_worker=False,
        baseline_samples_ms=(0.99, 1.0, 1.01) if verified else (),
        kernel_samples_ms=(candidate * .99, candidate, candidate * 1.01) if verified else (),
        hardware="test-gpu", layout="contiguous", action="Change the launch configuration",
        lesson="Raw reflection with no declared proposition",
        lesson_claims=claims, error_message="output mismatch" if not verified else "",
    )


def _claim(kind):
    return {
        "text": "The tested launch change does not improve this workload" if kind ==
            "performance_warning" else "The tested launch change improves this workload",
        "claim_type": kind, "conditions": {"layout": "contiguous"},
        "relation": "supports",  # Model-proposed polarity must never be trusted.
    }


def _robust_record(**kwargs):
    from triton_kernel_agent.measurement_policy import interleaved_pair
    record = _record(**kwargs)
    parent, candidate = interleaved_pair(lambda role: {
        "samples_ms": [1.0 if role == "parent" else record.kernel_time_ms] * 3,
    }, "parent", "candidate")
    return replace(record, baseline_samples_ms=tuple(parent["samples_ms"]),
                   kernel_samples_ms=tuple(candidate["samples_ms"]), comparison=candidate["comparison"])


def _report(store, kind):
    with store._connect() as connection:
        return next(row for row in lesson_report(connection) if row["claim_type"] == kind)


@pytest.mark.parametrize("conditions,facts,status", [
    ({"dtype": ["float32", "float16"]}, {"dtype": ["torch.float32"]}, "compatible"),
    ({"dtype": ["float32", "float16"]}, {"dtype": ["float64"]}, "conflict"),
    ({"hardware": "GPU-A"}, {"hardware": "GPU-B"}, "conflict"),
    ({"hardware": "GPU-A"}, {}, "unknown"),
    ({"architecture": "arch-A"}, {"architecture": "arch-B"}, "conflict"),
    ({"direction": "forward"}, {"direction": "backward"}, "conflict"),
    ({"rank": [1, 2]}, {"rank": [2]}, "compatible"),
    ({"shape": [{"min": 64, "max": 256}, None]}, {"shape": [[128, 512]]}, "compatible"),
    ({"shape": [{"min": 64, "max": 256}, None]}, {"shape": [[512, 128]]}, "conflict"),
    ({"shape": [None, 128]}, {"shape": [[128]]}, "conflict"),
    ({"shape": [{"min": "64"}]}, {"shape": [[128]]}, "unknown"),
    ({"layout": "contiguous"}, {"layout": ["contiguous", "strided"]}, "conflict"),
    ({"free_form_constraint": "launch bound"}, {}, "unknown"),
    ({"free_form_constraint": "launch bound", "hardware": "GPU-A"}, {"hardware": "GPU-B"}, "conflict"),
])
def test_condition_compatibility_is_bounded_and_operator_independent(conditions, facts, status):
    assert check(conditions, facts) == status


def test_layout_comes_from_observed_input_strides_not_a_declared_default():
    signature = OperatorSignature("unknown", interface={"inputs": [
        {"shape": [128, 257], "dtype": "torch.float32", "stride": [514, 2]},
    ]})
    facts = context(signature, layout="contiguous")
    assert facts["layout"] == ["strided"]
    assert check({"layout": "contiguous"}, facts) == "conflict"


def test_condition_checks_cover_additional_cases_not_only_default_input():
    signature = OperatorSignature("unknown", interface={
        "inputs": [{"shape": [128], "dtype": "torch.float32", "stride": [1]}],
        "cases": [{"case_key": "strided", "contract": {"inputs": [
            {"shape": [256], "dtype": "torch.float32", "stride": [2]},
        ]}}],
    })
    facts = context(signature)
    assert check({"shape": [{"min": 64, "max": 512}]}, facts) == "compatible"
    assert check({"layout": "contiguous"}, facts) == "conflict"
    assert check({"shape": [128]}, facts) == "conflict"


@pytest.mark.parametrize("shape,stride", [([128], None), ([None], [1]), ([128], [True])])
def test_missing_or_malformed_tensor_metadata_does_not_borrow_declared_layout(shape, stride):
    tensor = {"shape": shape, "dtype": "torch.float32"}
    if stride is not None:
        tensor["stride"] = stride
    facts = context(OperatorSignature("unknown", interface={"inputs": [tensor]}), layout="contiguous")
    assert check({"layout": "contiguous"}, facts) == "unknown"


def test_explicit_per_input_layout_is_observation_not_a_contract_default():
    facts = context(OperatorSignature("unknown", interface={"inputs": [
        {"shape": [128], "dtype": "float32", "layout": "contiguous"},
    ]}))
    assert check({"layout": "contiguous"}, facts) == "compatible"


@pytest.mark.parametrize("family", ["elementwise", "reduction", "matmul"])
def test_principles_require_current_conditions_but_allow_contained_shapes(tmp_path, family):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    claim = _claim("performance_recommendation")
    claim["conditions"] = {"hardware": "GPU-A", "dtype": ["float32", "float16"],
                           "layout": "contiguous", "shape": [{"min": 64, "max": 512}]}
    for size in (128, 256):
        record = _robust_record(shape=size, candidate=.5, claims=(claim,))
        store.add_online(replace(record, hardware="GPU-A",
                         signature=replace(record.signature, semantic_type=family)), session_id=str(size))
    def retrieve(dtype="float32", hardware="GPU-A", layout="contiguous", size=512):
        return [r for r in store.retrieve(OperatorSignature(family, dtypes=(dtype,) if dtype else (),
                shapes=((size,),)), "musa", "musa", direction="forward", hardware=hardware,
                layout=layout) if r.kind == "principle"]
    assert len(retrieve()) == 1  # New shape inside a declared range, not an exact-task match.
    for query in ({"hardware": "GPU-B"}, {"hardware": ""}, {"dtype": "float64"},
                  {"dtype": ""}, {"layout": "strided"}, {"layout": ""}, {"size": 1024}):
        assert not retrieve(**query)
    hit = retrieve()[0]
    event = store.log_retrieval(session_id="audit", stage="optimization", signature=hit.signature,
                               records=[hit], direction="forward")
    with store._connect() as connection:
        audit = json.loads(connection.execute("SELECT selection_json FROM experience_retrieval_events WHERE id=?", (event,)).fetchone()[0])
    assert audit[0]["applicability"]["status"] == "compatible"
    assert audit[0]["applicability"]["eligible_sessions"] == 2


def test_unconstrained_shape_transfers_but_devices_do_not_pool_confirmation(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    claim = _claim("performance_recommendation")
    claim["conditions"] = {}  # No model assertion of universal hardware validity.
    for size, hardware in ((128, "GPU-A"), (256, "GPU-B")):
        store.add_online(replace(_robust_record(shape=size, candidate=.5, claims=(claim,)),
                                 hardware=hardware), session_id=str(size))
    signature = OperatorSignature("elementwise", dtypes=("float32",), shapes=((1024,),))
    def principles(hardware):
        return [r for r in store.retrieve(signature, "musa", "musa", direction="forward",
                hardware=hardware, layout="contiguous") if r.kind == "principle"]
    assert not principles("GPU-A") and not principles("GPU-B")
    store.add_online(replace(_robust_record(shape=512, candidate=.5, claims=(claim,)),
                             hardware="GPU-A"), session_id="third")
    assert len(principles("GPU-A")) == 1
    assert not principles("GPU-B")


@pytest.mark.parametrize("other_domain", [
    {"hardware": "GPU-B"}, {"dtype": "float16"}, {"layout": "strided"},
])
def test_counterevidence_is_scoped_to_its_actual_domain(tmp_path, other_domain):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    claim = _claim("performance_recommendation")
    claim["conditions"] = {}
    for size in (128, 256):
        store.add_online(replace(_robust_record(shape=size, candidate=.5, claims=(claim,)),
                                 hardware="GPU-A"), session_id=str(size))
    counter = replace(_robust_record(shape=512, candidate=2., claims=(claim,)), hardware="GPU-A")
    if "dtype" in other_domain:
        counter = replace(counter, signature=replace(counter.signature, dtypes=(other_domain["dtype"],)))
    else:
        counter = replace(counter, **other_domain)
    store.add_online(counter, session_id="counter")
    assert _report(store, "performance_recommendation")["status"] == "contested"
    hits = store.retrieve(OperatorSignature("elementwise", dtypes=("float32",), shapes=((1024,),)),
                          "musa", "musa", direction="forward", hardware="GPU-A", layout="contiguous")
    assert len([r for r in hits if r.kind == "principle"]) == 1


@pytest.mark.parametrize("conditions,reason", [
    ({"hardware": "GPU-B"}, "conditions_conflict"),
    ({"unrecognized": "condition"}, "conditions_unknown"),
])
def test_inapplicable_model_claims_keep_measurements_but_cannot_promote(tmp_path, conditions, reason):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    claim = _claim("performance_recommendation")
    claim["conditions"] = conditions
    for size in (128, 256):
        run_id = store.add_online(replace(_robust_record(shape=size, candidate=.5, claims=(claim,)),
                                         hardware="GPU-A"), session_id=str(size))
    with store._connect() as connection:
        attempt = run_report(connection, run_id)["attempts"][0]
    typed = next(c for c in attempt["claims"] if c["claim_type"] == "performance_recommendation")
    assert typed["relation"] == "inconclusive" and typed["reason"] == reason
    assert attempt["measurements"]
    assert _report(store, "performance_recommendation")["status"] == "candidate"
    # Simulate old pre-gate evidence. Read-time safety must not depend on new writes.
    with store._connect() as connection:
        connection.execute("UPDATE experiment_claim_evidence SET relation='supports', reason='noise_aware_parent_comparison'")
    assert _report(store, "performance_recommendation")["status"] == "candidate"
    with store._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiment_claim_evidence WHERE relation='supports'").fetchone()[0] > 0


@pytest.mark.parametrize("missing", ["hardware", "dtype", "layout"])
def test_missing_observed_domain_cannot_validate_an_unconditional_claim(tmp_path, missing):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    claim = _claim("performance_recommendation")
    claim["conditions"] = {}
    for size in (128, 256):
        record = _robust_record(shape=size, candidate=.5, claims=(claim,))
        record = replace(record, signature=replace(record.signature, dtypes=())) if missing == "dtype" else replace(record, **{missing: ""})
        store.add_online(record, session_id=str(size))
    assert _report(store, "performance_recommendation")["status"] == "candidate"


def test_parent_improvement_does_not_imply_incumbent_adoption(tmp_path):
    manager = OptimizationManager.__new__(OptimizationManager)
    manager.experience_store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    manager._active_experience_signature = replace(_record().signature, interface={"inputs": [
        {"shape": [128], "dtype": "torch.float32", "stride": [1]},
    ]})
    manager._active_experience_direction = "forward"
    manager.worker_kwargs = {"target_platform": "musa", "kernel_backend": "musa", "layout": "contiguous"}
    manager.logger = SimpleNamespace(warning=lambda *args: None)
    manager.openai_model = "test-model"
    manager._record_experiences([{
        "attempt_kernel_code": "measured candidate",
        "attempt": {
            "passed_verification": True, "is_improvement": False,
            "time_before_ms": 1.0, "time_after_ms": 1.05,
            "evidence_baseline_ms": 1.1, "baseline_samples_ms": [1.09, 1.1, 1.11],
            "kernel_samples_ms": [1.04, 1.05, 1.06], "improvement_pct": -5.0,
        },
        "reflexion": {"claims": [_claim("performance_recommendation")]},
    }])
    store = manager.experience_store
    with store._connect() as connection:
        run_id = connection.execute("SELECT id FROM experiment_runs").fetchone()[0]
        report = run_report(connection, run_id)["attempts"][0]
    assert report["outcome"] == "improved"
    assert report["decision"]["parent_gain_pct"] == pytest.approx(100 / 22)
    assert report["decision"]["incumbent_gain_pct"] == pytest.approx(-5)
    assert report["decision"]["selected_by_worker"] is False
    with store._connect() as connection:
        raw=json.loads(connection.execute('SELECT environment_json FROM experiment_runs WHERE id=?',(run_id,)).fetchone()[0])['raw_reflection']
    assert raw['claims'][0]['claim_type']=='performance_recommendation'
    assert report["claims"][0]["relation"] == "supports"
    selected = store.retrieve(_record().signature, "musa", "musa", direction="forward")
    assert "Selected by worker: False" in format_experience_context(selected)


def test_regression_supports_warning_and_contradicts_recommendation(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    run_id = store.add_online(_record(claims=(
        _claim("performance_warning"), _claim("performance_recommendation"),
    )), session_id="one")
    with store._connect() as connection:
        claims = run_report(connection, run_id)["attempts"][0]["claims"]
    relations = {row["claim_type"]: row["relation"] for row in claims}
    assert relations == {
        "performance_warning": "supports", "performance_recommendation": "contradicts",
        "unclassified": "inconclusive",
    }
    assert _report(store, "performance_warning")["status"] == "candidate"
    assert _report(store, "performance_recommendation")["status"] == "contested"


def test_warning_needs_independent_measured_support_and_honors_direction(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    for shape, session in ((128, "one"), (256, "one"), (256, "two")):
        store.add_online(_robust_record(shape=shape, claims=(_claim("performance_warning"),)),
                         session_id=session)
        expected = "validated" if session == "two" else "candidate"
        assert _report(store, "performance_warning")["status"] == expected
    hits = store.retrieve(_record().signature, "musa", "musa", direction="forward",
                          hardware="test-gpu", layout="contiguous")
    principles = [row for row in hits if row.kind == "principle"]
    assert len(principles) == 1
    assert principles[0].claim_type == "performance_warning"
    assert "Supporting observed workloads: 2" in format_experience_context(principles)
    assert store.retrieve(_record().signature, "musa", "musa", direction="backward") == []
    # A subsequent measured win is counterevidence to this exact warning.
    store.add_online(_record(shape=512, candidate=.5, claims=(_claim("performance_warning"),)),
                     session_id="three")
    assert _report(store, "performance_warning")["status"] == "contested"
    assert all(row.kind != "principle" for row in store.retrieve(
        _record().signature, "musa", "musa", direction="forward"))


def test_failed_correctness_has_attempt_evidence_without_invented_timings(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    record = _record(verified=False, claims=({
        "text": "This attempted change violates the tested output contract",
        "claim_type": "correctness_constraint", "predicate": "candidate_fails_correctness",
        "conditions": {"contract": "tested output shape and values"},
    }, _claim("performance_warning")))
    run_id = store.add_online(record, session_id="failure")
    with store._connect() as connection:
        attempt = run_report(connection, run_id)["attempts"][0]
    assert attempt["measurements"] == []
    assert attempt["decision"]["candidate_ms"] is None
    assert attempt["decision"]["correctness"] == "failed"
    relations = {row["claim_type"]: row["relation"] for row in attempt["claims"]}
    assert relations["correctness_constraint"] == "inconclusive"
    assert next(c for c in attempt["claims"] if c["claim_type"] == "correctness_constraint")["reason"] == "conditions_unknown"
    assert relations["performance_warning"] == "inconclusive"
    assert _report(store, "correctness_constraint")["status"] == "candidate"


def test_unclassified_summary_never_promotes_from_good_timings(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    for shape, session in ((128, "one"), (256, "two")):
        store.add_online(_record(shape=shape, candidate=.5), session_id=session)
    report = _report(store, "unclassified")
    assert report["observed_runs"] == 2
    assert report["inconclusive_runs"] == 2
    assert report["eligible_runs"] == 0
    assert report["status"] == "candidate"


def test_claims_reject_forged_model_relations_and_decisions(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    bundle = {
        "workload": {"semantic_type": "elementwise", "platform": "musa",
                     "kernel_backend": "musa", "contract": {"layout": "contiguous"}},
        "run": {"provenance": "manual"},
        "attempts": [{"id": "bad", "verified": True, "outcome": "regressed",
                      "decision": {"parent_comparison": "faster"},
                      "measurements": [{"case_key": "one", "verified": True,
                                        "baseline_ms": 1, "kernel_ms": 2}]}],
        "claims": [{**_claim("performance_recommendation"),
                    "evidence": [{"attempt_id": "bad", "case_key": "one",
                                  "relation": "supports"}]}],
    }
    with store._connect() as connection:
        run_id = record_bundle(connection, bundle)
        claim = run_report(connection, run_id)["attempts"][0]["claims"][0]
    assert claim["relation"] == "contradicts"


def test_unknown_claim_attempt_rolls_back_bundle(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    bundle = {
        "workload": {"semantic_type": "elementwise", "platform": "musa",
                     "kernel_backend": "musa", "contract": {}},
        "run": {}, "attempts": [],
        "claims": [{**_claim("performance_warning"),
                    "evidence": [{"attempt_id": "missing"}]}],
    }
    with pytest.raises(ValueError, match="cite an attempt"):
        with store._connect() as connection:
            record_bundle(connection, bundle)
    with store._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0


def test_quarantined_attempt_cannot_validate_claim(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    first = _robust_record(claims=(_claim("performance_warning"),))
    store.add_online(first, session_id="one")
    store.add_online(_robust_record(shape=256, claims=(_claim("performance_warning"),)), session_id="two")
    assert _report(store, "performance_warning")["status"] == "validated"
    store.quarantine("case", first.id, "bad measurement")
    assert _report(store, "performance_warning")["status"] == "candidate"


@pytest.mark.parametrize("claims", [None, {}, ["plain text"], [{"claim_type": {}}],
    [{"text": "x", "claim_type": "performance_warning", "predicate": {}}]])
def test_malformed_optional_claims_do_not_break_reflection(claims):
    assert normalize_claims(claims) == []
    orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    reflection = orchestrator._parse_reflexion_response(
        json.dumps({"reasoning": "raw explanation", "claims": claims}),
        OptimizationAttempt(round_num=1),
    )
    assert reflection.reasoning == "raw explanation"
    assert reflection.claims == []


def test_reflection_prompt_separates_comparison_and_selection():
    prompt = PromptManager().render_reflexion_prompt(OptimizationAttempt(
        round_num=1, time_before_ms=1, time_after_ms=1.05, evidence_baseline_ms=1.1,
        baseline_samples_ms=[1.09, 1.1, 1.11], kernel_samples_ms=[1.04, 1.05, 1.06],
        passed_verification=True, is_improvement=False,
    ))
    assert "Remeasured parent: 1.1ms" in prompt
    assert "Selected by worker: False" in prompt
    assert "candidate_does_not_improve_parent" in prompt


def test_reflection_failure_has_status_not_a_model_lesson(tmp_path):
    orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    orchestrator.logger = Mock()
    orchestrator.prompt_manager = Mock()
    orchestrator.provider = Mock()
    orchestrator.provider.get_response.side_effect = ConnectionError('offline')
    orchestrator.model = 'test'
    orchestrator.artifact_dir = tmp_path
    attempt = OptimizationAttempt(round_num=1, passed_verification=True, is_improvement=True,
                                  time_before_ms=1., time_after_ms=.8)
    reflection = orchestrator._generate_reflexion(attempt)
    assert reflection.generation_status == 'failed'
    assert reflection.generation_error == 'ConnectionError'
    assert not reflection.was_diagnosis_correct and not reflection.was_fix_effective
    assert not reflection.lessons and not reflection.claims
    assert reflection.format_for_prompt() == ''
    malformed = orchestrator._parse_reflexion_response('not json', attempt)
    assert malformed.generation_status == 'failed'
    assert malformed.generation_error == 'invalid_or_missing_response'


@pytest.mark.parametrize("verified, error", [(True, ""), (False, "output mismatch")])
def test_manager_preserves_attempt_without_timing_and_rejects_stale_reflection(
    tmp_path, verified, error,
):
    manager = OptimizationManager.__new__(OptimizationManager)
    manager.experience_store = SQLiteExperienceStore(tmp_path / "memory.sqlite3")
    manager._active_experience_signature = _record().signature
    manager.worker_kwargs = {"target_platform": "musa", "kernel_backend": "musa"}
    manager.logger = SimpleNamespace(warning=lambda *args: None)
    manager.openai_model = "test-model"
    manager._record_experiences([{
        "attempt_kernel_code": "candidate without timing",
        "attempt": {"round_num": 2, "passed_verification": verified,
                    "time_after_ms": float("inf"), "error_message": error},
        "reflexion": {"round_num": 1, "lessons": ["stale lesson"],
                      "claims": [_claim("performance_recommendation")]},
    }])
    with manager.experience_store._connect() as connection:
        run_id = connection.execute("SELECT id FROM experiment_runs").fetchone()[0]
        attempt = run_report(connection, run_id)["attempts"][0]
    assert attempt["verified"] == int(verified)
    assert attempt["measurements"] == []
    assert attempt["claims"] == []
    assert attempt["decision"]["candidate_ms"] is None
    assert attempt["decision"]["correctness"] == ("passed" if verified else "failed")
    assert attempt["error_message"] == error


def test_failed_reflection_does_not_fabricate_performance_or_generic_rule():
    orchestrator = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    reflection = orchestrator._generate_reflexion(OptimizationAttempt(
        round_num=2, passed_verification=False, error_message="mismatched output stride",
    ))
    assert reflection.round_num == 2
    assert "mismatched output stride" in reflection.reasoning
    assert reflection.lessons == []
    assert reflection.claims == []


def test_unversioned_database_is_rejected_without_altering_attempt(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE experiment_attempts (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, parent_id TEXT, parent_ref TEXT NOT NULL,
            source_hash TEXT NOT NULL, kernel_code TEXT NOT NULL, action TEXT NOT NULL,
            config_changes_json TEXT NOT NULL, outcome TEXT NOT NULL, verified INTEGER NOT NULL,
            error_message TEXT NOT NULL, created_at TEXT NOT NULL)""")
        connection.execute("INSERT INTO experiment_attempts VALUES "
                           "('old','old-run',NULL,'','hash','code','','{}','regressed',1,'','2025')")
    with pytest.raises(ValueError, match="Only experience schema 9"):
        SQLiteExperienceStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT outcome FROM experiment_attempts WHERE id='old'").fetchone()[0] == "regressed"
        assert "decision_json" not in {row[1] for row in connection.execute("PRAGMA table_info(experiment_attempts)")}
