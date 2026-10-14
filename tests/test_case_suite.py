"""Generic multi-input coverage: shapes, layouts, gradients and evidence."""
from dataclasses import replace
from types import ModuleType

import pytest
import torch

from triton_kernel_agent.case_suite import (
    benchmark_cases, load_cases, measurement_is_covered, move, paired_cases, reference_contracts,
    suite_test, verify_cases,
    run_guarded_test,
)
from triton_kernel_agent.experience import ExperienceRecord, OperatorSignature, SQLiteExperienceStore
from triton_kernel_agent.experience.experiment_db import run_report
from triton_kernel_agent.experience.graph_identity import trace_reference_graph

SOURCE = '''import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x * x + 0.1 * x
def get_inputs():
    return [torch.randn(2, 4)]
def get_init_inputs():
    return []
def get_test_cases():
    return [{"case_key": "other-shape", "inputs": [torch.randn(3, 7)]},
            {"case_key": "strided", "inputs": [torch.randn(4, 10)[:, ::2]]}]
'''


def test_runtime_guard_allows_normal_test_exit(tmp_path):
    path = tmp_path/'normal.py'
    path.write_text('import torch\nassert torch.allclose(torch.ones(2), torch.ones(2))\nraise SystemExit(0)\n')
    with pytest.raises(SystemExit) as stopped:
        run_guarded_test(path)
    assert stopped.value.code == 0


def test_runtime_guard_detects_dynamic_patch_despite_success_exit(tmp_path, monkeypatch):
    path = tmp_path/'modified.py'
    path.write_text('import torch\nsetattr(torch, "all" + "close", lambda *args, **kwargs: True)\nraise SystemExit(0)\n')
    # Undo the deliberately changed callable after the test, including on failure.
    monkeypatch.setattr(torch, 'allclose', torch.allclose)
    with pytest.raises(RuntimeError, match='verification behavior.*allclose'):
        run_guarded_test(path)


def test_musa_single_forward_case_also_gets_deterministic_companion():
    source = SOURCE[:SOURCE.index('def get_test_cases')]
    companion = suite_test(source, device='musa')
    assert companion and 'verify_cases' in companion and "'musa', 'forward'" in companion


def problem(source=SOURCE):
    module = ModuleType("case_reference")
    exec(source, module.__dict__)
    return module


def good(x):
    return x * x + 0.1 * x


def test_explicit_shapes_and_strides_are_all_verified():
    rows = verify_cases(problem(), good, "cpu")
    assert [row["case_key"] for row in rows] == ["default", "other-shape", "strided"]
    assert all(row["status"] == "passed" for row in rows)
    assert rows[-1]["contract"]["inputs"][0]["stride"] == [10, 2]
    assert move(torch.randn(4, 10)[:, ::2], "cpu").stride() == (10, 2)


def test_primary_success_does_not_hide_other_shape_failure():
    def primary_only(x):
        return good(x) if tuple(x.shape) == (2, 4) else torch.zeros_like(x)
    rows = verify_cases(problem(), primary_only, "cpu")
    assert [row["status"] for row in rows] == ["passed", "failed", "failed"]
    assert all(row["error"] for row in rows[1:])


def test_nonuniform_upstream_detects_wrong_backward():
    class BadGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return good(x)
        @staticmethod
        def backward(ctx, upstream):
            x, = ctx.saved_tensors
            return 2 * x + 0.1  # Incorrectly assumes all-one upstream.
    assert all(row["status"] == "passed" for row in verify_cases(problem(), BadGradient.apply, "cpu", "forward"))
    rows = verify_cases(problem(), BadGradient.apply, "cpu", "joint")
    assert all(row["status"] == "failed" for row in rows)
    assert all(row["status"] == "passed" for row in verify_cases(problem(), good, "cpu", "joint"))


def test_backward_contract_records_saved_state_and_named_parameters():
    source = SOURCE.replace("class Model(torch.nn.Module):", '''class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))''').replace("return x * x + 0.1 * x", "return self.scale * x * x + 0.1 * x")
    rows = reference_contracts(problem(source), "backward")
    backward = rows[0]["contract"]["backward"]
    assert [item["name"] for item in backward["requested_gradients"]] == ["input:0", "parameter:scale"]
    assert len(backward["gradient_outputs"]) == 2
    assert backward["saved_state"]
    assert backward["upstream_gradients"][0]["shape"] == [2, 4]


def test_case_validation_does_not_invent_coverage():
    module = problem(SOURCE[:SOURCE.index("def get_test_cases")])
    assert len(load_cases(module)) == 1
    assert suite_test(SOURCE[:SOURCE.index("def get_test_cases")]) is None
    for invalid in ([{"case_key": "default", "inputs": []}], [{"case_key": "x"}], "not-a-list"):
        module.get_test_cases = lambda: invalid
        with pytest.raises(ValueError):
            load_cases(module)


def test_per_case_timing_and_contract_pairing():
    parent = benchmark_cases(problem(), good, "cpu", "forward", lambda invoke: [1., 2., 3.])
    candidate = benchmark_cases(problem(), good, "cpu", "forward", lambda invoke: [2., 3., 4.])
    rows = paired_cases(parent, candidate)
    assert len(rows) == 3
    assert all(row["baseline_samples_ms"] == [1., 2., 3.] for row in rows)
    assert all(row["benchmark"]["paired"] for row in rows)
    parent["case_measurements"][0]["inputs"] = {}
    assert paired_cases(parent, candidate)[0]["baseline_samples_ms"] == []
    assert candidate["aggregation"] == "equal_weight_mean_windows_v1"
    checked = verify_cases(problem(), good, "cpu")
    assert measurement_is_covered(candidate["case_measurements"], checked)
    assert not measurement_is_covered(candidate["case_measurements"], checked[:-1])
    checked[0]["contract"]["inputs"][0]["shape"] = [9, 9]
    assert not measurement_is_covered(candidate["case_measurements"], checked)


def test_case_correctness_and_timings_are_separate_immutable_evidence(tmp_path):
    module = problem()
    results = verify_cases(module, good, "cpu")
    parent = benchmark_cases(module, good, "cpu", "forward", lambda invoke: [1., 2., 3.])
    candidate = benchmark_cases(module, good, "cpu", "forward", lambda invoke: [2., 3., 4.])
    store = SQLiteExperienceStore(tmp_path / "experience.sqlite3")
    record = ExperienceRecord(signature=OperatorSignature("elementwise"), platform="musa", kernel_backend="musa",
        direction="forward", outcome="regressed", verified=True, kernel_code="kernel",
        baseline_time_ms=2., kernel_time_ms=3., kernel_samples_ms=(2., 3., 4.),
        baseline_samples_ms=(1., 2., 3.), case_results=tuple(results),
        case_measurements=tuple(paired_cases(parent, candidate)),
        lesson_claims=({"text": "This change regresses", "claim_type": "performance_warning", "conditions": {}},))
    run_id = store.add_online(record, session_id="first")
    with store._connect() as connection:
        attempt = run_report(connection, run_id)["attempts"][0]
    assert len(attempt["case_results"]) == 3
    assert len(attempt["measurements"]) == 4  # Three cases plus separately labelled suite score.
    assert len(attempt["claims"]) == 4  # Three typed citations plus inconclusive raw text.
    assert all(row["relation"] == "supports" for row in attempt["claims"] if row["claim_type"] == "performance_warning")
    bad_results = [dict(row) for row in results]
    bad_results[-1]["status"] = "missing"
    with pytest.raises(ValueError, match="verified attempt"):
        store.add_online(replace(record, case_results=tuple(bad_results)), session_id="invalid")
    with store._connect() as connection:
        assert connection.execute("SELECT count(*) FROM experiment_runs").fetchone()[0] == 1


def test_single_forward_default_keeps_paired_windows_and_database_contract(tmp_path):
    from triton_kernel_agent.measurement_policy import interleaved_pair
    source = SOURCE[:SOURCE.index('def get_test_cases')].replace(
        'torch.randn(2, 4)', 'torch.randn(4, 10)[:, ::2]')
    module = problem(source)
    checked = verify_cases(module, good, 'cpu')
    before, after = interleaved_pair(lambda role: benchmark_cases(
        module, good, 'cpu', 'forward', lambda invoke: [1.] * 3 if role == 'parent' else [.5] * 3),
        'parent', 'candidate')
    assert measurement_is_covered(before['case_measurements'], checked)
    assert measurement_is_covered(after['case_measurements'], checked)
    rows = paired_cases(before, after)
    assert len(rows) == 1 and rows[0]['case_key'] == 'default'
    assert rows[0]['benchmark']['paired']
    assert len(rows[0]['baseline_samples_ms']) == len(rows[0]['samples_ms']) == 15
    store = SQLiteExperienceStore(tmp_path/'experience.sqlite3')
    run_id = store.add_online(ExperienceRecord(
        signature=OperatorSignature('elementwise'), platform='musa', kernel_backend='musa',
        direction='forward', hardware='test-device', outcome='improved', verified=True, kernel_code='test fixture',
        baseline_time_ms=before['time_ms'], kernel_time_ms=after['time_ms'],
        baseline_samples_ms=tuple(before['samples_ms']), kernel_samples_ms=tuple(after['samples_ms']),
        comparison=after['comparison'], case_results=tuple(checked), case_measurements=tuple(rows)),
        session_id='single-forward')
    with store._connect() as connection:
        measurement = connection.execute("select m.* from experiment_measurements m join experiment_attempts a "
                                         "on a.id=m.attempt_id where a.run_id=? and m.case_key='default'",
                                         (run_id,)).fetchone()
    import json
    assert measurement['kernel_ms'] == .5 and measurement['baseline_ms'] == 1.
    assert len(json.loads(measurement['samples_json'])) == len(json.loads(measurement['baseline_samples_json'])) == 15
    assert json.loads(measurement['inputs_json'])['inputs'][0]['stride'] == [10, 2]


def test_failure_has_case_evidence_without_fabricated_timing(tmp_path):
    results = verify_cases(problem(), lambda x: torch.zeros_like(x), "cpu")
    record = ExperienceRecord(signature=OperatorSignature("elementwise"), platform="musa", kernel_backend="musa",
        direction="forward", outcome="failed", verified=False, kernel_code="bad",
        case_results=tuple(results), lesson_claims=({"text": "The proposal fails these checks",
        "claim_type": "correctness_constraint", "predicate": "candidate_fails_correctness"},))
    store = SQLiteExperienceStore(tmp_path / "experience.sqlite3")
    run_id = store.add_online(record, session_id="failed")
    with store._connect() as connection:
        attempt = run_report(connection, run_id)["attempts"][0]
    assert not attempt["measurements"]
    assert len(attempt["case_results"]) == 3
    assert all(row["relation"] == "supports" and row["measurement_id"] is None for row in attempt["claims"] if row["claim_type"] == "correctness_constraint")


def test_backward_identity_is_not_forward_graph():
    forward = trace_reference_graph(SOURCE)
    backward = trace_reference_graph("KERNELAGENT_OPTIMIZATION_TARGET = 'backward'\n" + SOURCE)
    assert forward is not None and backward is not None
    assert backward["version"] == "torch_make_fx_forward_backward_v1"
    assert backward["digest"] != forward["digest"]
    assert backward["interface"]["cases"][0]["contract"]["backward"]["requested_gradients"]


def test_missing_expected_coverage_is_recorded_without_fabricated_success(tmp_path):
    rows = reference_contracts(problem(), "forward")
    signature = OperatorSignature("elementwise", interface={"explicit_case_suite": True, "cases": rows})
    store = SQLiteExperienceStore(tmp_path / "experience.sqlite3")
    record = ExperienceRecord(signature=signature, platform="musa", kernel_backend="musa",
        direction="forward", outcome="failed", verified=False, kernel_code="failed build")
    run_id = store.add_online(record, session_id="failed-build")
    with store._connect() as connection:
        attempt = run_report(connection, run_id)["attempts"][0]
    assert [row["status"] for row in attempt["case_results"]] == ["missing"] * 3
    assert attempt["decision"]["case_coverage"]["complete"] is False
    assert not attempt["measurements"]


def test_companion_runs_before_primary_and_does_not_skip_coverage(tmp_path):
    import logging
    from types import SimpleNamespace
    from triton_kernel_agent.worker import VerificationWorker
    worker = VerificationWorker.__new__(VerificationWorker)
    worker.workdir = tmp_path
    worker.kernel_file = tmp_path / "kernel.py"
    worker._kernel_backend = SimpleNamespace(name="triton")
    worker.logger = logging.getLogger(__name__)
    worker.test_timeout_s = 30
    worker._write_files("def kernel_function(x): return x * x + 0.1 * x\n", [
        "raise RuntimeError('primary test fails')\n", suite_test(SOURCE, "forward", "cpu")])
    success, _, error = worker._run_test()
    assert not success and "primary test fails" in error
    assert len(worker.last_case_results) == 3
    assert all(row["status"] == "passed" for row in worker.last_case_results)


def test_backward_without_extra_cases_still_checks_primary_gradients():
    source = SOURCE[:SOURCE.index("def get_test_cases")]
    assert suite_test(source, "backward", "cpu") is not None
    rows = reference_contracts(problem(source), "backward")
    assert len(rows) == 1 and rows[0]["case_key"] == "default"


def test_runtime_contract_mismatch_does_not_get_verified_timing(tmp_path):
    results = verify_cases(problem(), good, "cpu")
    timing = benchmark_cases(problem(), good, "cpu", "forward", lambda invoke: [1., 2., 3.])
    timing["case_measurements"][0]["inputs"]["inputs"][0]["shape"] = [999]
    record = ExperienceRecord(signature=OperatorSignature("elementwise"), platform="musa", kernel_backend="musa",
        direction="forward", outcome="success", verified=True, kernel_code="kernel",
        case_results=tuple(results), case_measurements=tuple(paired_cases({}, timing)))
    store = SQLiteExperienceStore(tmp_path / "experience.sqlite3")
    run_id = store.add_online(record, session_id="mismatched-contract")
    with store._connect() as connection:
        attempt = run_report(connection, run_id)["attempts"][0]
    assert next(row for row in attempt["measurements"] if row["case_key"] == "default")["verified"] == 0
