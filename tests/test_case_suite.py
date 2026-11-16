"""Generic multi-input coverage: shapes, layouts, gradients and evidence."""
from dataclasses import replace
from types import ModuleType

import pytest
import torch

from triton_kernel_agent.case_suite import (
    benchmark_cases, load_cases, measurement_is_covered, move, paired_cases, reference_contracts,
    reference_slices, suite_test, verify_cases,
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


# --- Verification-gate tolerance: KernelBench official 1e-2 for long reductions ---
TOLERANCE_SOURCE = '''import torch
class Model(torch.nn.Module):
    def forward(self, A, B):
        return torch.matmul(A, B)
def get_inputs():
    return [torch.full((8, 64), 4.0), torch.full((64, 1), 4.0)]
def get_init_inputs():
    return []
'''

# Reference value is exactly 1024.0, so a 3e-6 relative deviation (0.0031) is far
# outside torch's fp32 default (atol 1e-5 + rtol 1.3e-6*1024) but far inside the
# KernelBench fp32 standard (atol 1e-2 + rtol 1e-2*1024).
def slightly_off_matmul(A, B):
    return torch.matmul(A, B) * (1 + 3e-6)


def zero_matmul(A, B):
    return torch.zeros(A.shape[0], B.shape[1])


def test_default_case_keeps_torch_dtype_tolerance_without_override(monkeypatch):
    for name in ("KERNELAGENT_VERIFY_RTOL", "KERNELAGENT_VERIFY_ATOL",
                 "KERNELAGENT_VERIFY_TOLERANCE_POLICY"):
        monkeypatch.delenv(name, raising=False)
    rows = verify_cases(problem(TOLERANCE_SOURCE), slightly_off_matmul, "cpu")
    assert rows[0]["status"] == "failed"
    assert "1.3e-06" in rows[0]["error"]


def test_kernelbench_tolerance_env_accepts_reduction_order_noise(monkeypatch):
    monkeypatch.setenv("KERNELAGENT_VERIFY_RTOL", "0.01")
    monkeypatch.setenv("KERNELAGENT_VERIFY_ATOL", "0.01")
    monkeypatch.setenv("KERNELAGENT_VERIFY_TOLERANCE_POLICY", "kernelbench_official_fp32")
    rows = verify_cases(problem(TOLERANCE_SOURCE), slightly_off_matmul, "cpu")
    assert [row["status"] for row in rows] == ["passed"]
    assert rows[0]["contract"]["verification_tolerance"] == {
        "rtol": 0.01, "atol": 0.01, "policy": "kernelbench_official_fp32"}


def test_explicit_case_tolerance_still_overrides_the_env_default(monkeypatch):
    monkeypatch.setenv("KERNELAGENT_VERIFY_RTOL", "0.01")
    monkeypatch.setenv("KERNELAGENT_VERIFY_ATOL", "0.01")
    monkeypatch.setenv("KERNELAGENT_VERIFY_TOLERANCE_POLICY", "kernelbench_official_fp32")
    source = TOLERANCE_SOURCE + '''def get_test_cases():
    return [{"case_key": "strict", "inputs": [torch.full((8, 64), 4.0), torch.full((64, 1), 4.0)],
             "rtol": 0, "atol": 0}]
'''
    rows = verify_cases(problem(source), slightly_off_matmul, "cpu")
    by_key = {row["case_key"]: row for row in rows}
    assert by_key["default"]["status"] == "passed"
    assert by_key["strict"]["status"] == "failed"
    assert by_key["default"]["contract"]["verification_tolerance"]["policy"] == "kernelbench_official_fp32"


def test_all_zero_candidate_is_never_excused_by_a_loose_tolerance(monkeypatch):
    monkeypatch.setenv("KERNELAGENT_VERIFY_RTOL", "0.01")
    monkeypatch.setenv("KERNELAGENT_VERIFY_ATOL", "0.01")
    rows = verify_cases(problem(TOLERANCE_SOURCE), zero_matmul, "cpu")
    assert rows[0]["status"] == "failed"
    assert "all-zero" in rows[0]["error"]


# --- Multi-output references: a tuple/list reference is not a single tensor ---
# The cp_seq_chunk_index_select case (KernelAgent case 6) returns
# (index_select(k_ag, 0, index), index_select(v_ag, 0, index)); the reference
# sanity probe used to call ``expected.numel()`` on that tuple and crashed with
# "AttributeError: 'tuple' object has no attribute 'numel'" on every round.
TUPLE_SOURCE = '''import torch
class Model(torch.nn.Module):
    def forward(self, k, v, index):
        return (torch.index_select(k, 0, index), torch.index_select(v, 0, index))
def get_inputs():
    return [torch.randn(2, 3, 5).to(torch.bfloat16), torch.randn(2, 3, 4).to(torch.bfloat16),
            torch.arange(2, dtype=torch.int32)]
def get_init_inputs():
    return []
'''


def tuple_reference(k, v, index):
    return (torch.index_select(k, 0, index), torch.index_select(v, 0, index))


def test_multi_output_reference_is_verified_without_tensor_only_assumptions():
    rows = verify_cases(problem(TUPLE_SOURCE), tuple_reference, "cpu")
    assert [row["status"] for row in rows] == ["passed"]
    outputs = rows[0]["contract"]["outputs"]
    assert [item["shape"] for item in outputs] == [[2, 3, 5], [2, 3, 4]]


def test_multi_output_reference_wrong_candidate_fails_cleanly():
    def zero_outputs(k, v, index):
        return (torch.zeros_like(torch.index_select(k, 0, index)),
                torch.zeros_like(torch.index_select(v, 0, index)))
    rows = verify_cases(problem(TUPLE_SOURCE), zero_outputs, "cpu")
    assert rows[0]["status"] == "failed"
    assert "all-zero" in rows[0]["error"]


def test_multi_output_all_zero_reference_is_compared_not_skipped():
    zero_source = TUPLE_SOURCE.replace(
        "return (torch.index_select(k, 0, index), torch.index_select(v, 0, index))",
        "return (torch.zeros_like(k[0:index.numel()]), torch.zeros_like(v[0:index.numel()]))")
    rows = verify_cases(problem(zero_source), tuple_reference, "cpu")
    assert rows[0]["status"] == "failed"
    assert rows[0]["reference_all_zero"] is True


# --- A degenerate graph is not an unsupported reference -----------------------
# KernelBench L2 #80: max over dim=1 collapses (1024, 8192) to (1024, 1) and the
# next step subtracts that row's own mean, which is exactly 0.0 for every row.
# The all-zero reference is the TRUE output, but the reference-sanity probe used
# to label it "reference-unsupported", and no status other than "passed" can
# satisfy the suite -- so 5/5 refine rounds were wasted and the correct zero
# kernel was never accepted.
DEGENERATE_SOURCE = '''import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(64, 32))
        self.bias = torch.nn.Parameter(torch.randn(64))
    def forward(self, x):
        y = x @ self.weight.t() + self.bias
        m = torch.max(y, dim=1, keepdim=True).values
        return torch.nn.functional.gelu(m - m.mean(dim=1, keepdim=True))
def get_inputs():
    return [torch.rand(8, 32)]
def get_init_inputs():
    return []
'''


def test_degenerate_reference_is_verified_against_its_true_zero_output():
    def zero_kernel(x):
        return torch.zeros(x.shape[0], 1)
    rows = verify_cases(problem(DEGENERATE_SOURCE), zero_kernel, "cpu")
    assert [row["status"] for row in rows] == ["passed"]
    assert rows[0]["reference_all_zero"] is True


def test_degenerate_reference_still_rejects_a_non_zero_candidate():
    def wrong_kernel(x):
        return torch.ones(x.shape[0], 1)
    rows = verify_cases(problem(DEGENERATE_SOURCE), wrong_kernel, "cpu")
    assert rows[0]["status"] == "failed"


# --- Reference chunking must not fire when output rows do not scale with inputs[0] ---
# moe_unpermute_mask (KernelAgent case 15): out rows = row_id_map.shape[1], while
# pin has 32768 activation rows. The single-probe estimate multiplied the fixed
# 4096-row output by 32768 rows, split into 512 slices and the concat collapsed
# to all-zero, failing every round as "reference-unsupported".
NONSCALING_SOURCE = '''import torch
class Model(torch.nn.Module):
    def forward(self, x, w):
        return torch.ones(w.shape[0], x.shape[1]) * x[0:1].mean(dim=0)
def get_inputs():
    return [torch.randn(2048, 8), torch.ones(3, dtype=torch.int64)]
def get_init_inputs():
    return []
'''


def test_fixed_row_output_is_not_split_by_leading_input_rows():
    module = problem(NONSCALING_SOURCE)
    x, w = (t.to("cpu") for t in module.get_inputs())
    assert reference_slices(module.Model(), [x, w]) == [slice(None)]
    rows = verify_cases(module, module.Model().forward, "cpu")
    assert [row["status"] for row in rows] == ["passed"]
    assert rows[0]["reference_chunked"] is False
