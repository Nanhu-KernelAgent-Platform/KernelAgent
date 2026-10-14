import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_kernel_agent.kernel_backend import KernelBundle
from triton_kernel_agent.experience import (
    ExperienceRecord, SQLiteExperienceStore, build_operator_signature,
)
from triton_kernel_agent.opt_manager import (
    OptimizationManager,
    _experience_direction,
    _target_improvement_reached,
    _validate_target_improvement_pct,
)


def test_experience_direction_requires_explicit_optimization_target():
    assert _experience_direction(
        'KERNELAGENT_OPTIMIZATION_TARGET = "forward"\n'
    ) == "forward"
    assert _experience_direction(
        "KERNELAGENT_OPTIMIZATION_TARGET = 'backward'\n"
    ) == "backward"
    assert _experience_direction(
        "def backward(ctx, grad_output): return grad_output"
    ) == "unknown"

_SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "run_opt_manager.py"
_SPEC = importlib.util.spec_from_file_location("run_opt_manager", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
summarize_result = _MODULE.summarize_result
write_best_kernel = _MODULE.write_best_kernel


def _result(**overrides):
    result = {
        "success": True,
        "kernel_code": "def kernel_function(): pass\n",
        "best_time_ms": 0.8,
        "initial_kernel_time_ms": 1.0,
        "pytorch_baseline_ms": 1.2,
        "total_rounds": 1,
        "top_kernels": [{"time_ms": 0.8}],
    }
    result.update(overrides)
    return result


def test_summarize_result_distinguishes_outcomes():
    assert summarize_result(_result())["status"] == "SUCCESS"
    assert (
        summarize_result(_result(best_time_ms=1.0))["status"] == "NO_GAIN"
    )
    assert (
        summarize_result(_result(top_kernels=[]))["status"] == "DEGRADED"
    )
    assert (
        summarize_result(_result(success=False, top_kernels=[]))["status"]
        == "FAILED"
    )


def test_write_best_kernel_materializes_native_bundle(tmp_path):
    files = {
        "kernel.py": "def kernel_function(x): return x\n",
        "binding.cpp": "// binding\n",
        "kernel.mu": "// kernel\n",
        "setup.py": "# setup\n",
    }
    result = _result(kernel_code=KernelBundle(files).render_for_prompt())
    paths = write_best_kernel(result, "MUSA", tmp_path)

    serialized = tmp_path / "optimized_kernel_musa.py"
    assert paths["serialized_kernel"] == str(serialized)
    assert serialized.is_file()
    assert paths["best_bundle_dir"] == str(tmp_path / "best_bundle")
    for name, content in files.items():
        assert (tmp_path / "best_bundle" / name).read_text() == content


def test_optimization_summary_is_json_serializable():
    json.dumps(summarize_result(_result()))


def test_target_improvement_uses_initial_kernel_latency():
    assert not _target_improvement_reached(10.0, 8.1, 20.0)
    assert _target_improvement_reached(10.0, 8.0, 20.0)
    assert _target_improvement_reached(10.0, 7.5, 20.0)


def test_manager_uses_backend_timing_windows_when_available(tmp_path):
    manager = OptimizationManager.__new__(OptimizationManager)
    manager.benchmarker = SimpleNamespace(
        benchmark_kernel_samples=lambda _source, _problem: [0.11, 0.09, 0.10]
    )
    assert manager._benchmark_initial_kernel_samples(
        "source", tmp_path / "problem.py"
    ) == [0.11, 0.09, 0.10]


def _run_target_scenario(tmp_path, target):
    best = {}

    def initialize(initial):
        best["value"] = initial

    def update(results, round_num):
        best["value"] = SimpleNamespace(
            program_id=f"round-{round_num}",
            kernel_code=f"kernel-{round_num}",
            metrics=SimpleNamespace(time_ms=results[0]["time_ms"]),
            generation=round_num,
        )

    strategy = SimpleNamespace(
        initialize=initialize,
        select_candidates=lambda round_num: [{"round": round_num}],
        update_with_results=update,
        get_best_program=lambda: best["value"],
    )
    manager = OptimizationManager.__new__(OptimizationManager)
    manager.max_rounds = 3
    manager.strategy = strategy
    manager.experience_store = None
    manager.worker_kwargs = {"experience_context": "stale operator experience"}
    manager.worker_runner = SimpleNamespace(
        worker_kwargs={"experience_context": "stale operator experience"}
    )
    manager.logger = SimpleNamespace(info=lambda *_args: None, warning=lambda *_args: None)
    manager.database = SimpleNamespace(get_top_k=lambda _limit: [best["value"]])
    manager._baseline_profile_cache = {}
    manager._verify_initial_kernel = lambda *_args: True
    manager._benchmark_pytorch_baseline = lambda *_args: 12.0
    manager._benchmark_pytorch_compile = lambda *_args: 11.0
    manager._benchmark_initial_kernel = lambda *_args: 10.0
    manager._record_experiences = lambda _results: None

    rounds = []
    times = iter([9.0, 8.0, 7.0])

    def run_workers(_candidates, round_num, *_args):
        rounds.append(round_num)
        return [{"success": True, "worker_id": 0, "time_ms": next(times)}]

    manager._run_workers = run_workers
    problem = tmp_path / "problem.py"
    problem.write_text("class Model: pass\n")
    result = manager.run_optimization(
        initial_kernel="initial",
        problem_file=problem,
        test_code="test",
        max_rounds=3,
        target_improvement_pct=target,
    )
    assert "experience_context" not in manager.worker_kwargs
    assert "experience_context" not in manager.worker_runner.worker_kwargs
    return result, rounds


def test_target_stops_before_starting_the_next_round(tmp_path):
    result, rounds = _run_target_scenario(tmp_path, 20)
    assert rounds == [1, 2]
    assert result["total_rounds"] == 2
    assert result["target_reached"] is True
    assert result["termination_reason"] == "target_improvement_reached"


def test_blank_target_runs_every_round(tmp_path):
    result, rounds = _run_target_scenario(tmp_path, None)
    assert rounds == [1, 2, 3]
    assert result["target_reached"] is False


@pytest.mark.parametrize(
    "retested_ms,adopted,reworded,same_source",
    [(7.0, True, False, False), (7.0, True, True, False),
     (11.0, False, False, False), (None, False, False, False),
     (9.9, False, False, False),
     ("invalid", False, False, False), (7.0, False, False, True)],
)
def test_historical_start_requires_current_faster_benchmark(
    tmp_path, retested_ms, adopted, reworded, same_source,
):
    problem = tmp_path / "problem.py"
    problem.write_text(
        'KERNELAGENT_OPTIMIZATION_TARGET = "forward"\n'
        'def get_inputs(): return [torch.randn(64, 1568, dtype=torch.float32)]\n'
        'def forward(x, weight): return torch.relu(torch.matmul(x, weight.T))\n'
    )
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    old = ExperienceRecord(
        signature=build_operator_signature(
            problem.read_text() + ("\n# previous session wording" if reworded else "")
        ),
        platform="musa", kernel_backend="musa", direction="forward",
        outcome="improved", verified=True,
        kernel_code="fresh source" if same_source else "old verified source",
        baseline_time_ms=12.0, kernel_time_ms=6.0, improvement_pct=50,
        hardware="MTT S5000",
    )
    store.add_online(old, session_id="prior")
    state = {}
    strategy = SimpleNamespace(
        initialize=lambda entry: state.update(best=entry),
        select_candidates=lambda _round: [{"parent": state["best"]}],
        update_with_results=lambda *_args: None,
        get_best_program=lambda: state["best"],
    )
    manager = OptimizationManager.__new__(OptimizationManager)
    manager.max_rounds = 1
    manager.strategy = strategy
    manager.experience_store = store
    manager.openai_model = "test-model"
    manager.worker_kwargs = {"target_platform": "musa", "kernel_backend": "musa"}
    manager.worker_runner = SimpleNamespace(worker_kwargs={})
    manager.benchmarker = SimpleNamespace(last_hardware="MTT S5000")
    from triton_kernel_agent.measurement_policy import interleaved_pair
    def measure_pair(parent, candidate, _problem):
        return interleaved_pair(lambda source: {
            "time_ms": (10.0 if source == parent else retested_ms),
            "samples_ms": ([10.0] * 3 if source == parent else
                           [retested_ms] * 3 if isinstance(retested_ms, (int, float)) else []),
        }, parent, candidate)
    manager.benchmarker.benchmark_pair = measure_pair
    manager.logger = SimpleNamespace(info=lambda *_args: None, warning=lambda *_args: None)
    manager.database = SimpleNamespace(get_top_k=lambda _limit: [state["best"]])
    manager._baseline_profile_cache = {}
    manager._verify_initial_kernel = lambda source, *_args: (
        source != old.kernel_code or retested_ms != "invalid"
    )
    manager._benchmark_pytorch_baseline = lambda *_args: 12.0
    manager._benchmark_pytorch_compile = lambda *_args: 11.0
    manager._benchmark_initial_kernel = lambda source, _problem: (
        10.0 if same_source else
        retested_ms if source == old.kernel_code else 10.0
    )
    manager._record_experiences = lambda _results: None
    parents = []
    manager._run_workers = lambda candidates, *_args: (
        parents.append(candidates[0]["parent"].kernel_code) or []
    )
    result = manager.run_optimization(
        "fresh source", problem, "test", target_improvement_pct=20,
    )
    assert bool(result["historical_seed_id"]) is adopted
    if adopted:
        assert result["historical_seed_id"].endswith(f":{old.id}")
    assert result["kernel_code"] == (old.kernel_code if adopted else "fresh source")
    assert result["initial_kernel_time_ms"] == 10.0
    assert parents == ([] if adopted else ["fresh source"])
    with store._connect() as connection:
        stored = connection.execute(
            "SELECT hardware, environment_json FROM experience_cases "
            "WHERE action='Benchmarked verified starting kernel'"
        ).fetchone()
        assert stored["hardware"] == "MTT S5000"
        assert json.loads(stored["environment_json"])["hardware_source"] == "benchmark"
        event = connection.execute(
            "SELECT seed_ids_json,outcome_json FROM experience_retrieval_events "
            "WHERE stage='optimization_seed'"
        ).fetchone()
    assert json.loads(event["seed_ids_json"])[0].endswith(f":{old.id}")
    assert json.loads(event["outcome_json"])["adopted"] is adopted
    assert json.loads(event["outcome_json"])["candidate_match"] == (
        "provisional_contract" if reworded else "exact_text"
    )
    assert json.loads(event["outcome_json"])["text_contract_match"] is (not reworded)
    assert json.loads(event["outcome_json"])["graph_contract_match"] is False
    assert json.loads(event["outcome_json"])["decision"] == (
        "same_source" if same_source else
        "verification_failed" if retested_ms == "invalid"
        else "invalid_benchmark" if retested_ms is None
        else "adopted" if adopted else "inconclusive" if retested_ms == 9.9 else "not_faster"
    )


def test_musa_runner_defaults_are_used_for_experience_namespace(
    tmp_path, monkeypatch
):
    from triton_kernel_agent.platform.registry import registry

    manager = OptimizationManager.__new__(OptimizationManager)
    manager.worker_kwargs = {}
    manager.log_dir = tmp_path
    manager.logger = SimpleNamespace(warning=lambda *_args: None)
    manager.benchmark_lock = object()
    manager.profiling_semaphore = object()
    manager.openai_model = "test-model"
    manager.high_reasoning_effort = False
    manager.bottleneck_override = None
    manager.gpu_ids = [0]
    manager.gpu_locks = {0: object()}
    runner = SimpleNamespace(worker_kwargs={
        "target_platform": "musa", "kernel_backend": "musa"
    })
    monkeypatch.setattr(
        registry, "create_from_config",
        lambda *_args, **_kwargs: {
            "verifier": object(), "benchmarker": object(), "worker_runner": runner
        },
    )
    monkeypatch.setattr(registry, "has", lambda *_args: False)
    manager._resolve_platform("musa")
    assert manager.worker_kwargs["target_platform"] == "musa"
    assert manager.worker_kwargs["kernel_backend"] == "musa"


def test_experience_records_measured_candidate_not_rolled_back_best(tmp_path):
    manager = OptimizationManager.__new__(OptimizationManager)
    manager.experience_store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    manager._active_experience_signature = build_operator_signature(
        "def forward(x): return torch.sigmoid(x)"
    )
    manager.worker_kwargs = {"target_platform": "musa", "kernel_backend": "musa"}
    manager.logger = SimpleNamespace(warning=lambda *_args: None)
    manager.openai_model = "test-model"
    manager._record_experiences([{
        "kernel_code": "best source",
        "attempt_kernel_code": "measured slower candidate",
        "attempt": {
            "passed_verification": True, "is_improvement": False,
            "time_before_ms": 1.0, "time_after_ms": 2.0,
            "kernel_samples_ms": [1.9, 2.0, 2.1],
            "baseline_samples_ms": [0.9, 1.0, 1.1],
            "evidence_baseline_ms": 1.0,
            "improvement_pct": -100.0,
        },
    }])
    records = manager.experience_store.search(
        manager._active_experience_signature, "musa", "musa"
    )
    assert len(records) == 1
    assert records[0].kernel_code == "measured slower candidate"
    assert records[0].outcome == "regressed"
    with manager.experience_store._connect() as connection:
        samples_json, baseline_json = connection.execute(
            "SELECT samples_json, baseline_samples_json "
            "FROM experiment_measurements"
        ).fetchone()
    assert json.loads(samples_json) == [1.9, 2.0, 2.1]
    assert json.loads(baseline_json) == [0.9, 1.0, 1.1]
    assert manager.experience_store.search_verified_seeds(
        manager._active_experience_signature, "musa", "musa"
    ) == []


@pytest.mark.parametrize("value", [0, -1, 100.01, float("inf"), "20", True])
def test_target_improvement_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="target_improvement_pct"):
        _validate_target_improvement_pct(value)
    assert _validate_target_improvement_pct(None) is None
