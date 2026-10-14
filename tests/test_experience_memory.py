import json
import math
import sqlite3
import pytest

from triton_kernel_agent.experience import (
    ExperienceRecord,
    SQLiteExperienceStore,
    build_operator_signature,
    format_experience_context,
    format_verified_seed_context,
)
def _problem(shape: tuple[int, int] = (1024, 4096)) -> str:
    return f"""
import torch
M, N = {shape}
class Model:
    def forward(self, x):
        return torch.rms_norm(x, (N,))
def get_inputs():
    return [torch.randn((M, N), dtype=torch.bfloat16)]
"""


def test_signature_extracts_operator_dtype_and_shape():
    signature = build_operator_signature(_problem())
    assert signature.semantic_type == "rmsnorm"
    assert "bfloat16" in signature.dtypes
    assert (1024, 4096) in signature.shapes
    assert signature.digest


def test_failed_benchmark_retrieval_audit_uses_strict_json(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    event_id = store.log_retrieval(
        session_id="failed-benchmark", stage="optimization",
        signature=build_operator_signature(_problem()), records=[],
    )
    store.set_retrieval_outcome(event_id, {
        "success": False, "initial_kernel_time_ms": math.inf,
        "best_time_ms": math.inf,
    })
    with sqlite3.connect(store.path) as connection:
        raw = connection.execute(
            "SELECT outcome_json FROM experience_retrieval_events WHERE id = ?",
            (event_id,),
        ).fetchone()[0]
    assert "Infinity" not in raw
    assert json.loads(raw) == {
        "success": False, "initial_kernel_time_ms": None,
        "best_time_ms": None,
    }


def test_prompt_retrieval_excludes_bare_generated_cases(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    bare = ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        outcome="generated", verified=True, kernel_code="bare source",
        action="Generated a correctness-verified kernel",
    )
    useful = ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        outcome="improved", verified=True, kernel_code="useful source",
        action="Increase grid parallelism", lesson="Small grids underutilize the GPU",
        improvement_pct=12,
    )
    store.add(bare)
    store.add(useful)
    assert [item.id for item in store.retrieve(
        signature, "musa", "musa", stage="generation"
    )] == [useful.id]


def test_correctness_only_generated_boilerplate_does_not_take_prompt_slot(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    boilerplate = ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        direction="forward", outcome="generated", verified=True,
        kernel_code="verified source", action="Generated a correctness-verified kernel",
        lesson="Reuse only after validating shape, dtype, and device constraints.",
    )
    store.add(boilerplate)
    assert store.retrieve(
        signature, "musa", "musa", stage="generation", direction="forward",
    ) == []
    assert [item.id for item in store.search_verified_seeds(
        signature, "musa", "musa", direction="forward",
    )] == [boilerplate.id]


def test_generated_seed_keeps_request_gate_with_observed_reference_signature(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    requested = build_operator_signature(_problem())
    observed = build_operator_signature(_problem() + "\n# verified test wording\n")
    assert requested.digest != observed.digest
    record = ExperienceRecord(
        signature=observed, platform="musa", kernel_backend="musa",
        direction="forward", outcome="generated", verified=True,
        kernel_code="verified source", environment={
            "request_signature": requested.to_dict(),
        },
    )
    store.add(record)
    assert [item.id for item in store.search_verified_seeds(
        requested, "musa", "musa", direction="forward",
    )] == [record.id]


def test_directional_cases_require_matching_observed_direction(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(
        "def forward(self, x): return F.relu(self.fc1(x))"
    )
    def add(direction, action, lesson, outcome="improved"):
        record = ExperienceRecord(
            signature=signature, platform="musa", kernel_backend="musa",
            direction=direction, outcome=outcome, verified=True,
            kernel_code=f"source: {direction}: {action}",
            action=action, lesson=lesson,
            improvement_pct=10 if outcome == "improved" else -5,
        )
        store.add(record)
        return record

    forward = add("forward", "Tune forward output tiles", "Reduce launch overhead")
    transferable = add(
        "backward", "Increase tile parallelism", "Improves occupancy",
    )
    backward_only = add(
        "backward", "Parallelize grad_bias reduction", "Reduce gradient latency",
        "regressed",
    )
    legacy_warning = add(
        "unknown", "Move grad_weight into a separate kernel", "Gradient overhead",
        "regressed",
    )
    legacy_ambiguous_warning = add(
        "unknown", "Change launch geometry", "Performance regressed",
        "regressed",
    )
    hits = store.retrieve(
        signature, "musa", "musa", stage="optimization",
        direction="forward", limit=4,
    )
    ids = {item.id for item in hits}
    assert forward.id in ids
    assert transferable.id not in ids
    assert backward_only.id not in ids
    assert legacy_warning.id not in ids
    assert legacy_ambiguous_warning.id not in ids
    assert all(item.direction == "forward" for item in hits)
    assert "Applicable execution directions: forward" in format_experience_context(hits)


def test_code_seeds_require_matching_execution_direction(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    backward = ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        direction="backward", outcome="generated", verified=True,
        kernel_code="backward source",
    )
    store.add(backward)
    assert store.search_verified_seeds(
        signature, "musa", "musa", direction="forward"
    ) == []
    assert [item.id for item in store.search_verified_seeds(
        signature, "musa", "musa", direction="backward"
    )] == [backward.id]


def test_online_direction_is_recorded_in_workload_and_retrieval_audit(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    record = ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        direction="backward", outcome="improved", verified=True,
        kernel_code="gradient kernel", baseline_time_ms=2.0,
        kernel_time_ms=1.0, improvement_pct=50,
    )
    store.add_online(record, session_id="backward-session")
    event_id = store.log_retrieval(
        session_id="backward-session", stage="optimization",
        signature=signature, records=[record], direction="backward",
    )
    with sqlite3.connect(store.path) as connection:
        workload = json.loads(connection.execute(
            "SELECT contract_json FROM experiment_workloads"
        ).fetchone()[0])
        event_direction = connection.execute(
            "SELECT direction FROM experience_retrieval_events WHERE id = ?",
            (event_id,),
        ).fetchone()[0]
    assert workload["direction"] == "backward"
    assert event_direction == "backward"


def test_store_deduplicates_and_retrieves_similar_experience(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    record = ExperienceRecord(
        signature=signature,
        platform="musa",
        kernel_backend="musa",
        outcome="improved",
        kernel_code="FILE: kernel.mu\n```cpp\n// vectorized load\n```",
        verified=True,
        action="Vectorize contiguous loads",
        bottleneck="memory",
        improvement_pct=18.5,
    )
    assert store.add(record)
    assert not store.add(record)

    matches = store.search(
        build_operator_signature(_problem((2048, 4096))),
        platform="musa",
        kernel_backend="musa",
    )
    assert len(matches) == 1
    assert matches[0].action == "Vectorize contiguous loads"


def test_store_separates_platforms_and_formats_failures(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    store.add(
        ExperienceRecord(
            signature=signature,
            platform="cuda",
            kernel_backend="triton",
            outcome="success",
            kernel_code="def kernel_function(): pass",
            verified=True,
        )
    )
    failure = ExperienceRecord(
        signature=signature,
        platform="musa",
        kernel_backend="musa",
        outcome="failed",
        kernel_code="bad source",
        verified=False,
        action="Use unsupported intrinsic",
        error_message="compiler rejected intrinsic",
    )
    store.add(failure)

    matches = store.search(signature, "musa", "musa", include_failures=True)
    assert [item.id for item in matches] == [failure.id]
    context = format_experience_context(matches)
    assert "CORRECTNESS FAILED" in context
    assert "compiler rejected intrinsic" in context
    assert "bad source" not in context


def test_verified_seed_library_requires_compatible_verified_source(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    success = ExperienceRecord(
        signature=signature,
        platform="musa",
        kernel_backend="musa",
        outcome="improved",
        kernel_code="FILE: kernel.mu\n```cpp\n// verified seed\n```",
        verified=True,
        kernel_time_ms=0.1,
    )
    store.add(success)
    store.add(
        ExperienceRecord(
            signature=signature,
            platform="musa",
            kernel_backend="musa",
            outcome="failed",
            kernel_code="bad source",
            verified=False,
        )
    )

    seeds = store.search_verified_seeds(signature, "musa", "musa")
    assert [seed.id for seed in seeds] == [success.id]
    context = format_verified_seed_context(seeds)
    assert "verified seed" in context
    assert "bad source" not in context


def test_signature_extracts_symbolic_shapes_from_prose_and_code():
    signature = build_operator_signature(
        """Generate a fused matmul kernel for this problem:

import torch
batch_size = 1024
in_features = 4096
out_features = 2058

def get_inputs():
    return [torch.randn(batch_size, in_features, dtype=torch.bfloat16)]

def get_init_inputs():
    return [in_features, out_features]
"""
    )
    assert signature.semantic_type == "matmul"
    assert signature.dtypes == ("bfloat16",)
    assert (1024, 4096) in signature.shapes
    assert (4096, 2058) in signature.shapes


def test_signature_classifies_sigmoid():
    signature = build_operator_signature(
        "def forward(x):\n    return torch.sigmoid(x)\n"
    )
    assert signature.semantic_type == "sigmoid"


def test_target_focus_does_not_invent_a_fused_family():
    description = """A CNN has conv1 and conv2 layers.
Only this block should be generated and optimized:
    x = F.relu(self.fc1(x))
Operator definition: fc1 = nn.Linear(1568, 128)
import torch
import torch.nn as nn
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(1568, 128)
    def forward(self, x):
        return F.relu(self.fc1(x))
def get_inputs():
    return [torch.randn(32, 1568, dtype=torch.float32)]
"""
    generated = build_operator_signature(description)
    optimized = build_operator_signature(
        "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n        self.fc1 = nn.Linear(1568, 128)\n"
        "    def forward(self, x):\n        return F.relu(self.fc1(x))\n"
        "def get_inputs():\n"
        "    return [torch.randn(32, 1568, dtype=torch.float32)]\n"
    )
    assert generated.semantic_type == optimized.semantic_type == "unknown"
    assert generated.shapes == optimized.shapes == ((32, 1568), (1568, 128))
    assert build_operator_signature(description.replace(
        "Only this block should be generated and optimized:\n"
        "    x = F.relu(self.fc1(x))\n", ""
    )).semantic_type == "convolution"


def test_tinycnn_inline_target_is_not_classified_as_weight_transpose():
    description = """Generate a fused operator for TinyCNN.
class TinyCNN:
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.fc1(x))
        return self.fc2(x)
Only this part: x = F.relu(self.fc1(x))
That is Linear(1568, 128) + ReLU with backward.
class LinearReLUFunction:
    def forward(self, x, weight, bias):
        pre_activation = torch.matmul(x, weight.transpose(0, 1)) + bias
        return torch.relu(pre_activation)
"""
    signature = build_operator_signature(description)
    assert signature.semantic_type == "unknown"
    assert signature.components == ()


def test_tinycnn_generate_and_optimize_block_target_ignores_surrounding_model():
    description = """Generate and optimize a fused Linear+ReLU operator.
class TinyCNN:
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.fc1(x))
        return self.fc2(x)
Only generate and optimize this block:
    x = F.relu(self.fc1(x))
class ModelFunction:
    def forward(self, x, weight, bias):
        pre_act = torch.matmul(x, weight.transpose(0, 1)) + bias
        return torch.relu(pre_act)
"""
    signature = build_operator_signature(description)
    assert signature.semantic_type == "unknown"
    assert signature.components == ()


def test_bound_matmul_relu_is_not_manually_equated_with_linear_relu():
    bound = """import torch
class LinearReLUFunction:
    def forward(self, x, weight, bias):
        pre_activation = torch.matmul(x, weight.transpose(0, 1)) + bias
        output = torch.relu(pre_activation)
        return output
    def backward(self, grad_output):
        grad_weight = torch.matmul(grad_output.transpose(0, 1), x)
        return grad_weight
"""
    signature = build_operator_signature(bound)
    assert signature.semantic_type != "linear_relu"
    assert "matmul_relu" in signature.components


def test_unconnected_matmul_and_relu_do_not_create_fused_signature():
    signature = build_operator_signature(
        "def forward(self, x, y, weight):\n"
        "    result = torch.matmul(x, weight.transpose(0, 1))\n"
        "    return result + torch.relu(y)\n"
    )
    assert signature.semantic_type != "linear_relu"
    assert "matmul_relu" not in signature.components


def test_prose_label_alone_does_not_establish_fused_equivalence(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    generated = build_operator_signature(
        "TinyCNN has conv and pool layers.\n"
        "Only this part: x = F.relu(self.fc1(x))\n"
        "Linear(1568, 128) with backward.\n"
    )
    optimized = build_operator_signature(
        "import torch\n"
        "class LinearReLUFunction:\n"
        "    def forward(self, x, weight, bias):\n"
        "        pre = torch.matmul(x, weight.transpose(0, 1)) + bias\n"
        "        return torch.relu(pre)\n"
    )
    record = ExperienceRecord(
        signature=generated, platform="musa", kernel_backend="musa",
        outcome="improved", kernel_code="verified source", verified=True,
        lesson="The local Linear+ReLU subgraph can share a fused launch",
        baseline_time_ms=0.2, kernel_time_ms=0.1, improvement_pct=50,
    )
    store.add(record)
    matches = store.retrieve(optimized, "musa", "musa", stage="optimization")
    assert matches == []
    # A related case is useful context, but a different full contract is not a code seed.
    assert store.search_verified_seeds(optimized, "musa", "musa") == []


def test_regressed_verified_experience_is_not_a_reusable_seed(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    store.add(ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        outcome="success", kernel_code="slower source", verified=True,
        baseline_time_ms=1.0, kernel_time_ms=1.1, improvement_pct=-10,
    ))
    assert store.search_verified_seeds(signature, "musa", "musa") == []
    assert store.search(signature, "musa", "musa", include_failures=False) == []
    assert "MEDIAN-ONLY OBSERVATION" in format_experience_context(
        store.search(signature, "musa", "musa")
    )


def test_search_keeps_older_relevant_cases_after_database_grows(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    relevant = ExperienceRecord(
        signature=build_operator_signature(_problem()),
        platform="musa", kernel_backend="musa", outcome="improved",
        kernel_code="relevant", verified=True, improvement_pct=12,
    )
    store.add(relevant)
    unrelated = build_operator_signature("def forward(x): return torch.sigmoid(x)")
    for index in range(205):
        store.add(ExperienceRecord(
            signature=unrelated, platform="musa", kernel_backend="musa",
            outcome="failed", kernel_code=f"unrelated {index}", verified=False,
        ))
    assert store.search(relevant.signature, "musa", "musa")[0].id == relevant.id


def test_untyped_reflections_remain_cases_and_cannot_become_principles(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    for index in range(2):
        store.add(ExperienceRecord(
            signature=signature, platform="musa", kernel_backend="musa",
            outcome="improved", kernel_code=f"kernel {index}", verified=True,
            lesson="Vectorize contiguous loads", improvement_pct=10 + index,
            config_changes={"BLOCK_SIZE": "128→256"},
        ))
    store.add(ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        outcome="generated", kernel_code="unmeasured", verified=True,
        lesson="Unmeasured advice",
    ))
    with sqlite3.connect(store.path) as connection:
        lessons = connection.execute(
            "SELECT text, claim_type FROM experiment_claims"
        ).fetchall()
    assert set(lessons) == {("Vectorize contiguous loads", "unclassified"),
                            ("Unmeasured advice", "unclassified")}
    matches = store.search(signature, "musa", "musa", limit=5)
    assert all(item.kind == "case" for item in matches)
    assert "Config changes" in format_experience_context(matches)


def test_verified_seed_rejects_rank_only_shape_match(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    store.add(ExperienceRecord(
        signature=build_operator_signature(_problem()),
        platform="musa", kernel_backend="musa", outcome="improved",
        kernel_code="verified", verified=True,
    ))
    different_shape = build_operator_signature(_problem((2048, 4096)))
    assert store.search_verified_seeds(different_shape, "musa", "musa") == []


def test_general_scope_alone_cannot_transfer_an_unvalidated_reflection(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    store.add(ExperienceRecord(
        signature=build_operator_signature(_problem()),
        platform="musa", kernel_backend="musa", outcome="improved",
        kernel_code="rmsnorm source", verified=True,
        lesson="Avoid redundant global memory round trips",
        improvement_pct=8, scope="general",
    ))
    sigmoid = build_operator_signature("def forward(x): return torch.sigmoid(x)")
    matches = store.search(sigmoid, "musa", "musa")
    assert matches == []
    assert store.search_verified_seeds(sigmoid, "musa", "musa") == []


def test_seed_hardware_and_layout_constraints_apply_when_known(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    signature = build_operator_signature(_problem())
    store.add(ExperienceRecord(
        signature=signature, platform="musa", kernel_backend="musa",
        outcome="improved", kernel_code="device-specific source", verified=True,
        hardware="GPU-A", layout="contiguous",
    ))
    assert store.search_verified_seeds(
        signature, "musa", "musa", hardware="GPU-B"
    ) == []
    assert store.search_verified_seeds(
        signature, "musa", "musa", layout="strided"
    ) == []
    assert len(store.search_verified_seeds(
        signature, "musa", "musa", hardware="GPU-A", layout="contiguous"
    )) == 1


def test_old_database_is_rejected_without_removing_rows(tmp_path):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE experiences (
            id TEXT PRIMARY KEY, signature_digest TEXT NOT NULL,
            semantic_type TEXT NOT NULL, platform TEXT NOT NULL,
            kernel_backend TEXT NOT NULL, outcome TEXT NOT NULL,
            verified INTEGER NOT NULL, source_hash TEXT NOT NULL,
            signature_json TEXT NOT NULL, kernel_code TEXT NOT NULL,
            action TEXT NOT NULL, bottleneck TEXT NOT NULL,
            lesson TEXT NOT NULL, error_message TEXT NOT NULL,
            baseline_time_ms REAL, kernel_time_ms REAL, improvement_pct REAL,
            profiler_metrics_json TEXT NOT NULL, environment_json TEXT NOT NULL,
            created_at TEXT NOT NULL)
        """)
        connection.execute("""INSERT INTO experiences VALUES (
            'old', 'digest', 'rmsnorm', 'musa', 'musa', 'improved', 1,
            'hash', '{"semantic_type":"rmsnorm","digest":"digest"}',
            'old kernel', '', '', '', '', 1.0, 0.8, 20.0, '{}', '{}',
            '2025-01-01T00:00:00+00:00')""")
    with pytest.raises(ValueError, match="Only experience schema 9"):
        SQLiteExperienceStore(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT kernel_code FROM experiences").fetchone()[0] == "old kernel"
        assert len(connection.execute("PRAGMA table_info(experiences)").fetchall()) == 20


def test_explicit_dataflow_component_can_supply_text_but_not_full_seed(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    projection = build_operator_signature(
        "def forward(self, x): return F.relu(torch.matmul(x, self.weight))"
    )
    store.add(ExperienceRecord(
        signature=projection, platform="musa", kernel_backend="musa",
        outcome="improved", kernel_code="projection-only source", verified=True,
        lesson="Fuse activation into the linear output epilogue",
        improvement_pct=18,
    ))
    composite = build_operator_signature(
        "class Model:\n"
        "    def forward(self, x):\n"
        "        x = self.conv(x)\n"
        "        return torch.softmax(F.relu(torch.matmul(x, self.weight)), dim=-1)\n"
    )
    assert composite.semantic_type == "softmax"
    assert "matmul_relu" in composite.components
    assert "convolution_matmul" in composite.components
    assert "linear_relu" not in composite.components
    matches = store.search(composite, "musa", "musa")
    assert {item.kind for item in matches} == {"case"}
    assert all(item.retrieval_match == "substructure" for item in matches)
    assert "local substructure only" in format_experience_context(matches)
    assert store.search_verified_seeds(composite, "musa", "musa") == []


def test_unrelated_linear_and_relu_do_not_match_fused_experience(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    store.add(ExperienceRecord(
        signature=build_operator_signature("return F.relu(self.fc1(x))"),
        platform="musa", kernel_backend="musa", outcome="improved",
        kernel_code="fused", verified=True,
    ))
    unrelated = build_operator_signature(
        "def forward(self, x, y):\n"
        "    a = self.conv(x)\n"
        "    b = self.fc1(y)\n"
        "    return a + F.relu(x) + b\n"
    )
    assert unrelated.components == ()
    assert store.search(unrelated, "musa", "musa") == []


def test_opaque_module_names_do_not_invent_linear_substructures():
    signature = build_operator_signature(
        "def forward(self, x):\n"
        "    x = self.conv(x)\n"
        "    x = self.fc1(x)\n"
        "    x = F.relu(x)\n"
        "    return x\n"
    )
    assert signature.semantic_type == "convolution"
    assert "linear_relu" not in signature.components
    assert signature.components == ()

    not_fused = build_operator_signature(
        "def forward(self, x):\n"
        "    x = self.conv(x)\n"
        "    x = self.fc1(x)\n"
        "    x = self.norm(x)\n"
        "    return F.relu(x)\n"
    )
    assert "linear_relu" not in not_fused.components


def test_component_experience_survives_many_exact_family_records(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    fused = ExperienceRecord(
        signature=build_operator_signature("return F.relu(torch.matmul(x, self.weight))"),
        platform="musa", kernel_backend="musa", outcome="improved",
        kernel_code="fused", verified=True,
    )
    store.add(fused)
    composite = build_operator_signature(
        "def forward(self, x): return torch.softmax(F.relu(torch.matmul(self.conv(x), self.weight)), dim=-1)"
    )
    for index in range(20):
        store.add(ExperienceRecord(
            signature=composite, platform="musa", kernel_backend="musa",
            outcome="improved", kernel_code=f"conv-{index}", verified=True,
        ))
    assert fused.id in {item.id for item in store.search(composite, "musa", "musa")}


def test_generic_connected_motif_matches_larger_operator_not_unrelated_calls(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    conv_relu = build_operator_signature(
        "def forward(self, x): return F.relu(self.conv(x))"
    )
    record = ExperienceRecord(
        signature=conv_relu, platform="musa", kernel_backend="musa",
        outcome="improved", kernel_code="conv-relu source", verified=True,
    )
    store.add(record)
    larger = build_operator_signature(
        "def forward(self, x):\n"
        "    x = F.relu(self.conv(x))\n"
        "    return self.attention(x)\n"
    )
    assert larger.semantic_type == "attention"
    assert "convolution_relu" in larger.components
    matches = store.search(larger, "musa", "musa")
    assert matches[0].id == record.id
    assert matches[0].retrieval_match == "substructure"
    assert store.search_verified_seeds(larger, "musa", "musa") == []

    unrelated = build_operator_signature(
        "def forward(self, x, y):\n"
        "    a = self.conv(x)\n"
        "    b = F.relu(y)\n"
        "    return self.attention(a + b)\n"
    )
    assert "convolution_relu" not in unrelated.components
    assert store.search(unrelated, "musa", "musa") == []


def test_seed_requires_same_problem_fingerprint(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    original = build_operator_signature(
        "def forward(self, x): return F.relu(self.fc1(x))\n"
        "# shape 32x1568"
    )
    store.add(ExperienceRecord(
        signature=original, platform="musa", kernel_backend="musa",
        outcome="generated", kernel_code="verified source", verified=True,
    ))
    different_contract = build_operator_signature(
        "def forward(self, x): return F.relu(self.fc1(x))\n"
        "# shape 64x1568"
    )
    assert store.search_verified_seeds(different_contract, "musa", "musa") == []


def test_seed_rejects_same_keywords_with_different_computation(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    original = build_operator_signature(
        "def forward(self, x): return torch.sigmoid(x + 1)\n"
        "def get_inputs(): return [torch.randn(32, 64)]"
    )
    changed = build_operator_signature(
        "def forward(self, x): return torch.sigmoid(x - 1)\n"
        "def get_inputs(): return [torch.randn(32, 64)]"
    )
    assert original.digest == changed.digest
    assert original.contract_digest != changed.contract_digest
    store.add(ExperienceRecord(
        signature=original, platform="musa", kernel_backend="musa",
        outcome="generated", kernel_code="source", verified=True,
    ))
    assert store.search_verified_seeds(changed, "musa", "musa") == []
