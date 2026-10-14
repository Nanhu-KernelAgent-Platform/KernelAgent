"""General graph identity checks; no operator-name matching table is involved."""

from triton_kernel_agent.experience import (
    ExperienceRecord, SQLiteExperienceStore, build_operator_signature,
    reference_source_from_test,
)
from types import SimpleNamespace
from dataclasses import replace
import pytest
import copy
import math
from triton_kernel_agent.experience.graph_identity import canonical_nodes, graph_digest, same_graph

from triton_kernel_agent.agent import TritonKernelAgent


def _scalar_graph(value, dtype='torch.float32', operator='aten::rsub.Scalar'):
    nodes = [{'op': 'input', 'args': [], 'kwargs': {}},
             {'op': operator, 'args': [{'node': 0}, value], 'kwargs': {}},
             {'op': 'output', 'args': [{'node': 1}], 'kwargs': {}}]
    specs = {str(i): {'dtype': dtype, 'shape': [8], 'stride': [1]} for i in (0, 1)}
    return {'graph_digest': graph_digest(nodes), 'graph_nodes': nodes, 'graph_tensor_specs': specs}


@pytest.mark.parametrize('operator', ['aten::rsub.Scalar', 'aten::mul.Scalar', 'aten::pow.Tensor_Scalar'])
@pytest.mark.parametrize('dtype', ['torch.float16', 'torch.bfloat16', 'torch.float32', 'torch.float64'])
def test_exact_scalar_representation_is_schema_and_dtype_driven(operator, dtype):
    integer = _scalar_graph(1, dtype, operator)
    floating = _scalar_graph(1.0, dtype, operator)
    assert integer['graph_digest'] != floating['graph_digest']
    untouched = copy.deepcopy(floating)
    assert same_graph(integer, floating)
    assert floating == untouched  # Historical evidence is not rewritten.
    normalized = canonical_nodes(floating['graph_nodes'], floating['graph_tensor_specs'])
    assert type(normalized[1]['args'][1]) is int


@pytest.mark.parametrize('dtype', ['torch.int32', 'torch.int64', 'torch.bool', 'torch.complex64', None])
def test_scalar_types_are_preserved_for_integer_complex_and_unknown_tensors(dtype):
    assert not same_graph(_scalar_graph(1, dtype), _scalar_graph(1.0, dtype))


@pytest.mark.parametrize('value', [True, -0.0, 1.5, float('inf'), float('nan'), float(2**54)])
def test_boolean_signed_zero_nonfinite_fractional_and_large_literals_are_not_folded(value):
    graph = _scalar_graph(value)
    normalized = canonical_nodes(graph['graph_nodes'], graph['graph_tensor_specs'])[1]['args'][1]
    assert type(normalized) is type(value)
    if isinstance(value, float) and math.isnan(value):
        assert math.isnan(normalized)
    else:
        assert normalized == value
    if value == 0:
        assert math.copysign(1, normalized) == math.copysign(1, value)


def test_non_scalar_dimension_and_dtype_arguments_keep_their_types():
    graph = _scalar_graph(1.0, operator='aten::select.int')
    assert canonical_nodes(graph['graph_nodes'], graph['graph_tensor_specs'])[1]['args'][1] == 1.0
    assert type(canonical_nodes(graph['graph_nodes'], graph['graph_tensor_specs'])[1]['args'][1]) is float
    assert not same_graph(_scalar_graph(1, operator='aten::sub.Tensor'),
                          _scalar_graph(1.0, operator='aten::sub.Tensor'))
    graph['graph_nodes'][1] = {'op': 'aten::sum.dim_IntList', 'args': [{'node': 0}, [1.0], True],
                               'kwargs': {'dtype': 6.0}}
    normalized = canonical_nodes(graph['graph_nodes'], graph['graph_tensor_specs'])[1]
    assert type(normalized['args'][1][0]) is float
    assert normalized['args'][2] is True
    assert type(normalized['kwargs']['dtype']) is float
    graph['graph_nodes'][1] = {'op': 'aten::add.Tensor', 'args': [{'node': 0}, {'node': 0}],
                               'kwargs': {'alpha': 1.0}}
    assert canonical_nodes(graph['graph_nodes'], graph['graph_tensor_specs'])[1]['kwargs']['alpha'] == 1


def test_scalar_requires_exact_dtype_representation_and_complete_tensor_metadata():
    graph = _scalar_graph(16777217.0)
    assert type(canonical_nodes(graph['graph_nodes'], graph['graph_tensor_specs'])[1]['args'][1]) is float
    graph = _scalar_graph(1.0)
    graph['graph_tensor_specs']['0'] = [{'dtype': 'torch.float32'}]
    assert type(canonical_nodes(graph['graph_nodes'], graph['graph_tensor_specs'])[1]['args'][1]) is float
    graph['graph_tensor_specs']['0'] = {'dtype': 'torch.float16'}
    assert type(canonical_nodes(graph['graph_nodes'], graph['graph_tensor_specs'])[1]['args'][1]) is float
    assert not same_graph(_scalar_graph(1), _scalar_graph(2.0))


def test_real_trace_integral_scalars_share_identity_but_changed_formula_does_not():
    def source(value):
        return ('import torch\nclass Model(torch.nn.Module):\n'
                f'    def forward(self, x): return torch.sin(x) * ({value} - x)\n'
                'def get_inputs(): return [torch.randn(2, 4, requires_grad=True)]\n')
    integer = build_operator_signature(source('1'), trace_graph=True)
    floating = build_operator_signature(source('1.0'), trace_graph=True)
    changed = build_operator_signature(source('2.0'), trace_graph=True)
    assert integer.graph_digest and integer.graph_digest == floating.graph_digest
    assert integer.interface == floating.interface
    assert integer.graph_digest != changed.graph_digest


def test_generation_seed_matches_derived_historical_scalar_identity_without_mutating_db(tmp_path):
    store = SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    old = _scalar_graph(1.0)
    new = _scalar_graph(1)
    from triton_kernel_agent.experience import OperatorSignature
    def signature(graph, family):
        return OperatorSignature(family, graph_version='torch_export_aten_v1',
            interface={'inputs': [{'dtype': 'torch.float32', 'shape': [8], 'stride': [1]}]},
            **graph)
    record = ExperienceRecord(signature(old, 'old-family'), 'musa', 'musa', 'generated', 'source', True,
                              direction='forward')
    store.add(record)
    target = signature(new, 'unknown')
    hits = store.search_verified_seeds(target, 'musa', 'musa', direction='forward')
    assert [r.id for r in hits] == [record.id]
    assert hits[0].candidate_match == 'graph_exact'
    with store._connect() as connection:
        raw = connection.execute('SELECT environment_json FROM experiment_runs').fetchone()[0]
        assert old['graph_digest'] in raw


def _source(expression: str, *, dtype: str = "float32", batch: int = 2) -> str:
    return (
        "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(8, 4)\n"
        "    def forward(self, x):\n"
        f"        return {expression}\n"
        "def get_inputs():\n"
        f"    return [torch.randn({batch}, 8, dtype=torch.{dtype})]\n"
    )


def test_describe2_graph_matches_plain_reference_and_ignores_surrounding_model():
    source = _source("F.relu(self.fc(x))")
    description = (
        "## Describe1\nA full CNN contains convolutions and pooling.\n"
        "Target operator only: x = F.relu(self.fc(x))\n"
        "## Describe2 (reference Model)\n" + source
    )
    generated = build_operator_signature(description, trace_graph=True)
    optimized = build_operator_signature(
        _source("torch.relu(self.fc(x))"), trace_graph=True,
    )
    assert generated.semantic_type == optimized.semantic_type == "unknown"
    assert generated.graph_version == optimized.graph_version == "torch_export_aten_v1"
    assert generated.graph_digest == optimized.graph_digest
    assert generated.interface == optimized.interface
    assert generated.interface["outputs"][0]["shape"] == [2, 4]


@pytest.mark.parametrize('body', [
    'return F.relu(self.fc1(x))',
    'return F.relu(self.linear_projection(x))',
    'y = self.fc1(x)\n    return F.relu(y)',
    'return torch.relu(F.linear(x, weight, bias))',
    'return torch.relu(torch.matmul(x, weight.T) + bias)',
])
def test_no_handwritten_linear_relu_family_or_module_name_inference(body):
    signature = build_operator_signature('def forward(self, x, weight, bias):\n    '+body+'\n')
    assert signature.semantic_type != 'linear_relu'
    assert 'linear_relu' not in signature.components
    assert not any(component.startswith('linear_') for component in signature.components)


def test_legacy_fused_label_recalled_by_real_graph_not_module_name(tmp_path):
    original = build_operator_signature(_source('F.relu(self.fc(x))'), trace_graph=True)
    renamed = build_operator_signature(_source('F.relu(self.fc(x))').replace('self.fc', 'self.projection'), trace_graph=True)
    assert original.semantic_type == renamed.semantic_type == 'unknown'
    assert original.graph_digest and original.graph_digest == renamed.graph_digest
    assert original.interface == renamed.interface
    store = SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    record = ExperienceRecord(signature=replace(original, semantic_type='linear_relu'),
        platform='musa', kernel_backend='musa', direction='forward', hardware='device-A',
        outcome='improved', verified=True, kernel_code='historical source', kernel_time_ms=1,
        lesson='Check dispatch and allocation overhead before changing vector width.')
    store.add(record)
    hits = store.retrieve(renamed, 'musa', 'musa', direction='forward', hardware='device-A')
    assert [hit.id for hit in hits] == [record.id]
    assert hits[0].retrieval_reason == 'full_graph_contract_text_only'
    assert not hits[0].kernel_code
    misleading = _source('F.relu(self.fc(x))').replace('nn.Linear(8, 4)', 'nn.Sigmoid()')
    other = build_operator_signature(misleading, trace_graph=True)
    assert other.graph_digest != original.graph_digest
    assert store.retrieve(other, 'musa', 'musa', direction='forward', hardware='device-A') == []
    with store._connect() as connection:
        assert connection.execute('select semantic_type from experience_cases where id=?', (record.id,)).fetchone()[0] == 'linear_relu'


def test_untraced_opaque_formula_does_not_become_a_code_candidate(tmp_path):
    signature = build_operator_signature(
        'def get_inputs(): return [torch.randn(4, 8)]\n'
        'def forward(x, weight): return torch.relu(x @ weight.T)\n')
    assert signature.semantic_type == 'unknown' and not signature.graph_digest
    store = SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    store.add_online(ExperienceRecord(signature=signature, platform='musa', kernel_backend='musa',
        direction='forward', outcome='success', verified=True,
        kernel_code='historical source', kernel_time_ms=1), session_id='old')
    assert store.search_measured_implementations(signature, 'musa', 'musa', direction='forward') == []


def test_graph_recall_requires_same_structure_and_runtime_interface(tmp_path):
    store = SQLiteExperienceStore(tmp_path / "experiences.sqlite3")
    original = build_operator_signature(
        _source("F.relu(self.fc(x))"), trace_graph=True,
    )
    same = build_operator_signature(
        _source("torch.relu(self.fc(x))"), trace_graph=True,
    )
    different_op = build_operator_signature(
        _source("torch.sigmoid(self.fc(x))"), trace_graph=True,
    )
    different_shape = build_operator_signature(
        _source("torch.relu(self.fc(x))", batch=3), trace_graph=True,
    )
    record = ExperienceRecord(
        signature=original, platform="musa", kernel_backend="musa",
        direction="forward", outcome="improved", verified=True,
        kernel_code="previously verified source", baseline_time_ms=0.2,
        kernel_time_ms=0.1, hardware="MTT S5000",
    )
    store.add_online(record, session_id="first")
    hits = store.search_measured_implementations(
        same, "musa", "musa", direction="forward", hardware="MTT S5000",
    )
    assert len(hits) == 1
    assert hits[0].candidate_match == "graph_exact"
    assert hits[0].candidate_graph_match is True
    assert hits[0].candidate_text_match is False
    assert store.search_measured_implementations(
        different_op, "musa", "musa", direction="forward",
        hardware="MTT S5000",
    ) == []
    assert store.search_measured_implementations(
        different_shape, "musa", "musa", direction="forward",
        hardware="MTT S5000",
    ) == []
    assert store.search_measured_implementations(
        same, "musa", "musa", direction="backward", hardware="MTT S5000",
    ) == []
    store.add_online(ExperienceRecord(
        signature=same, platform="musa", kernel_backend="musa",
        direction="forward", outcome="improved", verified=True,
        kernel_code="second measured source", baseline_time_ms=0.2,
        kernel_time_ms=0.09, hardware="MTT S5000",
    ), session_id="second")
    with store._connect() as connection:
        assert connection.execute(
            "SELECT COUNT(DISTINCT workload_id) FROM experiment_runs"
        ).fetchone()[0] == 1


def test_untraceable_reference_keeps_legacy_identity_without_graph():
    signature = build_operator_signature(
        "A custom operator: only Python prose, no Model", trace_graph=True,
    )
    assert signature.graph_digest == ""
    assert signature.interface == {}


def test_multi_operator_graph_is_independent_of_python_temporary_names():
    prefix = "import torch\nclass Model(torch.nn.Module):\n"
    suffix = "\ndef get_inputs(): return [torch.randn(3, 5)]\n"
    inline = prefix + (
        "    def forward(self, x):\n"
        "        return torch.sigmoid(torch.sin(x) + 1)\n"
    ) + suffix
    staged = prefix + (
        "    def forward(self, value):\n"
        "        intermediate = torch.sin(value)\n"
        "        return torch.sigmoid(intermediate + 1)\n"
    ) + suffix
    changed = inline.replace(" + 1)", " + 2)")
    first = build_operator_signature(inline, trace_graph=True)
    equivalent = build_operator_signature(staged, trace_graph=True)
    different = build_operator_signature(changed, trace_graph=True)
    assert first.graph_digest == equivalent.graph_digest
    assert first.interface == equivalent.interface
    assert first.graph_digest != different.graph_digest


def test_verified_test_reference_excludes_kernel_import_and_test_harness():
    test_code = (
        "import torch\nfrom kernel import kernel_function\n"
        'KERNELAGENT_OPTIMIZATION_TARGET = "forward"\n'
        "class Model(torch.nn.Module):\n"
        "    def forward(self, x): return x * torch.sigmoid(x)\n"
        "def get_inputs(): return [torch.randn(2, 4)]\n"
        "def fail(): raise RuntimeError('not part of reference')\n"
        "if __name__ == '__main__': fail()\n"
    )
    source = reference_source_from_test(test_code)
    assert source is not None
    assert "from kernel import" not in source
    assert "def fail" not in source
    assert build_operator_signature(source, trace_graph=True).graph_digest


def test_generation_experience_uses_verified_test_reference(tmp_path):
    test_code = (
        "import torch\nfrom kernel import kernel_function\n"
        'KERNELAGENT_OPTIMIZATION_TARGET = "forward"\n'
        "class Model(torch.nn.Module):\n"
        "    def forward(self, x): return x * torch.sigmoid(x)\n"
        "def get_inputs(): return [torch.randn(2, 4)]\n"
    )
    request = (
        "## Describe2\nimport torch\nclass Model:\n"
        "    def forward(self, x): return x / (1 + torch.exp(-x))\n"
        "## Describe2\ninvalid duplicate section\n"
    )
    recorded = []
    seen_directions = []
    agent = TritonKernelAgent.__new__(TritonKernelAgent)
    agent.logger = SimpleNamespace(info=lambda *_args: None,
                                   warning=lambda *_args: None)
    agent.log_dir = tmp_path
    agent.manager = SimpleNamespace(run_verification=lambda **_kwargs: {
        "success": True, "kernel_code": "verified source", "worker_id": 0,
        "rounds": 1,
    })
    agent._generate_kernel_seeds = lambda *_args, **kwargs: (
        seen_directions.append(kwargs["execution_direction"]) or ["verified source"])
    agent._kernel_backend = SimpleNamespace(name="musa")
    agent._platform_config = SimpleNamespace(name="musa")
    agent.model_name = "test-model"
    agent.experience_store = SimpleNamespace(add_online=lambda record, **_kwargs:
                                             recorded.append(record))
    agent.generate_kernel(request, test_code=test_code, generate_default_test=False)
    assert len(recorded) == 1
    result = recorded[0]
    assert result.signature.graph_digest == build_operator_signature(
        reference_source_from_test(test_code), trace_graph=True,
    ).graph_digest
    assert result.direction == "forward"
    assert result.environment["reference_source_kind"] == "verified_test_0"
    assert result.environment["reference_source_digest"]
    assert result.environment["request_digest"]
    assert result.environment["request_signature"]["contract_digest"] == (
        build_operator_signature(request).contract_digest
    )
    agent.generate_kernel(request, test_code=test_code, generate_default_test=False,
                          execution_direction="joint")
    assert recorded[-1].direction == "joint"
    assert recorded[-1].environment["execution_direction_source"] == "caller"
    without_marker = test_code.replace('KERNELAGENT_OPTIMIZATION_TARGET = "forward"\n', '')
    agent.generate_kernel(request, test_code=without_marker,
                          generate_default_test=False,
                          execution_direction="backward")
    assert recorded[-1].direction == "backward"
    assert seen_directions == ["forward", "joint", "backward"]
