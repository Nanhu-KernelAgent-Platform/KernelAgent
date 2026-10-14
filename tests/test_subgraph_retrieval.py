from dataclasses import replace
import json

import pytest

from triton_kernel_agent.experience import ExperienceRecord, SQLiteExperienceStore, build_operator_signature
from triton_kernel_agent.experience.models import OperatorSignature
from triton_kernel_agent.experience.store import format_experience_context
from triton_kernel_agent.experience.subgraph import contained_region


def graph(ops, *, family='unclassified', stride=(8, 1)):
    nodes = [{'op': 'input', 'args': [], 'kwargs': {}}]
    for op, args in ops:
        nodes.append({'op': op, 'args': args, 'kwargs': {}})
    nodes.append({'op': 'output', 'args': [{'node': len(nodes)-1}], 'kwargs': {}})
    return OperatorSignature(family, graph_version='torch_export_aten_v1',
        graph_digest=json.dumps(nodes), graph_nodes=tuple(nodes),
        graph_tensor_specs={str(i): {'shape': [4, 8], 'dtype': 'torch.float32', 'stride': list(stride)}
                            for i in range(len(nodes)-1)})


def ref(i):
    return {'node': i}


@pytest.mark.parametrize('ops', [
    [('aten::sin.', [ref(0)]), ('aten::cos.', [ref(1)])],
    [('aten::mul.Tensor', [ref(0), ref(0)]), ('aten::add.Tensor', [ref(1), 0.1])],
    [('aten::sum.dim_IntList', [ref(0), [1], True]), ('aten::sigmoid.', [ref(1)])],
])
def test_generic_connected_regions_and_intermediate_boundaries(ops):
    source = graph(ops)
    # Remap every source operand after prepending a producer.
    def shift(v):
        if isinstance(v, dict): return {'node': v['node']+1}
        if isinstance(v, list): return [shift(x) for x in v]
        return v
    target = graph([('aten::neg.', [ref(0)])] + [(op, shift(args)) for op, args in ops]
                   + [('aten::tanh.', [ref(3)])])
    hit = contained_region(source, target)
    assert hit['source_nodes'] == [1, 2]
    assert hit['target_nodes'] == [2, 3]
    assert hit['boundary_mapping'] == {'0': 1}


def test_repeated_nodes_literals_connectivity_and_layout():
    source = graph([('aten::sin.', [ref(0)]), ('aten::sin.', [ref(1)])])
    good = graph([('aten::sin.', [ref(0)]), ('aten::sin.', [ref(1)]), ('aten::cos.', [ref(2)])])
    assert contained_region(source, good)
    parallel = graph([('aten::sin.', [ref(0)]), ('aten::sin.', [ref(0)]), ('aten::add.Tensor', [ref(1), ref(2)])])
    assert not contained_region(source, parallel)
    assert not contained_region(source, replace(good, graph_version='other'))
    assert not contained_region(source, replace(good, graph_tensor_specs={}))
    for key, value in [('shape', [8,4]), ('dtype', 'torch.float16')]:
        changed_specs = {k: {**v, key: value} for k,v in good.graph_tensor_specs.items()}
        assert not contained_region(source, replace(good, graph_tensor_specs=changed_specs))
    assert not contained_region(source, graph([('aten::sin.', [ref(0)]), ('aten::sin.', [ref(1)]), ('aten::cos.', [ref(2)])], stride=(16, 2)))
    assert not contained_region(parallel, good)
    literal = graph([('aten::mul.Tensor', [ref(0), 0.1])])
    changed = graph([('aten::mul.Tensor', [ref(0), 0.2]), ('aten::sin.', [ref(1)])])
    assert not contained_region(literal, changed)
    operands = graph([('aten::sub.Tensor', [ref(0), ref(0)])])
    different_boundary = graph([('aten::sin.', [ref(0)]), ('aten::sub.Tensor', [ref(0), ref(1)])])
    assert not contained_region(operands, different_boundary)
    disconnected = graph([('aten::sin.', [ref(0)]), ('aten::cos.', [ref(0)])])
    assert not contained_region(disconnected, graph([('aten::sin.', [ref(0)]), ('aten::cos.', [ref(0)]), ('aten::add.Tensor', [ref(1), ref(2)])]))


def test_store_cross_family_scoped_text_audit_and_seed_isolation(tmp_path):
    store = SQLiteExperienceStore(tmp_path / 'memory.sqlite3')
    source = graph([('aten::sin.', [ref(0)]), ('aten::cos.', [ref(1)])], family='old-family')
    target = graph([('aten::sin.', [ref(0)]), ('aten::cos.', [ref(1)]), ('aten::neg.', [ref(2)])], family='new-family')
    record = ExperienceRecord(source, 'musa', 'musa', 'improved', 'DO NOT INJECT CODE', True,
        direction='forward', lesson='Fuse chained pointwise stages after checking extra consumers.', hardware='device-A')
    store.add(record)
    hits = store.retrieve(target, 'musa', 'musa', direction='forward', hardware='device-A')
    assert len(hits) == 1 and hits[0].id == record.id
    assert hits[0].retrieval_match == 'substructure' and not hits[0].kernel_code
    assert hits[0].graph_match_evidence['hardware_compatibility'] == 'exact'
    unknown = store.retrieve(target, 'musa', 'musa', direction='forward')
    assert unknown[0].graph_match_evidence['hardware_compatibility'] == 'unresolved'
    assert 'Recorded timings describe the original workload' in format_experience_context(hits)
    assert 'DO NOT INJECT CODE' not in format_experience_context(hits)
    assert not store.search_verified_seeds(target, 'musa', 'musa', direction='forward')
    assert not store.search_measured_implementations(target, 'musa', 'musa', direction='forward')
    assert not store.retrieve(target, 'musa', 'musa', direction='backward')
    assert not store.retrieve(target, 'musa', 'musa', direction='forward', hardware='device-B')
    event = store.log_retrieval(session_id='test', stage='generation', signature=target, records=hits, direction='forward')
    with store._connect() as connection:
        selection = json.loads(connection.execute('select selection_json from experience_retrieval_events where id=?',(event,)).fetchone()[0])
    assert selection[0]['graph_match_evidence']['target_nodes'] == [1,2]
    store.quarantine('case', record.id, 'suspect lesson')
    assert not store.retrieve(target, 'musa', 'musa', direction='forward')


@pytest.mark.parametrize('small,large', [
    ('torch.sin(x).cos()', 'torch.sin(x).cos().neg()'),
    ('x.sum(dim=1, keepdim=True).sigmoid()', 'x.sum(dim=1, keepdim=True).sigmoid().neg()'),
    ('x * x + 0.1 * x', 'torch.tanh(x * x + 0.1 * x)'),
])
def test_actual_export_metadata_and_generic_regions(small, large):
    def source(expr):
        return ('import torch\nclass Model(torch.nn.Module):\n'
                '    def forward(self,x):\n        return '+expr+'\n'
                'def get_inputs():\n    return [torch.randn(4,8)]\n')
    a = build_operator_signature(source(small), trace_graph=True)
    b = build_operator_signature(source(large), trace_graph=True)
    assert a.graph_tensor_specs and b.graph_tensor_specs
    assert contained_region(a,b)


def test_resource_bound_fails_closed():
    source = graph([('aten::sin.', [ref(0)])])
    huge = graph([('aten::sin.', [ref(i)]) for i in range(257)])
    assert not contained_region(source,huge)


def measured_record(signature, candidate_ms, *, reported_outcome='regressed', direction='forward', source='candidate'):
    from triton_kernel_agent.measurement_policy import interleaved_pair
    before, after = interleaved_pair(lambda role: {
        'samples_ms': [1.0 if role == 'parent' else candidate_ms] * 3,
    }, 'parent', 'candidate')
    return ExperienceRecord(signature, 'musa', 'musa', reported_outcome, source, True,
        direction=direction, hardware='device-A',
        action='Change element-to-thread mapping',
        lesson='Measure scheduling overhead before increasing grid parallelism; check all layouts.',
        baseline_time_ms=1.0, kernel_time_ms=candidate_ms,
        baseline_samples_ms=tuple(before['samples_ms']), kernel_samples_ms=tuple(after['samples_ms']),
        comparison=after['comparison'])


@pytest.mark.parametrize('candidate_ms,label', [(.99, 'improved'), (1.01, 'regressed')])
def test_generation_keeps_inconclusive_region_with_positive_or_negative_median(tmp_path, candidate_ms, label):
    store = SQLiteExperienceStore(tmp_path / 'memory.sqlite3')
    a = graph([('aten::sin.', [ref(0)]), ('aten::cos.', [ref(1)])], family='old-family')
    b = graph([('aten::sin.', [ref(0)]), ('aten::cos.', [ref(1)]), ('aten::neg.', [ref(2)])], family='new-family')
    # Abundant older heuristic positives must not consume the caution slot.
    for i in range(5):
        store.add(ExperienceRecord(OperatorSignature('new-family'), 'musa', 'musa', 'improved', f'old-{i}', True,
                  direction='forward', action='Tune launch geometry', lesson=f'Past launch geometry {i}', improvement_pct=10))
    record = measured_record(a, candidate_ms, reported_outcome=label)
    store.add_online(record, session_id='A')
    hits = store.retrieve(b, 'musa', 'musa', direction='forward', hardware='device-A', stage='generation', limit=3)
    hit = next(item for item in hits if item.id == record.id)
    assert hit.text_evidence_status == 'inconclusive'
    assert hit.graph_match_evidence and not hit.kernel_code
    prompt = format_experience_context([hit])
    assert 'INCONCLUSIVE PERFORMANCE' in prompt
    assert 'neither a gain nor a regression is established' in prompt
    assert 'ROBUST PARENT REGRESSION' not in prompt and 'ROBUST PARENT GAIN' not in prompt
    assert 'FAILED/REGRESSED' not in prompt
    assert len([item for item in hits if item.text_evidence_status == 'inconclusive']) == 1
    assert not store.search_verified_seeds(b,'musa','musa',direction='forward')
    assert not store.retrieve(b,'musa','musa',direction='backward',stage='generation')
    event = store.log_retrieval(session_id='B',stage='generation',signature=b,records=hits,direction='forward')
    with store._connect() as c:
        selection = json.loads(c.execute('select selection_json from experience_retrieval_events where id=?',(event,)).fetchone()[0])
    assert next(x for x in selection if x['id']==record.id)['text_evidence_status'] == 'inconclusive'


@pytest.mark.parametrize('candidate_ms,expected',[(.8,'ROBUST PARENT GAIN'),(1.2,'ROBUST PARENT REGRESSION')])
def test_robust_status_is_independent_of_historical_outcome(tmp_path,candidate_ms,expected):
    store = SQLiteExperienceStore(tmp_path / 'memory.sqlite3')
    signature = graph([('aten::sin.', [ref(0)])])
    record = measured_record(signature,candidate_ms,reported_outcome='success')
    store.add_online(record,session_id='A')
    hits = store.retrieve(signature,'musa','musa',direction='forward',stage='generation')
    assert len(hits)==1 and expected in format_experience_context(hits)


@pytest.mark.parametrize('stage', ['generation', 'optimization'])
@pytest.mark.parametrize('limit', [1, 2, 3, 4])
def test_region_caution_survives_exact_task_caution_competition(tmp_path, stage, limit):
    store = SQLiteExperienceStore(tmp_path / 'memory.sqlite3')
    a = graph([('aten::sin.', [ref(0)]), ('aten::cos.', [ref(1)])], family='A')
    b = graph([('aten::sin.', [ref(0)]), ('aten::cos.', [ref(1)]), ('aten::neg.', [ref(2)])], family='B')
    a = replace(a, interface={'inputs': ['complete-contract']})
    b = replace(b, interface={'inputs': ['complete-contract']})
    region = measured_record(a, .99)
    exact = measured_record(b, .99, source='exact')
    store.add_online(region, session_id='A')
    store.add_online(exact, session_id='old-B')
    # Later exact-task warnings cannot erase all matching local-region warnings.
    hits = store.retrieve(b, 'musa', 'musa', stage=stage, direction='forward',
                          hardware='device-A', limit=limit)
    assert [r.id for r in hits] == [region.id]
    assert hits[0].retrieval_match == 'substructure' and not hits[0].kernel_code
    assert hits[0].text_evidence_status == 'inconclusive'
    with store._connect() as c:
        assert c.execute('select count(*) from experience_cases').fetchone()[0] == 2


def test_generation_cautions_are_bounded_directional_quarantinable_and_text_only(tmp_path):
    store = SQLiteExperienceStore(tmp_path / 'memory.sqlite3')
    signature = OperatorSignature('pointwise')
    records=[]
    for i in range(3):
        r=ExperienceRecord(signature,'musa','musa','failed',f'INVALID CODE {i}',False,
                           direction='forward',action='Use unavailable device intrinsic',error_message='Intrinsic is unsupported on this backend.')
        store.add(r);records.append(r)
    hits=store.retrieve(signature,'musa','musa',direction='forward',stage='generation')
    assert len(hits)==1 and not hits[0].kernel_code
    assert 'CORRECTNESS FAILED' in format_experience_context(hits)
    assert 'INVALID CODE' not in format_experience_context(hits)
    assert not store.retrieve(signature,'musa','musa',direction='unknown',stage='generation')
    assert not store.retrieve(signature,'musa','musa',direction='backward',stage='generation')
    for r in records: store.quarantine('case',r.id,'invalid advisory')
    assert not store.retrieve(signature,'musa','musa',direction='forward',stage='generation')


def test_empty_failure_does_not_occupy_generation_warning_slot(tmp_path):
    store = SQLiteExperienceStore(tmp_path / 'memory.sqlite3')
    signature=OperatorSignature('pointwise')
    store.add(ExperienceRecord(signature,'musa','musa','failed','invalid',False,
                               direction='forward',action='Tried a new kernel'))
    assert not store.retrieve(signature,'musa','musa',stage='generation',direction='forward')


def test_cli_preview_exposes_structural_warning_confidence(tmp_path,monkeypatch,capsys):
    from scripts import experience_db
    path=tmp_path/'memory.sqlite3'
    store=SQLiteExperienceStore(path)
    a=graph([('aten::sin.',[ref(0)])],family='small')
    b=graph([('aten::sin.',[ref(0)]),('aten::neg.',[ref(1)])],family='large')
    record=measured_record(a,1.01)
    store.add_online(record,session_id='A')
    problem=tmp_path/'reference.py';problem.write_text('reference fixture',encoding='utf-8')
    def build(source,*,trace_graph=False):
        assert trace_graph and source=='reference fixture'
        return b
    monkeypatch.setattr(experience_db,'build_operator_signature',build)
    experience_db.main(['--db',str(path),'retrieve',str(problem),'--platform','musa',
                       '--kernel-backend','musa','--direction','forward','--trace-graph',
                       '--hardware','device-A'])
    result=json.loads(capsys.readouterr().out)
    assert result['seed_ids']==[]
    assert result['selected'][0]['text_evidence_status']=='inconclusive'
    assert result['selected'][0]['retrieval_reason']=='connected_aten_region_text_only'
    assert result['selected'][0]['graph_match_evidence']['hardware_compatibility']=='exact'
