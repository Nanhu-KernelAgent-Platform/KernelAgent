from dataclasses import replace
import json

import pytest

from triton_kernel_agent.experience import ExperienceRecord, SQLiteExperienceStore
from triton_kernel_agent.experience.models import OperatorSignature
from triton_kernel_agent.experience.store import format_experience_context


def signature(family, *, request='old'):
    return OperatorSignature(family,dtypes=('float32',),shapes=((4,8),),
        digest=request,contract_digest=request,
        graph_version='torch_export_aten_v1',graph_digest='recorded-graph',
        interface={'inputs':[{'shape':[4,8],'dtype':'torch.float32','stride':[8,1]}],
                   'outputs':[{'shape':[4,8],'dtype':'torch.float32','stride':[8,1]}]})


def record(s, **kwargs):
    return ExperienceRecord(s,'musa','musa','improved','SOURCE MUST NOT ENTER TEXT',True,
        direction='forward',hardware='device-A',layout='contiguous',kernel_time_ms=1,
        lesson='Check dispatch and allocation overhead before changing vector width.',**kwargs)


@pytest.mark.parametrize('direction', ['forward', 'backward', 'joint'])
@pytest.mark.parametrize('family', ['unknown', 'unrelated-description'])
def test_complete_graph_seed_ignores_prose_but_preserves_direction_and_device(tmp_path, direction, family):
    store = SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    old = replace(signature('old-description'), interface={
        **signature('old-description').interface, 'direction': direction})
    source = replace(record(old), direction=direction, outcome='generated',
                     environment={'request_signature': signature('original-request').to_dict()})
    store.add(source)
    target = replace(old, semantic_type=family, digest='new-request',
                     contract_digest='new-contract', components=('different-prose-label',))
    def seeds(**kwargs):
        return store.search_verified_seeds(target, 'musa', 'musa', direction=direction,
                                           hardware='device-A', layout='contiguous', **kwargs)
    hits = seeds()
    assert [item.id for item in hits] == [source.id]
    assert hits[0].candidate_match == 'graph_exact'
    assert hits[0].retrieval_reason == 'full_graph_contract_seed'
    event = store.log_retrieval(session_id='test', stage='generation', signature=target,
                               records=[], seeds=hits, direction=direction)
    with store._connect() as connection:
        selection = json.loads(connection.execute(
            'SELECT selection_json FROM experience_retrieval_events WHERE id=?', (event,)).fetchone()[0])
    assert selection[0]['candidate_match'] == 'graph_exact'
    assert selection[0]['reason'] == 'full_graph_contract_seed'
    assert not seeds(limit=0)
    for kwargs in ({'direction': 'unknown'}, {'direction': None}, {'hardware': ''},
                   {'hardware': 'device-B'}, {'layout': 'strided'},
                   {'direction': 'forward' if direction != 'forward' else 'joint'}):
        query = dict(direction=direction, hardware='device-A', layout='contiguous')
        query.update(kwargs)
        assert not store.search_verified_seeds(target, 'musa', 'musa', **query)
    assert not store.search_verified_seeds(target, 'cuda', 'musa', direction=direction)
    assert not store.search_verified_seeds(target, 'musa', 'triton', direction=direction)
    store.quarantine('case', source.id, 'suspect source')
    assert not seeds()


@pytest.mark.parametrize('change', ['graph', 'version', 'shape', 'stride', 'dtype',
                                   'outputs', 'parameters', 'cases', 'gradients'])
def test_graph_seed_conflicts_cannot_fall_back_to_identical_request_text(tmp_path, change):
    import copy
    store = SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    old = signature('same-request')
    source = record(old)
    store.add(source)
    interface = copy.deepcopy(old.interface)
    target = old
    if change == 'graph':
        target = replace(old, graph_digest='different-complete-computation')
    elif change == 'version':
        target = replace(old, graph_version='unsupported-trace-version')
    else:
        if change in {'shape', 'stride', 'dtype'}:
            interface['inputs'][0][change] = {'shape': [8, 4], 'stride': [16, 2],
                                             'dtype': 'torch.float16'}[change]
        elif change == 'outputs':
            interface['outputs'] = [{'shape': [4], 'dtype': 'torch.float32', 'stride': [1]}]
        else:
            interface[change] = [{'changed': True}]
        target = replace(old, interface=interface)
    assert not store.search_verified_seeds(target, 'musa', 'musa', direction='forward',
                                           hardware='device-A', layout='contiguous')


def test_graphless_unknown_task_cannot_get_code_by_same_empty_fingerprint(tmp_path):
    store = SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    unknown = replace(signature('unknown'), graph_digest='', graph_version='', interface={})
    store.add(replace(record(unknown), hardware='', layout=''))
    assert not store.search_verified_seeds(unknown, 'musa', 'musa', direction='forward')


@pytest.mark.parametrize('old_family',['unknown','arbitrary-old-family'])
@pytest.mark.parametrize('stage',['generation','optimization'])
def test_graph_contract_bypasses_both_sql_and_score_family_gates(tmp_path,old_family,stage):
    store=SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    old=signature(old_family);target=signature('new-prose-family',request='new')
    r=record(old);store.add(r)
    hits=store.retrieve(target,'musa','musa',direction='forward',stage=stage,hardware='device-A')
    assert len(hits)==1 and hits[0].id==r.id
    assert hits[0].retrieval_reason=='full_graph_contract_text_only'
    assert hits[0].retrieval_match=='whole' and not hits[0].kernel_code
    assert hits[0].graph_match_evidence['interface_match']
    assert hits[0].graph_match_evidence['hardware_compatibility']=='exact'
    assert 'Different prose family labels' in format_experience_context(hits)
    assert 'SOURCE MUST NOT ENTER TEXT' not in format_experience_context(hits)
    assert not store.search_verified_seeds(target,'musa','musa',direction='forward',hardware='device-A')
    event=store.log_retrieval(session_id='S',stage=stage,signature=target,records=hits,direction='forward')
    with store._connect() as c:
        selection=json.loads(c.execute('select selection_json from experience_retrieval_events where id=?',(event,)).fetchone()[0])
        original=c.execute('select semantic_type,kernel_code,outcome from experience_cases where id=?',(r.id,)).fetchone()
    assert selection[0]['reason']=='full_graph_contract_text_only'
    assert tuple(original)==(old_family,'SOURCE MUST NOT ENTER TEXT','success')


@pytest.mark.parametrize('field,value',[
    ('graph_version','unsupported-version'),('graph_digest','different'),
    ('graph_digest',''),('interface',{}),
    ('interface',{'inputs':[{'shape':[4,8],'dtype':'torch.float16','stride':[8,1]}]}),
    ('interface',{'inputs':[{'shape':[4,8],'dtype':'torch.float32','stride':[16,2]}]}),
    ('interface',{'gradient_targets':['parameter:weight'],'upstream':'different'}),
])
def test_graph_identity_without_complete_contract_does_not_bypass_labels(tmp_path,field,value):
    store=SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    store.add(record(signature('unrelated-family')))
    target=replace(signature('current-family',request='new'),**{field:value})
    assert not store.retrieve(target,'musa','musa',direction='forward',stage='generation')


def test_exact_graph_recall_preserves_direction_device_layout_and_quarantine(tmp_path):
    store=SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    r=record(signature('old-family'));store.add(r)
    target=signature('new-family',request='new')
    assert not store.retrieve(target,'musa','musa',direction='backward')
    assert not store.retrieve(target,'cuda','triton',direction='forward')
    assert not store.retrieve(target,'musa','musa',direction='forward',hardware='device-B')
    assert not store.retrieve(target,'musa','musa',direction='forward',layout='strided')
    hits=store.retrieve(target,'musa','musa',direction='forward')
    assert hits[0].graph_match_evidence['hardware_compatibility']=='unresolved'
    store.quarantine('case',r.id,'invalid source observation')
    assert not store.retrieve(target,'musa','musa',direction='forward')


def test_correctness_only_exact_graph_case_still_cannot_consume_text_slot(tmp_path):
    store=SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    store.add(replace(record(signature('old-family')),outcome='generated',lesson='Correct kernel generated'))
    assert not store.retrieve(signature('new-family',request='new'),'musa','musa',direction='forward')


@pytest.mark.parametrize('stage', ['generation', 'optimization'])
def test_shape_transfer_ranks_graph_above_heuristic_cases_without_relaxing_seeds(tmp_path, stage):
    store = SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    old = signature('arbitrary-old-family')
    target = replace(signature('unknown', request='new'), shapes=((3, 7),), interface={
        'inputs': [{'shape': [3, 7], 'dtype': 'torch.float32', 'stride': [7, 1]}],
        'outputs': [{'shape': [3, 7], 'dtype': 'torch.float32', 'stride': [7, 1]}],
    })
    source = record(old)
    store.add(source)
    for i in range(6):
        store.add(record(replace(target, graph_digest='', digest=f'heuristic-{i}'),
                         id=f'heuristic-{i}'))
    hits = store.retrieve(target, 'musa', 'musa', direction='forward', stage=stage,
                          hardware='device-A', limit=1)
    assert hits[0].id == source.id
    assert hits[0].retrieval_reason == 'same_graph_shape_transfer_text_only'
    assert hits[0].graph_match_evidence['interface_match'] is False
    assert not hits[0].kernel_code
    assert not store.search_measured_implementations(target, 'musa', 'musa',
                                                     direction='forward', hardware='device-A')


@pytest.mark.parametrize('change', ['dtype', 'rank', 'layout', 'device', 'direction', 'parameters'])
def test_shape_transfer_does_not_override_incompatible_domain(tmp_path, change):
    store = SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    source = record(signature('unrelated-family'))
    store.add(source)
    inputs = {'shape': [3, 7], 'dtype': 'torch.float32', 'stride': [7, 1]}
    interface = {'inputs': [inputs]}
    options = {'direction': 'forward', 'hardware': 'device-A'}
    if change == 'dtype': inputs['dtype'] = 'torch.float16'
    if change == 'rank': inputs.update(shape=[21], stride=[1])
    if change == 'layout': inputs['stride'] = [14, 2]
    if change == 'parameters': interface['parameters'] = [{'shape': [7], 'dtype': 'torch.float32'}]
    if change == 'device': options['hardware'] = 'device-B'
    if change == 'direction': options['direction'] = 'backward'
    target = replace(signature('new-family'), interface=interface)
    assert not store.retrieve(target, 'musa', 'musa', **options)


def test_comparable_graph_caution_prefers_recent_evidence_in_single_slot(tmp_path):
    store = SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    before = signature('old-label')
    target = replace(signature('unknown'), interface={
        'inputs': [{'shape': [3, 7], 'dtype': 'torch.float32', 'stride': [7, 1]}]})
    for identifier, date in [('old', '2026-10-09'), ('new', '2026-10-10')]:
        store.add(replace(record(before), id=identifier, created_at=date,
                          kernel_code=f'source {identifier}',
                          outcome='regressed', baseline_time_ms=1,
                          comparison={'rule': 'interleaved_mad_v1', 'status': 'inconclusive', 'adopt': False}))
    hits = store.retrieve(target, 'musa', 'musa', direction='forward', hardware='device-A', limit=3)
    assert [hit.id for hit in hits] == ['new']
    for compact in (True, False):
        prompt = format_experience_context(hits, compact=compact)
        assert 'different input contract' in prompt
        assert 'SOURCE MUST NOT ENTER TEXT' not in prompt
