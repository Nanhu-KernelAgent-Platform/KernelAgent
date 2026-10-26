import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from triton_kernel_agent.agent import TritonKernelAgent
from triton_kernel_agent.llm_truncation import LLMCallResult
from triton_kernel_agent.experience import ExperienceRecord, SQLiteExperienceStore, build_operator_signature
from triton_kernel_agent.experience import generation

SOURCE = '''import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x) * x
def get_inputs():
    return [torch.randn(4,8)]
def get_init_inputs():
    return []
'''


@pytest.fixture(scope='module')
def signatures():
    return {direction: build_operator_signature(SOURCE, trace_graph=True, execution_direction=direction)
            for direction in ('forward', 'backward', 'joint')}


@pytest.fixture
def memory(tmp_path,monkeypatch,signatures):
    monkeypatch.setattr(generation,'build_operator_signature',lambda source,**kw:signatures[kw['execution_direction']])
    store=SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    records={}
    for direction in ('forward','backward','joint'):
        record=ExperienceRecord(signatures[direction],'musa','musa','success',f'{direction.upper()} CODE',True,
              direction=direction,kernel_time_ms=1.0,lesson=f'{direction.upper()} LESSON: Check launch overhead and keep the recorded contract.')
        store.add(record);records[direction]=record
    return store,records


def test_direction_override_keeps_text_contract_and_traces_true_backward(signatures):
    assert len({s.digest for s in signatures.values()})==1
    assert len({s.contract_digest for s in signatures.values()})==1
    assert signatures['forward'].graph_version=='torch_export_aten_v1'
    assert signatures['backward'].graph_version=='torch_make_fx_forward_backward_v1'
    assert signatures['backward'].graph_digest != signatures['forward'].graph_digest
    assert signatures['joint'].interface['direction']=='joint'
    assert signatures['backward'].interface['direction']=='backward'


def test_joint_channels_isolate_text_and_retain_only_joint_seeds(memory):
    store,records=memory
    result=generation.generation_memory(store,SOURCE,direction='joint',platform='musa',backend='musa',session_id='J',logger=Mock())
    forward=result.context.split('## FORWARD text experience channel')[1].split('## BACKWARD text experience channel')[0]
    backward=result.context.split('## BACKWARD text experience channel')[1]
    assert 'FORWARD LESSON' in forward and 'BACKWARD LESSON' not in forward
    assert 'BACKWARD LESSON' in backward and 'FORWARD LESSON' not in backward
    assert 'FORWARD CODE' not in result.context and 'BACKWARD CODE' not in result.context
    assert [r.id for r in result.seeds]==[records['joint'].id]
    assert len(result.text_events)==2
    with store._connect() as c:
        rows=c.execute("select stage,direction,case_ids_json,seed_ids_json from experience_retrieval_events order by created_at").fetchall()
    assert [(r['stage'],r['direction']) for r in rows]==[
        ('generation','joint'),('generation_forward_text','forward'),('generation_backward_text','backward')]
    assert json.loads(rows[1]['seed_ids_json'])==json.loads(rows[2]['seed_ids_json'])==[]
    assert json.loads(rows[1]['case_ids_json'])==[records['forward'].id]
    assert json.loads(rows[2]['case_ids_json'])==[records['backward'].id]


def test_single_forward_channel_does_not_retrieve_backward(memory):
    store,records=memory
    result=generation.generation_memory(store,SOURCE,direction='forward',platform='musa',backend='musa',session_id='F',logger=Mock())
    assert 'FORWARD LESSON' in result.context and 'BACKWARD LESSON' not in result.context
    assert not result.text_events and [r.direction for r in result.seeds]==['forward']


def test_reworded_joint_generation_retrieves_only_complete_joint_seed(memory, monkeypatch, signatures):
    store, records = memory
    def reworded(source, **kwargs):
        return replace(signatures[kwargs['execution_direction']], semantic_type='unknown',
                       digest='changed-prose', contract_digest='changed-prose-contract')
    monkeypatch.setattr(generation, 'build_operator_signature', reworded)
    result = generation.generation_memory(store, SOURCE, direction='joint', platform='musa',
        backend='musa', session_id='reworded', logger=Mock())
    assert [(item.id, item.candidate_match) for item in result.seeds] == [
        (records['joint'].id, 'graph_exact')]
    with store._connect() as connection:
        root = connection.execute('SELECT seed_ids_json, selection_json FROM experience_retrieval_events WHERE id=?',
                                  (result.root_event,)).fetchone()
        child_seeds = connection.execute("SELECT seed_ids_json FROM experience_retrieval_events "
                                         "WHERE session_id='reworded' AND stage != 'generation'").fetchall()
    assert json.loads(root[0]) == [records['joint'].id]
    assert all(json.loads(row[0]) == [] for row in child_seeds)
    assert any(item['reason'] == 'full_graph_contract_seed' for item in json.loads(root[1]))


def test_agent_prompt_and_child_completion_audit(memory):
    store,records=memory
    agent=TritonKernelAgent.__new__(TritonKernelAgent)
    agent.num_workers=1;agent.provider=Mock();agent.provider.supports_multiple_completions.return_value=False
    agent.logger=Mock();agent.model_name='fixture';agent.no_cusolver=False
    agent._platform_config=SimpleNamespace(name='musa');agent._kernel_backend=SimpleNamespace(name='musa')
    agent.experience_store=store;agent.prompt_manager=Mock()
    agent.prompt_manager.render_kernel_generation_prompt.return_value='BASE REQUIRED OUTPUT FORMAT'
    agent._call_llm=Mock(return_value='generated')
    agent._call_llm_detailed=Mock(return_value=LLMCallResult('generated', None, None))
    agent._extract_kernel_payload=Mock(return_value='candidate')
    assert agent._generate_kernel_seeds(SOURCE,'test',execution_direction='joint')==['candidate']
    # The seed path calls the signal-carrying variant; audit the prompt there.
    prompt=agent._call_llm_detailed.call_args.args[0][0]['content']
    assert 'BASE REQUIRED OUTPUT FORMAT' in prompt
    assert 'FORWARD LESSON' in prompt and 'BACKWARD LESSON' in prompt and 'JOINT CODE' in prompt
    assert 'FORWARD CODE' not in prompt and 'BACKWARD CODE' not in prompt
    agent._finish_text_retrievals({'success':True,'verified':True})
    with store._connect() as c:
        for event in agent._last_experience_text_retrieval_ids:
            outcome=json.loads(c.execute('select outcome_json from experience_retrieval_events where id=?',(event,)).fetchone()[0])
            assert outcome['parent_generation_event_id']==agent._last_experience_retrieval_id
            assert outcome['parent_generation_direction']=='joint' and outcome['text_only'] and outcome['verified']


def test_empty_backward_graph_and_prompt_budget_are_not_fabricated(memory,monkeypatch,signatures):
    store,records=memory
    def build(source,**kw):
        s=signatures[kw['execution_direction']]
        return replace(s,graph_version='',graph_digest='',graph_nodes=(),graph_tensor_specs={}) if kw['execution_direction']=='backward' else s
    monkeypatch.setattr(generation,'build_operator_signature',build)
    with store._connect() as c:
        c.execute("update experiment_runs set environment_json=json_set(environment_json, '$.case_context.lesson', ?) "
                  "where json_extract(environment_json, '$.case_context.id')=?",
                  ('oversized '*10000,records['forward'].id))
    result=generation.generation_memory(store,SOURCE,direction='joint',platform='musa',backend='musa',session_id='B',logger=Mock())
    assert 'oversized' not in result.context
    assert 'BACKWARD LESSON' in result.context # explicitly directional heuristic, not a structural match
    with store._connect() as c:
        forward=c.execute("select case_ids_json from experience_retrieval_events where session_id='B' and direction='forward'").fetchone()
        backward=c.execute("select signature_json from experience_retrieval_events where session_id='B' and direction='backward'").fetchone()
    assert json.loads(forward[0])==[]
    assert json.loads(backward[0])['graph_digest']==''


def test_shared_validated_text_is_not_duplicated_across_channels(memory):
    store,_=memory
    shared=ExperienceRecord(build_operator_signature('shared'),'musa','musa','supported','',True,
        kind='principle',lesson='SHARED PRINCIPLE',applicable_directions=('forward','backward'),
        observed_directions=('forward','backward'),cross_direction=True)
    store.retrieve=Mock(side_effect=lambda *a,**kw: [] if kw['direction']=='joint' else [shared])
    result=generation.generation_memory(store,SOURCE,direction='joint',platform='musa',backend='musa',session_id='S',logger=Mock())
    assert result.context.count('SHARED PRINCIPLE')==1


def test_unused_joint_budget_keeps_positive_and_caution_in_both_directions(memory):
    from triton_kernel_agent.measurement_policy import interleaved_pair
    store,records=memory
    store.quarantine('case',records['joint'].id,'No eligible shared text in this fixture')
    before,after=interleaved_pair(lambda role:{'samples_ms':[1.0 if role=='parent' else 1.01]*3},'parent','candidate')
    for direction in ('forward','backward'):
        r=replace(records[direction],id=direction+'-caution',kernel_code=direction+'-caution-code',outcome='regressed',
                  baseline_time_ms=1.,kernel_time_ms=1.01,baseline_samples_ms=tuple(before['samples_ms']),
                  kernel_samples_ms=tuple(after['samples_ms']),comparison=after['comparison'],
                  lesson='Check launch overhead under the recorded conditions before increasing grid parallelism. '*20)
        store.add_online(r,session_id='caution-'+direction)
    result=generation.generation_memory(store,SOURCE,direction='joint',platform='musa',backend='musa',session_id='budget',logger=Mock())
    assert len(result.context)<12000
    with store._connect() as c:
        for row in c.execute("select direction,selection_json from experience_retrieval_events where session_id='budget' and direction!='joint'"):
            selection=json.loads(row['selection_json'])
            assert len(selection)==2 and any(x['text_evidence_status']=='inconclusive' for x in selection)
