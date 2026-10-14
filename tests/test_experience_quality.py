import json
from dataclasses import replace

import pytest

from triton_kernel_agent.experience import ExperienceRecord, OperatorSignature, SQLiteExperienceStore
from triton_kernel_agent.experience.quality import informative, same_idea, redundant, useful
from triton_kernel_agent.experience.consolidation import review_merge, review_unmerge, proposals
from triton_kernel_agent.experience.experiment_db import lesson_report
from triton_kernel_agent.measurement_policy import interleaved_pair


def observed(text,shape=128,*,candidate=.8,direction='forward',conditions=None):
    parent,after=interleaved_pair(lambda role:{'samples_ms':[1. if role=='parent' else candidate]*3},'parent','candidate')
    claim={'text':text,'claim_type':'performance_recommendation','predicate':'candidate_improves_parent',
           'conditions':conditions or {'layout':'contiguous'}}
    return ExperienceRecord(OperatorSignature('elementwise',dtypes=('float32',),shapes=((shape,),)),'musa','musa','improved',f'code-{shape}-{candidate}',True,
         direction=direction,hardware='test-gpu',layout='contiguous',kernel_time_ms=candidate,baseline_time_ms=1.,
         baseline_samples_ms=tuple(parent['samples_ms']),kernel_samples_ms=tuple(after['samples_ms']),comparison=after['comparison'],
         lesson='Evaluate launch overhead before changing grid parallelism.',lesson_claims=(claim,))


def claim_id(c,text):
    return c.execute('select id from experiment_claims where text=?',(text,)).fetchone()[0]


@pytest.mark.parametrize('text',['Generated a correct kernel','Kernel is correct','生成了正确 Kernel','优化完成',''])
def test_boilerplate_has_no_information(text):
    assert not informative(text)


@pytest.mark.parametrize('marked', [False, True])
def test_failed_summary_is_ledger_only_even_with_action_and_bottleneck(tmp_path, marked):
    store = SQLiteExperienceStore(tmp_path / 'memory.sqlite3')
    reflection = {'reasoning': 'Performance changed by +0.3%', 'lessons': [], 'claims': []}
    if marked:
        reflection.update(generation_status='failed', generation_error='ConnectionError')
    record = ExperienceRecord(OperatorSignature('elementwise'), 'musa', 'musa', 'improved', 'code', True,
        direction='forward', action='Change grid parallelism after profiling', bottleneck='underutilized device',
        lesson=reflection['reasoning'], environment={'raw_reflection': reflection})
    store.add_online(record, session_id='summary-failed')
    assert not store.retrieve(record.signature, 'musa', 'musa', direction='forward')
    with store._connect() as c:
        assert c.execute('select count(*) from experiment_attempts').fetchone()[0] == 1
        assert c.execute('select count(*) from experience_cases').fetchone()[0] == 1
    # Summary failures do not suppress real correctness diagnostics.
    assert useful(replace(record, verified=False, outcome='failed', error_message='Output shape mismatch'))


def test_prompt_quality_preserves_records_and_deduplicates_without_losing_direction(tmp_path):
    store=SQLiteExperienceStore(tmp_path/'memory.sqlite3');s=OperatorSignature('elementwise')
    texts=['Kernel is correct','Check launch overhead before changing grid parallelism.','CHECK launch overhead before changing grid parallelism.']
    for i,text in enumerate(texts):
        store.add(ExperienceRecord(s,'musa','musa','success',str(i),True,direction='forward',lesson=text))
    hits=store.retrieve(s,'musa','musa',direction='forward',limit=4)
    assert len(hits)==1 and 'launch overhead' in hits[0].lesson.lower()
    with store._connect() as c:assert c.execute('select count(*) from experience_cases').fetchone()[0]==3
    assert not same_idea('Use 128 threads for this layout','Use 256 threads for this layout')
    assert not same_idea('Do not fuse the launch stages','Do fuse the launch stages')
    assert not same_idea('Increase vector width when the layout permits contiguous access',
                         'Decrease vector width when the layout permits contiguous access')
    assert not redundant(replace(hits[0],direction='backward'),hits)


def test_reviewed_consolidation_unions_sessions_keeps_counterevidence_and_can_be_revoked(tmp_path):
    store=SQLiteExperienceStore(tmp_path/'memory.sqlite3')
    a='The measured launch change improves the tested workload'
    b='The measured launch change improves this tested workload'
    store.add_online(observed(a),session_id='one');store.add_online(observed(b,256),session_id='two')
    with store._connect() as c:
        ids=[claim_id(c,text) for text in [a,b]]
        before=[c.execute('select count(*) from '+table).fetchone()[0] for table in ['experiment_claims','experiment_claim_evidence','experiment_measurements']]
        assert proposals(c) and all(p['requires_review'] for p in proposals(c))
        assert all(x['status']=='candidate' for x in lesson_report(c) if x['claim_type']=='performance_recommendation')
        review_merge(c,ids[1],ids[0],reviewer='tester',reason='Same bounded proposition and conditions; raw evidence inspected')
        report=next(x for x in lesson_report(c) if x['claim_type']=='performance_recommendation')
        assert report['status']=='validated' and report['eligible_runs']==2 and set(report['member_claim_ids'])==set(ids)
        assert before==[c.execute('select count(*) from '+table).fetchone()[0] for table in ['experiment_claims','experiment_claim_evidence','experiment_measurements']]
    store.add_online(observed(b,512,candidate=1.2),session_id='counterexample')
    with store._connect() as c:
        assert next(x for x in lesson_report(c) if x['claim_type']=='performance_recommendation')['status']=='contested'
        assert review_unmerge(c,ids[1],reviewer='tester',reason='Reassess scope after counterexample')
        assert c.execute('select revoked_by from experiment_claim_aliases').fetchone()[0]=='tester'
        assert len([x for x in lesson_report(c) if x['claim_type']=='performance_recommendation'])==2


def test_alias_does_not_turn_same_session_into_independent_support(tmp_path):
    store=SQLiteExperienceStore(tmp_path/'memory.sqlite3');a='Change grid mapping under measured launch overhead';b=a+' conditions'
    store.add_online(observed(a),session_id='same');store.add_online(observed(b,256),session_id='same')
    with store._connect() as c:
        review_merge(c,claim_id(c,b),claim_id(c,a),reviewer='tester',reason='Equivalent bounded statements')
        report=next(x for x in lesson_report(c) if x['claim_type']=='performance_recommendation')
        assert report['eligible_runs']==1 and report['status']=='candidate'


@pytest.mark.parametrize('direction,conditions',[('backward',None),('forward',{'layout':'strided'})])
def test_merge_rejects_direction_or_condition_drift(tmp_path,direction,conditions):
    store=SQLiteExperienceStore(tmp_path/'memory.sqlite3');a='Apply launch mapping after checking layout';b=a+' requirements'
    store.add_online(observed(a),session_id='one');store.add_online(observed(b,256,direction=direction,conditions=conditions),session_id='two')
    with store._connect() as c:
        with pytest.raises(ValueError):review_merge(c,claim_id(c,b),claim_id(c,a),reviewer='tester',reason='Should be refused')


def test_alias_cycle_and_blank_approval_are_refused(tmp_path):
    store=SQLiteExperienceStore(tmp_path/'memory.sqlite3');a='Check launch count before grid changes';b=a+' locally'
    store.add_online(observed(a),session_id='one');store.add_online(observed(b,256),session_id='two')
    with store._connect() as c:
        aid,bid=claim_id(c,a),claim_id(c,b)
        with pytest.raises(ValueError):review_merge(c,bid,aid,reviewer='',reason='')
        review_merge(c,bid,aid,reviewer='tester',reason='Reviewed')
        with pytest.raises(ValueError):review_merge(c,aid,bid,reviewer='tester',reason='Cycle')


def test_review_cli_approves_and_revokes_without_erasing_raw_claims(tmp_path,capsys):
    from scripts.experience_db import main
    path=tmp_path/'memory.sqlite3';store=SQLiteExperienceStore(path)
    a='Check vector layout before changing launch mapping';b='Check the vector layout before changing launch mapping'
    store.add_online(observed(a),session_id='one');store.add_online(observed(b,256),session_id='two')
    with store._connect() as c: aid,bid=claim_id(c,a),claim_id(c,b)
    main(['--db',str(path),'merge-proposals']);assert json.loads(capsys.readouterr().out)
    main(['--db',str(path),'merge-claims',bid,aid,'--reviewer','tester','--reason','Verified equivalence'])
    assert json.loads(capsys.readouterr().out)['aliased']==bid
    main(['--db',str(path),'unmerge-claim',bid,'--reviewer','tester','--reason','Review revoked'])
    assert json.loads(capsys.readouterr().out)['alias_revoked']
    with store._connect() as c:
        assert c.execute('select count(*) from experiment_claims where id in (?,?)',(aid,bid)).fetchone()[0]==2
        assert c.execute('select revocation_reason from experiment_claim_aliases').fetchone()[0]=='Review revoked'
