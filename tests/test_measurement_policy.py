from dataclasses import replace
import logging
from types import SimpleNamespace

import pytest

from triton_kernel_agent.measurement_policy import compare, interleaved_pair, RULE
from triton_kernel_agent.experience import ExperienceRecord, OperatorSignature, SQLiteExperienceStore
from triton_kernel_agent.experience.experiment_db import run_report, lesson_report
from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import OptimizationOrchestrator


def samples(blocks):
    return [value for block in blocks for value in [block] * 3]


@pytest.mark.parametrize('parent,candidate,status', [
    ([1.] * 5, [.8] * 5, 'faster'),
    ([1.] * 5, [1.2] * 5, 'slower'),
    ([1.] * 5, [.99] * 5, 'inconclusive'),
    ([1.] * 5, [.8, 1.2, .7, 1.1, .85], 'inconclusive'),
    ([.001] * 5, [.00095] * 5, 'inconclusive'),
])
def test_practical_gain_and_noise_envelope(parent, candidate, status):
    result = compare(samples(parent), samples(candidate), interleaved=True)
    assert result['status'] == status
    assert result['adopt'] is (status == 'faster')


def test_missing_legacy_and_invalid_samples_are_inconclusive():
    for parent, candidate, interleaved in (([1] * 3, [.5] * 3, True),
            ([1] * 15, [.5] * 15, False), ([1] * 15, [float('nan')] * 15, True)):
        assert not compare(parent, candidate, interleaved=interleaved)['adopt']


def test_duplicate_measurement_cannot_hide_a_missing_case():
    from triton_kernel_agent.case_suite import measurement_is_covered
    checked = [{'case_key': key, 'status': 'passed', 'contract': {}} for key in ('a', 'b')]
    measured = [{'case_key': 'a', 'inputs': {}}] * 2
    assert not measurement_is_covered(measured, checked)


def test_broad_window_jitter_is_not_hidden_by_identical_block_medians():
    result = compare([.5, 1., 2.] * 5, [1.5, .9, .3] * 5, interleaved=True)
    assert result['status'] == 'inconclusive' and not result['adopt']


def test_ab_ba_schedule_preserves_raw_windows_and_vetoes_case_regression():
    order = []
    def measure(role):
        order.append(role)
        return {'samples_ms': [1 if role == 'parent' else .8] * 3,
                'case_measurements': [{'case_key': 'one', 'inputs': {'shape': [10]},
                     'samples_ms': [1 if role == 'parent' else 1.2] * 3, 'time_ms': 1}]}
    parent, candidate = interleaved_pair(measure, 'parent', 'candidate')
    assert order == ['parent', 'candidate', 'candidate', 'parent', 'parent', 'candidate',
                     'candidate', 'parent', 'parent', 'candidate']
    assert len(parent['samples_ms']) == len(candidate['samples_ms']) == 15
    assert candidate['comparison']['status'] == 'faster'
    assert candidate['comparison']['reason'] == 'case_coverage_or_material_regression'
    assert not candidate['comparison']['adopt']


def test_regression_label_does_not_exclude_verified_measured_code(tmp_path):
    store = SQLiteExperienceStore(tmp_path / 'memory.sqlite3')
    signature = OperatorSignature('elementwise', shapes=((10,),), dtypes=('float32',), contract_digest='same')
    record = ExperienceRecord(signature=signature, platform='musa', kernel_backend='musa', direction='backward',
        outcome='regressed', verified=True, kernel_code='previous selected code', selected_by_worker=True,
        baseline_time_ms=.8, kernel_time_ms=.9, hardware='GPU')
    store.add_online(record, session_id='old')
    hits = store.search_measured_implementations(signature, 'musa', 'musa', direction='backward', hardware='GPU')
    assert [row.kernel_code for row in hits] == [record.kernel_code]
    assert hits[0].outcome == 'regressed'
    assert not store.search_measured_implementations(signature, 'musa', 'musa', direction='forward', hardware='GPU')
    store.quarantine('case', record.id, 'invalid verification')
    assert not store.search_measured_implementations(signature, 'musa', 'musa', direction='backward', hardware='GPU')


def test_inconclusive_comparison_is_stored_but_cannot_validate_claim(tmp_path):
    before, after = interleaved_pair(lambda role: {'samples_ms': [1 if role == 'parent' else .99] * 3}, 'parent', 'candidate')
    store = SQLiteExperienceStore(tmp_path / 'memory.sqlite3')
    record = ExperienceRecord(signature=OperatorSignature('elementwise'), platform='musa', kernel_backend='musa',
        direction='forward', outcome='improved', verified=True, kernel_code='code',
        baseline_samples_ms=tuple(before['samples_ms']), kernel_samples_ms=tuple(after['samples_ms']),
        comparison=after['comparison'], selected_by_worker=False,
        lesson_claims=({'text': 'The launch change helps', 'claim_type': 'performance_recommendation'},))
    run_id = store.add_online(record, session_id='new')
    with store._connect() as connection:
        attempt = run_report(connection, run_id)['attempts'][0]
        report = lesson_report(connection)
    assert attempt['decision']['comparison_rule'] == RULE
    assert attempt['decision']['noise_aware_comparison']['status'] == 'inconclusive'
    assert all(row['relation'] == 'inconclusive' for row in attempt['claims'])
    assert all(row['eligible_runs'] == 0 for row in report)


def test_noise_rejection_keeps_incumbent_even_if_median_is_faster():
    worker = OptimizationOrchestrator.__new__(OptimizationOrchestrator)
    worker.logger = logging.getLogger(__name__)
    worker.divergence_threshold = 20
    result = worker._update_kernels('candidate', .99, 0, 'parent', 'parent', 1, 0,
                                   'parent', 1, 0, 1, allow_runtime_improvement=False)
    assert result[:3] == ('parent', 'parent', 1)


def test_forged_comparison_status_cannot_override_observed_samples(tmp_path):
    before, after = interleaved_pair(lambda role: {'samples_ms': [1 if role == 'parent' else 2] * 3}, 'parent', 'candidate')
    forged = {**after['comparison'], 'status': 'faster', 'adopt': True}
    store = SQLiteExperienceStore(tmp_path / 'memory.sqlite3')
    record = ExperienceRecord(signature=OperatorSignature('elementwise'), platform='musa', kernel_backend='musa',
        direction='forward', outcome='improved', verified=True, kernel_code='code',
        baseline_samples_ms=tuple(before['samples_ms']), kernel_samples_ms=tuple(after['samples_ms']), comparison=forged,
        lesson_claims=({'text': 'Helps', 'claim_type': 'performance_recommendation'},))
    run_id = store.add_online(record, session_id='test')
    with store._connect() as connection:
        attempt = run_report(connection, run_id)['attempts'][0]
    assert attempt['decision']['noise_aware_comparison']['status'] == 'slower'
    assert next(row for row in attempt['claims'] if row['claim_type'] == 'performance_recommendation')['relation'] == 'contradicts'


def test_musa_pair_holds_one_lock_and_restores_it_on_error(tmp_path):
    from pathlib import Path
    from triton_kernel_agent.platform.musa import MusaBenchmarker
    class Lock:
        held = False
        acquisitions = 0
        def acquire(self):
            assert not self.held
            self.held = True
            self.acquisitions += 1
        def release(self):
            assert self.held
            self.held = False
    lock = Lock()
    backend = MusaBenchmarker(tmp_path, logging.getLogger(__name__), lock)
    seen = []
    def measure(path, problem):
        assert lock.held and backend._paired_lock_held
        seen.append(path.name)
        return {'samples_ms': [1 if path.name == 'parent.py' else .8] * 3}
    backend.benchmark_kernel = measure
    before, after = backend.benchmark_pair(Path('parent.py'), Path('candidate.py'), Path('problem.py'))
    assert lock.acquisitions == 1 and not lock.held
    assert len(seen) == 10 and after['comparison']['adopt']
    def fail(*args):
        raise RuntimeError('benchmark error')
    backend.benchmark_kernel = fail
    with pytest.raises(RuntimeError):
        backend.benchmark_pair(Path('parent.py'), Path('candidate.py'), Path('problem.py'))
    assert not lock.held and not backend._paired_lock_held


def test_legacy_median_evidence_does_not_promote_noise_aware_principle(tmp_path):
    store = SQLiteExperienceStore(tmp_path / 'memory.sqlite3')
    for shape in (10, 20):
        store.add_online(ExperienceRecord(signature=OperatorSignature('elementwise', shapes=((shape,),)),
            platform='musa', kernel_backend='musa', direction='forward', verified=True, outcome='improved',
            kernel_code=str(shape), baseline_samples_ms=(1., 1., 1.), kernel_samples_ms=(.5, .5, .5),
            lesson_claims=({'text': 'Old median-only claim', 'claim_type': 'performance_recommendation'},)),
            session_id=str(shape))
    with store._connect() as connection:
        reports = lesson_report(connection)
    assert all(report['status'] == 'candidate' and report['eligible_runs'] == 0 for report in reports)
@pytest.mark.parametrize('values,count,expected', [
    ([1.] * 3, 3, True), ([1.] * 15, 3, False), ([1.] * 3, 15, False),
    ([1.] * 15, 15, True), ([True, 1., 1.], 3, False),
    ([float('inf'), 1., 1.], 3, False), ([], None, False),
])
def test_shared_sample_check_preserves_stage_specific_window_counts(values, count, expected):
    from triton_kernel_agent.case_suite import valid_samples
    assert valid_samples(values, count) is expected
