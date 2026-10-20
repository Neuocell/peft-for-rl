"""Protect result reporting against corrupted sampling and ambiguous step names."""
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/analysis/report_local_eval_protocols.py'
spec = importlib.util.spec_from_file_location('report_local_eval_protocols', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows():
    return [dict(benchmark='aime24', id='q', samples_per_problem=2, sample_index=i,
                 correct=i == 0, problem='Question', gold_answer='42') for i in range(2)]


def test_reject_duplicate_samples():
    sample = rows()
    sample[1]['sample_index'] = 0
    with pytest.raises(ValueError, match='Duplicate'):
        module.measure(sample)


def test_reject_complete_count_with_invalid_indices():
    sample = rows()
    sample[1]['sample_index'] = 2
    with pytest.raises(ValueError, match='sample indices'):
        module.measure(sample)


def test_reject_question_gold_drift_within_group():
    sample = rows()
    sample[1]['gold_answer'] = '43'
    with pytest.raises(ValueError, match='question/gold'):
        module.measure(sample)


def test_incomplete_group_is_not_an_evaluation():
    with pytest.raises(ValueError, match='Incomplete'):
        module.measure(rows()[:1])


def test_full_parameter_checkpoint_uses_terminal_step():
    assert module.step('full_parameter_step50_seed42_v2_step200_fullbench') == 200


def test_conflicting_sources_cannot_be_merged():
    def inventory(source, digest):
        return dict(roots=['runs'], candidates=[dict(checkpoint='example', source=source,
                    source_sha256=digest, status='ready')], summary_without_ready_records=[])
    with pytest.raises(ValueError, match='Conflicting checkpoint'):
        module.merge_inventories([inventory('one','a'), inventory('two','b')])


def test_summary_only_aliases_are_not_counted_as_separate_experiments():
    def inventory(path):
        return dict(roots=['runs'], candidates=[], summary_without_ready_records=[dict(checkpoint='example',path=path)])
    result = module.merge_inventories([inventory('one'), inventory('two')])
    assert len(result['summary_without_ready_records']) == 1
    assert result['summary_without_ready_records'][0]['aliases'] == ['two']
