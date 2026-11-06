"""Diagnosis must preserve results and its absolute deadline; never train."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('timeout_diagnosis', SCRIPTS / 'diagnose_bounded_retention_loss_timeout.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


@pytest.mark.parametrize('now,expected', [(219.999,False),(220,True),(900,True)])
def test_loss_deadline_is_absolute_despite_nested_progress(now, expected):
    assert diag.deadline(started=10, loss_started=100, now=now) is expected


def test_initialization_uses_separate_unchanged_120_second_guard():
    assert not diag.deadline(10, None, 129.999)
    assert diag.deadline(10, None, 130)
    assert diag.LIMIT == 120 and diag.ARMS == ('p1','candidate','control')


def test_trace_preserves_identity_arguments_return_and_nested_parent(tmp_path):
    tracer = diag.Trace(tmp_path/'trace.jsonl')
    value = object()
    calls = []
    def inner(arg, *, option):
        calls.append((arg, option))
        return arg
    assert tracer.call('outer', lambda: tracer.call('inner', inner, value, option=7)) is value
    tracer.file.close()
    rows = [json.loads(s) for s in (tmp_path/'trace.jsonl').read_text().splitlines()]
    assert calls == [(value,7)]
    assert [(r['kind'],r['name']) for r in rows] == [('enter','outer'),('enter','inner'),('exit','inner'),('exit','outer')]
    assert rows[1]['parent'] == rows[0]['id'] and tracer.stack == []


def test_trace_keeps_original_exception_without_retry(tmp_path):
    tracer = diag.Trace(tmp_path/'trace.jsonl')
    failure = ValueError('numerical failure')
    def fail():
        raise failure
    with pytest.raises(ValueError) as caught:
        tracer.call('point', fail)
    assert caught.value is failure and not tracer.stack
    tracer.file.close()
    assert len((tmp_path/'trace.jsonl').read_text().splitlines()) == 2


def test_metadata_view_constructs_original_loss_without_detector():
    import torch
    from regionssod.training.p1 import P1DetectionLoss
    view = diag.metadata_view(torch, {'box':7.5,'cls':.5,'dfl':1.5}, 'cpu')
    criterion = P1DetectionLoss(view)
    assert criterion.nc == 10 and criterion.reg_max == 16
    assert criterion.stride.tolist() == [8,16,32]
    assert criterion.hyp.box == 7.5 and criterion.p1_options == view.p1_options
    assert not hasattr(view, 'forward') and not hasattr(view, 'train')


def test_tree_movement_keeps_non_tensor_source_observations():
    import torch
    source = {'source_object_id':17, 'occurrence_id':33}
    assert diag.move_tree(torch, {'observations':[source]}) == {'observations':[source]}


def test_real_cached_domain_records_data_without_pickling_methods():
    from test_bounded_retention_point_search import fixture
    domain = fixture(n=3)
    before = domain._point_cached.cache_info()
    data = diag.domain_data(domain)
    row = json.loads(data)
    assert row['active_sides'] == list(domain.active_sides)
    assert row['source_key'] and b'_lru_cache_wrapper' not in data
    assert domain._point_cached.cache_info() == before
    assert diag.domain_data(fixture(n=3)) == data


def test_python_stack_snapshot_contains_only_location_metadata():
    import threading
    frames = diag.python_stack(threading.get_ident())
    assert frames and all(set(f) == {'file','function','line'} for f in frames)
    assert any(f['function'] == 'test_python_stack_snapshot_contains_only_location_metadata' for f in frames)
    assert diag.python_stack(-1) == []
