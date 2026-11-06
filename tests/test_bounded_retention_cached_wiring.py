"""Protect output isolation, conflicting evidence hashes and diagnostic deadlines."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def wiring(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('cached_wiring_test', scripts / 'check_bounded_retention_cached_wiring.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_factory_preserves_real_code_and_scientific_args_without_global_mutation(wiring, tmp_path):
    namespace = {'expected_args': lambda config: dict(config)}
    exec('def make_trainer(arm, config, *, source=None):\n    return arm, expected_args(config), source', namespace)
    original = namespace['make_trainer']
    launcher = SimpleNamespace(make_trainer=original, expected_args=namespace['expected_args'])
    isolated = wiring.diagnostic_factory(launcher, tmp_path)
    config = {'name': 'candidate', 'project': 'production', 'save_dir': 'production/candidate',
              'save': True, 'seed': 0, 'amp': True, 'batch': 8, 'epochs': 30, 'lr0': .01}
    marker = object()
    arm, args, source = isolated('candidate', config, source=marker)
    assert isolated.__code__ is original.__code__ and source is marker and arm == 'candidate'
    assert args == config | {'project': str(tmp_path), 'save_dir': str(tmp_path / 'candidate'), 'save': False}
    assert original('candidate', config)[1] == config
    assert namespace['expected_args'] is launcher.expected_args
    assert not list(tmp_path.iterdir())


def test_conflicting_bound_input_cannot_be_silently_overwritten(wiring, tmp_path):
    path = tmp_path / 'source.py'
    canonical = str(path.resolve())
    assert wiring.merge_hashes({path: 'a'}, {canonical: 'a'}) == {canonical: 'a'}
    with pytest.raises(RuntimeError, match='conflicting input SHA'):
        wiring.merge_hashes({path: 'a'}, {canonical: 'b'})


@pytest.mark.parametrize('seconds', [-1., math.nan, math.inf, -math.inf, math.nextafter(120., math.inf)])
def test_polling_gap_or_nonfinite_stage_time_is_rejected(wiring, seconds):
    with pytest.raises(RuntimeError, match='120-second'):
        wiring.elapsed_ok(seconds)


@pytest.mark.parametrize('seconds', [0., 119.99, 120.])
def test_fixed_inclusive_stage_deadline(wiring, seconds):
    wiring.elapsed_ok(seconds)


def test_tensor_signed_zero_is_not_byte_identical(wiring):
    import torch
    a = {'preds': [torch.tensor([0., 1.])]}
    assert wiring.strict_tree_equal(a, {'preds': [a['preds'][0].clone()]})
    assert not wiring.strict_tree_equal(a, {'preds': [torch.tensor([-0., 1.])]})
    assert not wiring.strict_tree_equal(a, {'preds': [a['preds'][0].double()]})


def test_new_heartbeat_cannot_hide_late_return(wiring, tmp_path, monkeypatch):
    ticks = iter([0., 120.001])
    monkeypatch.setattr(wiring.time, 'perf_counter', lambda: next(ticks))
    beat = wiring.StrictHeartbeat(tmp_path)
    beat('loss')
    with pytest.raises(RuntimeError, match='120-second'):
        beat('backward')
    assert len(list((tmp_path / 'heartbeats').glob('*.json'))) == 1
