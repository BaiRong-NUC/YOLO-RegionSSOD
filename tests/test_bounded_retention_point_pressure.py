"""Pressure population and checkpoint guards; no real-data forward in pytest."""
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('pressure_audit', SCRIPTS / 'check_bounded_retention_point_pressure.py')
pressure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pressure)


def test_fixed_population_has_both_pressure_states_and_complete_epoch9_prefix():
    cases = pressure.fixed_cases()
    assert len(cases) == len(set(cases)) == 34
    assert cases[:2] == [('initial', 'pressure'), ('epoch9', 'pressure')]
    assert cases[2:] == [('epoch9', f'{p}_{i:02d}') for p in ('open', 'closed') for i in range(16)]
    assert pressure.ARMS == ('p1', 'new_candidate', 'new_control')
    assert pressure.SCALE == 1024 and pressure.TIMEOUT == 120


@pytest.mark.parametrize('checkpoint', [
    {'epoch': 9, 'ema': object()}, {'epoch': 7, 'ema': object()},
    {'epoch': 8, 'model': object()}, {'epoch': 8, 'ema': None},
])
def test_reject_wrong_epoch_or_online_model_fallback(checkpoint):
    with pytest.raises(RuntimeError):
        pressure.verify_epoch_checkpoint(checkpoint)


def test_uses_only_epoch9_ema_without_consulting_optimizer():
    class Checkpoint(dict):
        def get(self, name, *args):
            assert name in ('epoch', 'ema')
            return super().get(name, *args)
    ema = object()
    assert pressure.verify_epoch_checkpoint(Checkpoint(epoch=8, ema=ema, optimizer='forbidden')) is ema


def test_supervision_requires_consumed_density_not_just_observation():
    from types import SimpleNamespace
    criterion = SimpleNamespace(supports={
        'a': {'domain': SimpleNamespace(dimension=3)},
        'b': {'domain': SimpleNamespace(dimension=2), 'density_diagnostics': {7: {}, 9: {}}}})
    assert pressure.supervision_profile(criterion) == {'3': 0, '2': 2}
