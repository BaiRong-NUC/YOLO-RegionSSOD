"""Install the independent solver only inside zero-training loss checks."""
import pytest
import torch
import test_bounded_retention_native_e_loss as frozen_checks
from regionssod.training import bounded_retention_objective_v6 as objective
from regionssod.training.bounded_retention_point_search_v2 import optimize_point


@pytest.fixture(autouse=True)
def independent_solver(monkeypatch):
    monkeypatch.setattr(objective, 'optimize_point', optimize_point)


DEVICES=['cpu',pytest.param('cuda',marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='GPU absent'))]


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('case',['negative','overflow'])
@pytest.mark.parametrize('mode',['reference','candidate','control'])
def test_native_all_known_loss_and_gradients_unchanged(device,case,mode):
    frozen_checks.test_wholly_known_native_loss_and_gradient_parity(case,mode,device)


@pytest.mark.parametrize('device',DEVICES)
@pytest.mark.parametrize('dimensions',[1,2])
def test_same_control_geometry_assignment_and_rng_with_distinct_loss_gradients(device,dimensions):
    frozen_checks.test_full_partial_loss_same_assignment_geometry_and_independent_rng(device,dimensions)


@pytest.mark.parametrize('which',['negative_e','empty_bins'])
def test_continuous_semantics_preserved(which):
    frozen_checks.test_both_old_blockers_use_actual_continuous_interval_loss(which)
