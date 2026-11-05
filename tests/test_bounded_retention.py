"""Counterexamples and native-DFL parity; no real labels or optimizer steps."""
from decimal import Decimal
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from ultralytics.utils.loss import DFLoss

from regionssod.training.bounded_retention import (
    UnsupportedObservation, bounded_dfl, edge_intervals, feasible_atoms, serialized_interval,
)
from regionssod.data.annotations import Tile, YoloBox, clip_box_to_tile


@pytest.mark.parametrize("value", [.1, .100000000049, .99999999996, .0123456789123, 1e-15, 0.])
def test_decimal_envelope_contains_formatter_input(value):
    assert serialized_interval(format(value, ".10g")).contains(Decimal.from_float(value))


def test_known_decimal_does_not_certify_exact_preexport_coordinate():
    a, b = .23456789121, .23456789124
    assert a != b and format(a, ".10g") == format(b, ".10g")
    interval = serialized_interval(format(a, ".10g"))
    assert interval.contains(Decimal.from_float(a)) and interval.contains(Decimal.from_float(b))
    assert interval.lo < interval.hi


def test_boundary_contact_is_ambiguity_not_cut_certificate():
    intervals = edge_intervals([".2", ".3", ".4", ".2"])
    assert intervals[0].contains(0)
    assert intervals[0].hi > 0  # an inward edge is also compatible with rounding
    assert intervals[1].lo > 0


def test_legally_retained_input_is_excluded_by_literal_float32_exact_section():
    # All source coordinates here are binary-exact. This is independent of
    # the source decoder's .10g uncertainty and uses no actual hidden labels.
    source = YoloBox(0, .2421875, .3125, .109375, .0625)
    visible = clip_box_to_tile(source, Tile(160, 160, 260, 260), 640, 640, .35)
    assert visible is not None
    text = [format(v, ".10g") for v in (visible.x, visible.y, visible.w, visible.h)]
    assert text == ["0.15", "0.4", "0.3", "0.4"]
    original_right = (source.x + source.w / 2) * 640
    nominal = np.asarray(text, dtype=np.float32)
    fixed_right = float(nominal[0] + nominal[2] / 2) * 100 + 160
    assert original_right == 190
    assert fixed_right > original_right
    # Enlarging an AREA upper bound cannot make these unequal coordinates equal.
    for area_cap in (1200 / (.35 - 1e-12), 1e9):
        assert 70 * 40 < area_cap
        assert fixed_right != original_right


def source_audit_module():
    spec = importlib.util.spec_from_file_location("bounded_source_audit",
        Path(__file__).parents[1] / "scripts/audit_bounded_retention_source.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_collision_uses_actual_export_function_and_preserves_known_coordinate_ambiguity():
    result = source_audit_module().synthetic_rounding_collision()
    assert result["known_right_edge_a"] != result["known_right_edge_b"]
    assert result["synthetic_only"]


def test_source_audit_rejects_hidden_label_paths_before_read():
    audit = source_audit_module()
    for path in (audit.ROOT / "data/visdrone_area10/unlabeled/labels/train/x.txt",
                 audit.LEGAL / "../outside.txt", audit.LEGAL / "x.json"):
        with pytest.raises(ValueError, match="explicitly allowed"):
            audit.normalized_path(path)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_all_known_matches_original_dfl_value_and_logits_gradient(seed):
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(4, 16, generator=gen, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    target = torch.tensor([3.25, 1.1, 7.3, 2.8], dtype=torch.float64)
    actual = bounded_dfl(a, target, torch.zeros(4, dtype=torch.bool), 100.)
    expected = DFLoss(16)(b, target[None].clone()).squeeze()
    assert torch.equal(actual, expected)
    actual.backward()
    expected.backward()
    assert torch.equal(a.grad, b.grad)


def test_mass_and_map_have_different_gradients_on_same_set():
    logits = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        logits[2:, 3] = .7  # unique compatible MAP; uniform ties share max gradients
    target = torch.tensor([2., 2., 2., 2.], dtype=torch.float64)
    mask = torch.tensor([False, False, True, True])
    mass = bounded_dfl(logits, target, mask, 16 / .35)
    map_loss = bounded_dfl(logits, target, mask, 16 / .35, mode="map")
    g_mass, = torch.autograd.grad(mass, logits, retain_graph=True)
    g_map, = torch.autograd.grad(map_loss, logits)
    assert mass < map_loss
    assert torch.isfinite(g_mass).all() and torch.isfinite(g_map).all()
    assert not torch.equal(g_mass, g_map)


def test_uniform_map_ties_can_share_mass_gradient_despite_different_loss():
    logits = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([2., 2., 2., 2.], dtype=torch.float64)
    mask = torch.tensor([False, False, True, True])
    mass = bounded_dfl(logits, target, mask, 16 / .35)
    map_loss = bounded_dfl(logits, target, mask, 16 / .35, mode="map")
    gm, = torch.autograd.grad(mass, logits, retain_graph=True)
    gp, = torch.autograd.grad(map_loss, logits)
    assert mass < map_loss
    assert torch.equal(gm, gp)


def test_area_couples_unknown_edges_and_uses_continuous_known_edges():
    target = torch.tensor([2.25, 2., 2., 2.], dtype=torch.float64)
    mask = torch.tensor([False, False, True, True])
    atoms, geometry = feasible_atoms(target, mask, 16 / .35)
    assert torch.equal(geometry[:, 0], torch.full_like(geometry[:, 0], 2.25))
    assert ((atoms == torch.tensor([2, 6])).all(1)).any()
    assert ((atoms == torch.tensor([6, 2])).all(1)).any()
    assert not ((atoms == torch.tensor([6, 6])).all(1)).any()


@pytest.mark.parametrize("target,area,reason", [
    ([-.1, 2, 2, 2], 46, "negative"),
    ([15.5, 2, 2, 2], 500, "exact distance"),
    ([2, 2, 16, 2], 500, "empty"),
    ([2, 2, 2, 2], 15, "empty"),
])
def test_partial_unknown_invalid_event_fails_without_clamp(target, area, reason):
    with pytest.raises(UnsupportedObservation, match=reason):
        bounded_dfl(torch.zeros(4, 16), torch.tensor(target, dtype=torch.float64),
                    torch.tensor([False, False, True, True]), area)


def test_extreme_finite_logits_remain_finite_in_log_space():
    logits = torch.zeros(4, 16, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        logits[:, 15] = 10000
    loss = bounded_dfl(logits, torch.tensor([2., 2., 2., 2.], dtype=torch.float64),
                       torch.tensor([False, False, True, True]), 16 / .35)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
