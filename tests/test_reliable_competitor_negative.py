"""Synthetic-only contract tests: no dataset, detector, forward, or optimizer.

The oracle is autograd through unreduced BCE, including an unchanged positive
term and a shared per-anchor scale/denominator. L1 equality is local to logits;
it does not assert equality of parameter gradients or optimization trajectories.
"""

from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


MODES = ("reference", "candidate", "control")
INVALID_INPUT = (TypeError, ValueError)


@pytest.fixture(scope="module")
def multipliers():
    # Delayed import permits collection while the main session implements the
    # interface. An absent/broken implementation is an error, never a silent skip.
    return import_module(
        "regionssod.training.reliable_competitor_negative"
    ).negative_multipliers


def anchor_inputs(device="cpu", dtype=torch.float32):
    logits = torch.tensor(
        [[2., -.5, .7, -.2], [-1., 1.2, .3, -.4],
         [.9, -.8, 1.7, .1], [1.5, -.1, -.6, .5],
         [.2, 1.1, -.4, .8], [-.3, .4, 1.3, -.7]],
        device=device, dtype=dtype,
    )
    labels = torch.tensor([0, 1, 2, 3, 2, 3], device=device)
    human = torch.tensor([True, True, False, False, True, True], device=device)
    raw = torch.zeros(6, 4, device=device, dtype=torch.float32)
    # Row 2 is explicitly pseudo even though its target/confidence is one.
    # Row 4 has zero mass and assigned class 2, not all-zero argmax class 0.
    raw[torch.arange(6, device=device), labels] = torch.tensor(
        [.625, .375, 1., .75, 0., 8.708838983295664e-9], device=device,
    )
    return logits, raw.to(dtype).clone(), human, labels, raw


def negative_mask(labels, nc):
    return torch.arange(nc, device=labels.device)[None, :] != labels[:, None]


def bce_gradient(logits, targets, weights=None, *, amp=False, scale=None):
    context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
    with context:
        terms = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if weights is not None:
            terms = terms * weights
        if scale is not None:
            terms = terms * scale
        loss = terms.sum() / 7.25
    gradient, = torch.autograd.grad(loss, logits)
    return terms.detach(), gradient.detach()


@pytest.mark.parametrize("mode", MODES)
def test_exact_weight_contract_and_no_input_mutation(multipliers, mode):
    inputs = anchor_inputs()
    logits, targets, human, labels, raw = inputs
    logits.requires_grad_()
    before = [tensor.detach().clone() for tensor in inputs]
    weights, active = multipliers(*inputs, mode=mode)
    eligible = human & (targets.gather(1, labels[:, None]).squeeze(1) > 0)
    assert active.dtype == torch.bool
    assert active.shape == human.shape and active.device == human.device
    # Reference may report eligibility or an empty action mask.
    assert torch.equal(active, eligible) or (mode == "reference" and not active.any())
    assert weights.shape == logits.shape and weights.device == logits.device
    assert weights.is_floating_point() and torch.isfinite(weights).all()
    assert not weights.requires_grad and weights.grad_fn is None
    assert (weights >= 1).all() and (weights <= 2).all()
    assert all(torch.equal(value, saved) for value, saved in zip(inputs, before))

    expected = torch.ones_like(weights)
    # Independent row-wise oracle uses explicit label exclusion and sigmoid
    # probability ties; it never infers a class from targets.argmax().
    for row in range(len(logits)):
        if not eligible[row] or mode == "reference":
            continue
        classes = [c for c in range(logits.shape[1]) if c != labels[row].item()]
        probabilities = [logits[row, c].detach().float().sigmoid().item() for c in classes]
        best = max(range(len(classes)), key=probabilities.__getitem__)
        if mode == "candidate":
            expected[row, classes[best]] = 2
        else:
            expected[row, classes] = 1 + probabilities[best] / sum(probabilities)
    torch.testing.assert_close(weights, expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("mode", MODES)
def test_human_and_confidence_one_pseudo_have_distinct_actions(multipliers, mode):
    logits = torch.tensor([[.2, 1., -.5], [.2, 1., -.5]], requires_grad=True)
    raw = torch.tensor([[1., 0., 0.], [1., 0., 0.]])
    labels = torch.tensor([0, 0])
    weights, active = multipliers(
        logits, raw, torch.tensor([True, False]), labels, raw, mode=mode,
    )
    _, base = bce_gradient(logits, raw)
    _, changed = bce_gradient(logits, raw, weights)
    assert torch.equal(weights[1], torch.ones_like(weights[1]))
    assert torch.equal(changed[1], base[1])
    assert not active[1]
    if mode != "reference":
        assert active[0] and (changed[0, 1:] > base[0, 1:]).any()


@pytest.mark.parametrize("device,dtype,amp,rtol,atol", [
    pytest.param("cpu", torch.float32, False, 3e-6, 2e-8, id="float32"),
    pytest.param("cuda", torch.float16, True, 5e-3, 3e-5, id="cuda-amp-fp16",
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
def test_real_bce_gradients_positive_untouched_and_rowwise_added_l1(
    multipliers, device, dtype, amp, rtol, atol,
):
    inputs = anchor_inputs(device, dtype)
    logits, targets, human, labels, _ = inputs
    logits.requires_grad_()
    neg = negative_mask(labels, logits.shape[1])
    eligible = human & (targets.gather(1, labels[:, None]).squeeze(1) > 0)
    scale = torch.tensor([1., .375, .8, 1., .6, .5], device=device)[:, None]
    base_terms, base_grad = bce_gradient(logits, targets, amp=amp, scale=scale)
    deltas = {}
    for mode in MODES:
        context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            weights, _ = multipliers(*inputs, mode=mode)
        terms, gradient = bce_gradient(logits, targets, weights, amp=amp, scale=scale)
        assert torch.isfinite(gradient).all() and torch.isfinite(terms).all()
        assert not weights.requires_grad
        # Exact equality protects the actual BCE positive terms, including
        # soft targets whose positive-class derivative can be negative.
        assert torch.equal(weights[~neg], torch.ones_like(weights[~neg]))
        assert torch.equal(terms[~neg], base_terms[~neg])
        assert torch.equal(gradient[~neg], base_grad[~neg])
        assert torch.equal(gradient[~eligible], base_grad[~eligible])
        assert torch.equal(gradient[weights == 1], base_grad[weights == 1])
        if mode == "reference":
            assert torch.equal(terms, base_terms) and torch.equal(gradient, base_grad)
            continue
        delta = gradient.float() - base_grad.float()
        assert (delta[neg] >= 0).all()
        assert torch.count_nonzero(delta[~neg]) == 0
        deltas[mode] = delta
        # Missing detach would contribute a BCE*d(weight)/d(logit) term.
        expected = (weights - 1).float() * logits.detach().float().sigmoid() * scale / 7.25
        torch.testing.assert_close(delta, expected, rtol=rtol, atol=atol)

    candidate_l1 = (deltas["candidate"].abs() * neg).sum(1)
    control_l1 = (deltas["control"].abs() * neg).sum(1)
    torch.testing.assert_close(candidate_l1, control_l1, rtol=rtol, atol=atol)
    assert (candidate_l1[eligible] > 0).all()  # matching two no-ops cannot pass
    assert torch.count_nonzero(candidate_l1[~eligible]) == 0
    assert not torch.equal(deltas["candidate"][0], deltas["control"][0])


@pytest.mark.parametrize("mode", ("candidate", "control"))
@pytest.mark.parametrize("mass", [0., 8.708838983295664e-9, 2.0 ** -24, .5])
def test_fp16_cast_zero_and_smallest_nonzero_mass(multipliers, mode, mass):
    logits = torch.tensor([[2., 1., .5, -.5]], dtype=torch.float16, requires_grad=True)
    raw = torch.tensor([[0., 0., 0., mass]])
    targets = raw.half()
    weights, active = multipliers(
        logits, targets, torch.tensor([True]), torch.tensor([3]), raw, mode=mode,
    )
    assert active.item() == bool(targets[0, 3] > 0)
    assert weights[0, 3] == 1
    _, base = bce_gradient(logits, targets)
    _, changed = bce_gradient(logits, targets, weights)
    assert changed[0, 3] == base[0, 3]
    if targets[0, 3] == 0:
        assert torch.equal(weights, torch.ones_like(weights))
        assert torch.equal(changed, base)
    else:
        assert (weights[0, :3] > 1).any()


@pytest.mark.parametrize("label,expected", [(0, 1), (1, 0), (3, 0)])
@pytest.mark.parametrize("level", [0., 100.])
def test_candidate_probability_ties_choose_lowest_nontrue_id(multipliers, label, expected, level):
    logits = torch.full((1, 4), level)
    if level == 100.:
        # Distinct saturated logits still tie in sigmoid probabilities.
        logits[0] += torch.arange(4)
    raw = torch.zeros_like(logits)
    raw[0, label] = .5
    weights, active = multipliers(
        logits, raw, torch.tensor([True]), torch.tensor([label]), raw, mode="candidate",
    )
    assert active.item()
    assert torch.equal((weights[0] != 1).nonzero().flatten(), torch.tensor([expected]))
    assert weights[0, expected] == 2


@pytest.mark.parametrize("mode", MODES)
def test_no_anchors_returns_empty_shapes_and_zero_differentiable_loss(multipliers, mode):
    logits = torch.empty(0, 4, requires_grad=True)
    raw = torch.empty_like(logits)
    weights, active = multipliers(
        logits, raw, torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.long), raw, mode=mode,
    )
    assert weights.shape == (0, 4) and active.shape == (0,)
    assert active.dtype == torch.bool and not weights.requires_grad
    terms, gradient = bce_gradient(logits, raw, weights)
    assert terms.numel() == 0 and gradient.shape == logits.shape


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("nc", [0, 1])
@pytest.mark.parametrize("n", [0, 1])
def test_fewer_than_two_classes_is_rejected_even_for_empty_input(multipliers, mode, nc, n):
    logits = torch.zeros(n, nc)
    with pytest.raises(INVALID_INPUT):
        multipliers(logits, logits.clone(), torch.ones(n, dtype=torch.bool),
                    torch.zeros(n, dtype=torch.long), logits.clone(), mode=mode)


@pytest.mark.parametrize("mode", ("candidate", "control"))
def test_two_classes_have_identical_candidate_and_control_action(multipliers, mode):
    logits = torch.tensor([[.2, -.7], [.8, -.5]], requires_grad=True)
    raw = torch.tensor([[.5, 0.], [0., .25]])
    labels = torch.tensor([0, 1])
    weights, active = multipliers(logits, raw, torch.ones(2, dtype=torch.bool), labels, raw, mode=mode)
    assert active.all()
    assert torch.equal(weights, torch.tensor([[1., 2.], [2., 1.]]))
    _, base = bce_gradient(logits, raw)
    _, gradient = bce_gradient(logits, raw, weights)
    neg = negative_mask(labels, 2)
    torch.testing.assert_close(gradient[neg], 2 * base[neg], rtol=0, atol=0)
    assert torch.equal(gradient[~neg], base[~neg])


@pytest.mark.parametrize("values", [
    [1000., -1000., -1000., -1000.],
    [-1000., 1000., -1000., 1000.],
    [0., -80., -81., -82.],
    [0., -1000., -80., -1000.],
])
def test_extremes_and_tiny_nonzero_probability_denominators(multipliers, values):
    logits = torch.tensor([values], requires_grad=True)
    raw = torch.tensor([[.5, 0., 0., 0.]])
    human, labels = torch.tensor([True]), torch.tensor([0])
    _, base = bce_gradient(logits, raw)
    probability = logits.detach().sigmoid()[0, 1:]
    total = probability.sum()
    extra = {}
    for mode in ("candidate", "control"):
        weights, active = multipliers(logits, raw, human, labels, raw, mode=mode)
        assert active.item()
        assert torch.isfinite(weights).all() and not weights.requires_grad
        assert (weights >= 1).all() and (weights <= 2).all() and weights[0, 0] == 1
        terms, gradient = bce_gradient(logits, raw, weights)
        assert torch.isfinite(terms).all() and torch.isfinite(gradient).all()
        assert gradient[0, 0] == base[0, 0]
        extra[mode] = (gradient - base)[0, 1:].abs().sum()
        if total > 0 and mode == "control":
            # A blanket clamp_min(1e-12) would erase this ratio for -80 logits.
            expected = torch.full_like(weights[0, 1:], 1 + (probability.max() / total).item())
            torch.testing.assert_close(weights[0, 1:], expected, rtol=2e-6, atol=0)
        if total == 0:
            assert extra[mode] == 0
    torch.testing.assert_close(extra["candidate"], extra["control"], rtol=3e-6, atol=0)
    if total > 0:
        assert extra["candidate"] > 0


@pytest.mark.parametrize("mode", MODES)
def test_noncontiguous_inputs_preserve_explicit_binding(multipliers, mode):
    inputs = anchor_inputs()
    strided = []
    for tensor in inputs:
        backing = torch.zeros((*tensor.shape[:-1], tensor.shape[-1] * 2), dtype=tensor.dtype)
        view = backing[..., ::2]
        view.copy_(tensor)
        assert not view.is_contiguous()
        strided.append(view)
    expected = multipliers(*inputs, mode=mode)
    actual = multipliers(*strided, mode=mode)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("field", ["logits", "targets", "raw_targets"])
@pytest.mark.parametrize("dtype", [torch.long, torch.bool])
def test_nonfloating_logits_or_targets_rejected(multipliers, mode, field, dtype):
    values = dict(zip(("logits", "targets", "is_human", "labels", "raw_targets"), anchor_inputs()))
    values[field] = values[field].to(dtype)
    with pytest.raises(INVALID_INPUT):
        multipliers(**values, mode=mode)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bad", [
    "logits_rank", "target_shape", "raw_shape", "human_column", "human_length",
    "labels_column", "labels_length", "float_labels", "int32_labels", "bool_labels",
    "float_human", "integer_human", "nan_human", "label_negative", "label_out_of_range",
    "nan_logits", "inf_logits", "nan_target", "inf_raw", "nan_raw",
    "negative_raw", "raw_above_one", "offclass_mass", "changed_target", "one_ulp_target",
])
@pytest.mark.parametrize("state", ["positive", "zero", "underflow", "pseudo"])
def test_invalid_input_rejected_before_source_or_quality_filter(multipliers, mode, bad, state):
    dtype = torch.float16 if state == "underflow" else torch.float32
    logits = torch.zeros(1, 3, dtype=dtype)
    mass = {"positive": .5, "zero": 0., "underflow": 8.708838983295664e-9, "pseudo": 1.}[state]
    raw = torch.tensor([[0., 0., mass]])
    values = {"logits": logits, "targets": raw.to(dtype).clone(),
              "is_human": torch.tensor([state != "pseudo"]), "labels": torch.tensor([2]),
              "raw_targets": raw}
    if bad == "logits_rank":
        values["logits"] = logits[0]
    elif bad == "target_shape":
        values["targets"] = values["targets"][:, :2]
    elif bad == "raw_shape":
        values["raw_targets"] = raw[:, :2]
    elif bad in {"human_column", "human_length", "labels_column", "labels_length"}:
        field = "is_human" if bad.startswith("human") else "labels"
        values[field] = values[field][:, None] if bad.endswith("column") else values[field][:0]
    elif bad in {"float_labels", "int32_labels", "bool_labels"}:
        values["labels"] = values["labels"].to(
            {"float_labels": torch.float32, "int32_labels": torch.int32, "bool_labels": torch.bool}[bad]
        )
    elif bad in {"float_human", "integer_human", "nan_human"}:
        values["is_human"] = {"float_human": torch.tensor([1.]),
                              "integer_human": torch.tensor([1]),
                              "nan_human": torch.tensor([float("nan")])}[bad]
    elif bad in {"label_negative", "label_out_of_range"}:
        values["labels"][0] = -1 if bad == "label_negative" else 3
    elif bad in {"nan_logits", "inf_logits", "nan_target", "inf_raw", "nan_raw"}:
        field = {"nan_logits": "logits", "inf_logits": "logits", "nan_target": "targets",
                 "inf_raw": "raw_targets", "nan_raw": "raw_targets"}[bad]
        values[field][0, 2] = float("nan") if bad.startswith("nan") else float("inf")
    elif bad in {"negative_raw", "raw_above_one", "offclass_mass"}:
        raw[0, 0 if bad == "offclass_mass" else 2] = {
            "negative_raw": -.1, "raw_above_one": 1.1, "offclass_mass": 1e-9,
        }[bad]
        values["targets"] = raw.to(dtype).clone()
    elif bad == "changed_target":
        values["targets"][0, 2] = .25
    else:
        values["targets"][0, 2] = torch.nextafter(
            values["targets"][0, 2], torch.tensor(float("inf"), dtype=dtype),
        )
    with pytest.raises(INVALID_INPUT):
        multipliers(**values, mode=mode)


@pytest.mark.parametrize("mode", ["semantic", "strength", "", None])
def test_unknown_mode_rejected(multipliers, mode):
    with pytest.raises(INVALID_INPUT):
        multipliers(*anchor_inputs(), mode=mode)


@pytest.fixture(scope="module")
def source_module():
    return import_module("regionssod.training.reliable_competitor_negative")


def synthetic_source_batch():
    """Metadata only: one human and one confidence-one pseudo, no image tensors."""
    return {
        "cls": torch.tensor([[1.], [2.]]),
        "bboxes": torch.tensor([[.5, .5, .2, .2], [.3, .4, .1, .1]]),
        "batch_idx": torch.tensor([0., 0.]),
        "pseudo_weights": torch.ones(2),
        "object_is_human": torch.tensor([True, False]),
        "object_id": torch.tensor([3, 4]),
        "object_class": torch.tensor([1, 2]),
        "object_table": torch.tensor([[1., 1., 0., 1., 3.], [2., 1., 0., 0., 4.]]),
        "ignore_bboxes": torch.tensor([[.7, .6, .1, .1]]),
        "ignore_batch_idx": torch.tensor([0.]),
    }


def test_batch_validation_preserves_confidence_one_pseudo_source(source_module):
    batch = synthetic_source_batch()
    before = {key: value.clone() for key, value in batch.items()}
    source_module.validate_batch_provenance(batch, 3)
    assert all(torch.equal(batch[key], saved) for key, saved in before.items())
    assert batch["pseudo_weights"].eq(1).all()
    assert torch.equal(batch["object_is_human"], torch.tensor([True, False]))


@pytest.mark.parametrize("corruption", [
    "human_nan", "human_float", "pseudo_promoted", "fractional_class", "class_binding",
    "token_binding", "token_reserved", "token_precision", "table_human_nan",
    "table_human_fractional", "table_ignore", "table_dtype", "duplicate_source",
])
def test_bad_batch_source_fails_before_parent_loss_without_model(source_module, monkeypatch, corruption):
    batch = synthetic_source_batch()
    if corruption in {"human_nan", "human_float"}:
        batch["object_is_human"] = batch["object_is_human"].float()
        if corruption == "human_nan":
            batch["object_is_human"][0] = float("nan")
    elif corruption == "pseudo_promoted":
        batch["object_is_human"][1] = True
    elif corruption == "fractional_class":
        batch["cls"][0, 0] = 1.5
    elif corruption == "class_binding":
        batch["object_class"][0] = 2
    elif corruption == "token_binding":
        batch["object_id"][0] = 5
    elif corruption in {"token_reserved", "token_precision"}:
        token = 2 if corruption == "token_reserved" else 2**24
        batch["object_id"][0] = token
        batch["object_table"][0, 4] = token
    elif corruption in {"table_human_nan", "table_human_fractional"}:
        batch["object_table"][0, 3] = float("nan") if corruption.endswith("nan") else .5
    elif corruption == "table_ignore":
        batch["object_table"][0, 2] = 1
    elif corruption == "table_dtype":
        batch["object_table"] = batch["object_table"].half()
    else:
        # Each row is locally self-consistent but the same token claims two sources.
        batch["object_id"][1] = 3
        batch["object_table"][1, 4] = 3

    def forbidden_parent(*args, **kwargs):
        pytest.fail("invalid provenance reached P1/TAL instead of failing first")

    monkeypatch.setattr(source_module.P1DetectionLoss, "get_assigned_targets_and_loss", forbidden_parent)
    owner = SimpleNamespace(nc=3)
    with pytest.raises(INVALID_INPUT):
        source_module.ReliableCompetitorLoss.get_assigned_targets_and_loss(owner, {}, batch)


def fake_decoding_dataset(source_module, monkeypatch, *, ignored=False):
    """Stub augmentation output; never initialize/load a real dataset or image."""
    dataset = object.__new__(source_module.SourceDataset)
    dataset.data = {"names": {0: "a", 1: "b", 2: "c"}}
    dataset.p1_lookup = torch.tensor([
        [0., 1., 0., 1., 0.], [1., 1., 0., 1., 1.], [2., 1., 0., 1., 2.],
        [1., 1., float(ignored), 0., 3.],
    ])
    sample = {"cls": torch.tensor([[3.]]), "bboxes": torch.tensor([[.5, .5, .2, .2]]),
              "batch_idx": torch.tensor([0.])}
    monkeypatch.setattr(source_module.YOLODataset, "__getitem__", lambda self, index: {
        key: value.clone() for key, value in sample.items()
    })
    return dataset, sample


@pytest.mark.parametrize("token", [float("nan"), float("inf"), -1., 2., 3.5, 4.])
def test_augmented_invalid_scalar_token_is_rejected(source_module, monkeypatch, token):
    dataset, sample = fake_decoding_dataset(source_module, monkeypatch)
    sample["cls"][0, 0] = token
    with pytest.raises(INVALID_INPUT):
        dataset[0]


@pytest.mark.parametrize("ignored", [False, True])
def test_stub_decoder_keeps_confidence_one_pseudo_and_valid_ignore(source_module, monkeypatch, ignored):
    dataset, _ = fake_decoding_dataset(source_module, monkeypatch, ignored=ignored)
    result = dataset[0]
    source_module.validate_batch_provenance(result, 3)
    assert len(result["cls"]) == (0 if ignored else 1)
    assert len(result["ignore_bboxes"]) == int(ignored)
    if not ignored:
        assert result["pseudo_weights"].item() == 1
        assert not result["object_is_human"].item()


@pytest.mark.parametrize("column,value", [
    pytest.param(2, float("nan"), id="ignore-nan"),
    pytest.param(2, .5, id="ignore-fractional"),
    pytest.param(2, -1., id="ignore-negative"),
    pytest.param(2, 2., id="ignore-above-one"),
])
def test_corrupt_ignore_flag_cannot_hide_object_before_validation(source_module, monkeypatch, column, value):
    dataset, _ = fake_decoding_dataset(source_module, monkeypatch)
    dataset.p1_lookup[3, column] = value
    with pytest.raises(INVALID_INPUT):
        result = dataset[0]
        source_module.validate_batch_provenance(result, 3)


@pytest.mark.parametrize("column,value", [
    pytest.param(0, float("nan"), id="class-nan"),
    pytest.param(0, 1.5, id="class-fractional"),
    pytest.param(0, 3., id="class-out-of-range"),
    pytest.param(1, float("nan"), id="confidence-nan"),
    pytest.param(3, float("nan"), id="source-nan"),
    pytest.param(3, .5, id="source-fractional"),
    pytest.param(4, 3.5, id="token-fractional"),
    pytest.param(4, 4., id="token-binding"),
])
def test_ignored_row_still_requires_legal_source_and_token(source_module, monkeypatch, column, value):
    dataset, _ = fake_decoding_dataset(source_module, monkeypatch, ignored=True)
    dataset.p1_lookup[3, column] = value
    with pytest.raises(INVALID_INPUT):
        result = dataset[0]
        source_module.validate_batch_provenance(result, 3)


@pytest.mark.parametrize("field,value", [
    pytest.param("labels", 1.5, id="fractional-label"),
    pytest.param("assigned_ids", .5, id="fractional-object-id"),
    pytest.param("assigned_ids", -.5, id="negative-fractional-object-id"),
])
@pytest.mark.parametrize("mass", [.5, 0., 8.708838983295664e-9])
def test_source_bce_rejects_corrupt_assignment_before_integer_cast(
    source_module, field, value, mass,
):
    # Invoke only the BCE adapter on handcrafted TAL outputs. No TAL/model runs.
    batch = synthetic_source_batch()
    source_module.validate_batch_provenance(batch, 3)
    logits = torch.zeros(1, 1, 3, dtype=torch.float16, requires_grad=True)
    raw = torch.tensor([[[0., mass, 0.]]])
    labels, assigned_ids = torch.tensor([[1.]]), torch.tensor([[0.]])
    (labels if field == "labels" else assigned_ids)[0, 0] = value
    owner = SimpleNamespace(current_batch=batch, mode="candidate", diagnostics=False,
                            assigner=SimpleNamespace(assigned=(
                                labels, torch.zeros(1, 1, 4), raw,
                                torch.tensor([[True]]), assigned_ids,
                            )))
    adapter = source_module.SourceBCE(torch.nn.BCEWithLogitsLoss(reduction="none"), owner)
    with pytest.raises(INVALID_INPUT):
        adapter(logits, raw.half())


@pytest.mark.parametrize("field,value", [
    pytest.param("ignore_bboxes", float("nan"), id="ignore-box-nan"),
    pytest.param("ignore_batch_idx", .5, id="ignore-image-fractional"),
    pytest.param("ignore_batch_idx", -1., id="ignore-image-negative"),
    pytest.param("ignore_batch_idx", float("nan"), id="ignore-image-nan"),
])
def test_batch_provenance_rejects_invalid_ignore_geometry_or_image_binding(source_module, field, value):
    batch = synthetic_source_batch()
    batch[field].view(-1)[0] = value
    with pytest.raises(INVALID_INPUT):
        source_module.validate_batch_provenance(batch, 3)
