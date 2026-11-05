"""Pure synthetic transport tests: no files, datasets, checkpoints or models."""
from copy import deepcopy
from fractions import Fraction
import random

import numpy as np
import pytest
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import (
    Albumentations, Compose, CopyPaste, CutMix, Format, LetterBox, MixUp,
    Mosaic, RandomFlip, RandomPerspective, v8_transforms,
)
from ultralytics.utils.instance import Instances

from regionssod.training.bounded_retention_transport import (
    ObservationFormat, ObservationInstances, ObservationMosaic,
    ObservationRandomPerspective, copy_observations, make_observations,
    outward_inverse_range, replace_observation_transforms,
)


SOURCE_KEYS = ("object_id", "occurrence_id", "human", "original_xywh",
               "original_serialized_intervals", "original_unknown", "original_area_cap", "payload")


def observations(boxes, occurrence=1):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    n = len(boxes)
    intervals = np.stack((boxes - 1e-10, boxes + 1e-10), axis=-1)
    return make_observations(
        object_id=np.arange(n), occurrence_id=np.full(n, occurrence), human=np.arange(n) % 2 == 0,
        original_xywh=boxes, original_serialized_intervals=intervals,
        original_unknown=np.tile([True, False, False, True], (n, 1)),
        original_area_cap=boxes[:, 2] * boxes[:, 3] / .35,
        payload=np.arange(n * 6).reshape(n, 2, 3),
    )


def instances(boxes, transport=True):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    base = Instances(boxes.copy(), np.zeros((0, 1000, 2), dtype=np.float32),
                     bbox_format="xywh", normalized=True)
    return ObservationInstances.from_instances(base, observations(boxes)) if transport else base


def labels(boxes, transport=True, h=60, w=80):
    return {"img": (np.arange(h * w * 3).reshape(h, w, 3) % 251).astype(np.uint8),
            "cls": np.arange(len(boxes), dtype=np.float32).reshape(-1, 1),
            "instances": instances(boxes, transport), "im_file": "synthetic",
            "ori_shape": (h, w), "resized_shape": (h, w)}


def assert_bits(a, b):
    if isinstance(a, torch.Tensor):
        a, b = a.cpu().numpy(), b.cpu().numpy()
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def rng_state():
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()


def assert_rng_equal(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0]
    np.testing.assert_array_equal(a[1][1], b[1][1])
    assert a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_signed_geometry_clipping_and_source_fields_are_independent():
    boxes = [[.1, .5, .2, .4], [.8, .7, .2, .2]]
    obs, base = instances(boxes), instances(boxes, False)
    original = copy_observations(obs.obs)
    operations = [
        ("denormalize", (81, 60)), ("denormalize", (81, 60)),
        ("scale", (1.5, .75)), ("add_padding", (-10, 3)),
        ("convert_bbox", ("xyxy",)), ("clip", (80, 40)),
        ("fliplr", (80,)), ("flipud", (40,)),
    ]
    for method, args in operations:
        getattr(obs, method)(*args)
        getattr(base, method)(*args)
        assert_bits(obs.bboxes, base.bboxes)
    np.testing.assert_allclose(obs.obs["affine"], np.tile([-121.5, -45, 90, 37], (2, 1)))
    np.testing.assert_allclose(obs.obs["current_window"], [[0, 0, 80, 37]] * 2)
    x = original["original_xywh"]
    lo, hi = x[:, :2] - x[:, 2:] / 2, x[:, :2] + x[:, 2:] / 2
    a = lo * [-121.5, -45] + [90, 37]
    b = hi * [-121.5, -45] + [90, 37]
    np.testing.assert_allclose(obs.obs["unclipped"], np.concatenate((np.minimum(a, b), np.maximum(a, b)), 1))
    assert obs.obs["unclipped"][0, 2] > 80  # known coordinate survives clipping
    for key in SOURCE_KEYS:
        assert_bits(obs.obs[key], original[key])
    pixel = copy_observations(obs.obs)
    obs.normalize(80, 40)
    once = copy_observations(obs.obs)
    obs.normalize(80, 40)
    for key in once:
        assert_bits(once[key], obs.obs[key])
    obs.denormalize(80, 40)
    for key in ("affine", "current_window", "unclipped"):
        np.testing.assert_allclose(obs.obs[key], pixel[key])
    # The optional cap remains in original units; only this derived cap scales.
    np.testing.assert_allclose(obs.obs["original_area_cap"] * np.abs(obs.obs["affine"][:, 0] * obs.obs["affine"][:, 1]),
                               original["original_area_cap"] * 121.5 * 45)


def test_initial_resize_is_actual_width_height_not_ideal_long_side_ratio():
    obj = instances([[.5, .5, .4, .2]])
    obj.denormalize(640, 399)  # actual integer dimensions, irrespective of source aspect ratio
    np.testing.assert_array_equal(obj.obs["affine"], [[640, 399, 0, 0]])
    np.testing.assert_array_equal(obj.obs["current_window"], [[0, 0, 640, 399]])


def test_selection_concatenation_zero_filter_and_empty_rows():
    obj = instances([[.5, .5, .2, .2], [1.5, .5, .2, .2], [.2, .2, .1, .1]])
    obj.denormalize(100, 80)
    before = copy_observations(obj.obs)
    obj.clip(100, 80)
    good = obj.remove_zero_area_boxes()
    np.testing.assert_array_equal(good, [True, False, True])
    for key in SOURCE_KEYS:
        assert_bits(obj.obs[key], before[key][good])
    for index in (0, np.int64(1), slice(None, None, -1), np.array([False, True]), np.array([1, 0, 1])):
        selected = obj[index]
        assert isinstance(selected, ObservationInstances)
        selected.obs["affine"][:] = 999
        assert not np.any(obj.obs["affine"] == 999)
    joined = ObservationInstances.concatenate([obj, obj[:0], obj[::-1]])
    np.testing.assert_array_equal(joined.obs["object_id"], [0, 2, 2, 0])
    joined.obs["payload"][:] = -99
    assert not np.any(obj.obs["payload"] == -99)
    empty = ObservationInstances.concatenate([obj[:0], obj[:0]])
    assert empty.obs["original_serialized_intervals"].shape == (0, 4, 2)
    np.testing.assert_array_equal(empty.remove_zero_area_boxes(), np.array([], dtype=bool))


def test_bbox_only_scratch_does_not_transport_observations():
    obj = instances([[.5, .5, .2, .2]])
    original = copy_observations(obj.obs)
    obj.scale(2, 3, bbox_only=True)
    for key in original:
        assert_bits(obj.obs[key], original[key])


def test_float64_source_is_not_recovered_from_rounded_or_clipped_bboxes():
    source = np.array([[.1234567890123, .51, .2469135780246, .3]], dtype=np.float64)
    base = instances(source, False)
    obj = ObservationInstances.from_instances(base, observations(source))
    original = copy_observations(obj.obs)
    obj.denormalize(613, 397)
    obj.add_padding(-15, 9)
    obj.clip(600, 390)
    obj.fliplr(600)
    assert obj.obs["original_xywh"].dtype == np.float64
    assert obj.obs["original_xywh"][0, 0] != float(base.bboxes[0, 0])
    for key in SOURCE_KEYS:
        assert_bits(original[key], obj.obs[key])
    expected_right = 615.0  # known source left=0 is outside the clipped image after reflection
    assert obj.obs["unclipped"][0, 2] == expected_right
    np.testing.assert_array_equal(obj.obs["original_unknown"], original["original_unknown"])


def test_external_physical_oracles_and_arbitrary_fields_follow_positive_rows():
    obj = instances([[.5, .5, .3, .3], [1.5, .5, .2, .2], [.2, .2, .1, .1]])
    oracle0, oracle2 = object(), object()
    sidecar = dict(obj.obs, physical=np.array([oracle0, None, oracle2], dtype=object),
                   source_object_id=np.array([101, 102, 103], dtype=np.int64),
                   tensor_payload=np.arange(24).reshape(3, 2, 2, 2))
    obj = ObservationInstances.from_instances(obj, sidecar)
    obj.denormalize(80, 60)
    obj.clip(80, 60)
    obj.remove_zero_area_boxes()
    combined = ObservationInstances.concatenate([obj, obj[::-1]])
    result = labels([], h=60, w=80)
    result["instances"] = combined
    result["cls"] = np.zeros((len(combined), 1), dtype=np.float32)
    out = ObservationFormat()(result)["obs"]
    np.testing.assert_array_equal(out["source_object_id"], [101, 103, 103, 101])
    assert list(out["physical"]) == [oracle0, oracle2, oracle2, oracle0]
    np.testing.assert_array_equal(out["tensor_payload"], sidecar["tensor_payload"][[0, 2, 2, 0]])
    out["physical"][0] = None
    assert combined.obs["physical"][0] is oracle0


def test_area_cap_is_optional_and_copy_does_not_share_numeric_storage():
    obs = observations([[.5, .5, .2, .2]])
    obs.pop("original_area_cap")
    obj = ObservationInstances.from_instances(instances([[.5, .5, .2, .2]], False), obs)
    duplicate = deepcopy(obj)
    duplicate.denormalize(80, 60)
    duplicate.obs["original_xywh"][:] = 0
    np.testing.assert_array_equal(obj.obs["affine"], [[1, 1, 0, 0]])
    assert obj.obs["original_xywh"][0, 0] == .5


class SyntheticDataset:
    """Deterministic generated pixels; get_image_and_label never opens a file."""
    cache = None
    buffer = [0]  # force repeated source, distinct occurrences
    use_obb = use_keypoints = False
    data = {}

    def __init__(self, transport, imgsz=64):
        self.transport = transport
        self.imgsz = imgsz
        self.calls = 0

    def __len__(self):
        return 1

    def get_image_and_label(self, index):
        self.calls += 1
        result = labels([[.5, .5, .6, .6], [.03, .5, .06, .3],
                         [.85, .85, .25, .25], [.5, .5, .005, .005]],
                        self.transport, h=int(self.imgsz * 431 / 640), w=self.imgsz)
        if self.transport:
            result["instances"].obs["occurrence_id"][:] = self.calls
        return result


def test_mosaic_repeated_source_occurrences_keep_per_patch_affine_and_windows():
    seed_all(5)
    d0 = SyntheticDataset(False)
    original = Mosaic(d0, imgsz=64)(d0.get_image_and_label(0))
    state0 = rng_state()
    seed_all(5)
    d1 = SyntheticDataset(True)
    transformed = ObservationMosaic(d1, imgsz=64)(d1.get_image_and_label(0))
    assert_rng_equal(state0, rng_state())
    assert d0.calls == d1.calls == 4
    assert_bits(original["img"], transformed["img"])
    assert_bits(original["cls"], transformed["cls"])
    assert_bits(original["instances"].bboxes, transformed["instances"].bboxes)
    obs = transformed["instances"].obs
    assert len(np.unique(obs["occurrence_id"])) == 4
    offsets = []
    for occurrence in np.unique(obs["occurrence_id"]):
        selected = obs["occurrence_id"] == occurrence
        affine = obs["affine"][selected]
        np.testing.assert_array_equal(affine, np.tile(affine[0], (selected.sum(), 1)))
        np.testing.assert_array_equal(affine[0, :2], [64, 43])
        offsets.append(tuple(affine[0, 2:]))
        tx, ty = affine[0, 2:]
        expected_window = np.array([tx, ty, tx + 64, ty + 43]).clip(0, 128)
        np.testing.assert_array_equal(obs["current_window"][selected], np.tile(expected_window, (selected.sum(), 1)))
    assert len(set(offsets)) == 4


def test_rp_clip_filter_and_matrix_transport_match_original_bitwise():
    boxes = [[.05, .5, .1, .5], [.5, .5, .5, .5], [.99, .5, .01, .01], [.5, .99, .01, .01]]
    M = np.array([[1.25, 0, -12], [0, 1.25, -10], [0, 0, 1]], dtype=np.float32)
    params = {"M": M, "scale": 1.25, "orig_shape": (60, 80), "size": (70, 60)}
    plain, observed = labels(boxes, False), labels(boxes)
    RandomPerspective().apply_instances(plain, params)
    ObservationRandomPerspective().apply_instances(observed, params)
    assert_bits(plain["instances"].bboxes, observed["instances"].bboxes)
    assert_bits(plain["cls"], observed["cls"])
    obj = observed["instances"]
    assert len(obj) == 1  # left entirely clipped, tiny boxes fail size/area filter
    np.testing.assert_array_equal(obj.obs["object_id"], [1])
    np.testing.assert_array_equal(obj.obs["affine"], [[100, 75, -12, -10]])
    np.testing.assert_array_equal(obj.obs["current_window"], [[0, 0, 70, 60]])
    np.testing.assert_allclose(obj.obs["unclipped"], [[13, 8.75, 63, 46.25]])


def test_rp_all_rows_removed_can_still_be_formatted():
    params = {"M": np.array([[1, 0, 1000], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
              "scale": 1., "orig_shape": (60, 80), "size": (80, 60)}
    data = labels([[.5, .5, .5, .5]])
    data = ObservationRandomPerspective().apply_instances(data, params)
    assert len(data["instances"]) == 0
    data = ObservationFormat()(data)
    assert data["obs"]["affine"].shape == (0, 4)
    assert data["obs"]["original_serialized_intervals"].shape == (0, 4, 2)
    assert data["bboxes"].shape == (0, 4)


@pytest.mark.parametrize("empty", [False, True])
def test_format_exports_pixel_copy_without_changing_rng_or_original_outputs(empty):
    boxes = [] if empty else [[.5, .5, .2, .3]]
    plain, observed = labels(boxes, False), labels(boxes)
    inst = observed["instances"]
    seed_all(13)
    a = Format(bbox_format="xywh", normalize=True)(plain)
    state = rng_state()
    seed_all(13)
    b = ObservationFormat(bbox_format="xywh", normalize=True)(observed)
    assert_rng_equal(state, rng_state())
    for key in ("img", "cls", "bboxes", "batch_idx"):
        assert_bits(a[key], b[key])
    np.testing.assert_array_equal(b["obs"]["affine"], np.tile([80, 60, 0, 0], (len(boxes), 1)))
    b["obs"]["affine"][:] = 777
    assert not np.any(inst.obs["affine"] == 777)


@pytest.mark.parametrize("mosaic", [0., 1.])
@pytest.mark.parametrize("seed", [0, 1, 7, 22, 319])
@pytest.mark.parametrize("imgsz", [64, 640])
def test_full_default_detect_chain_preserves_bits_rng_and_row_alignment(mosaic, seed, imgsz):
    def run(transport):
        seed_all(seed)
        dataset = SyntheticDataset(transport, imgsz=imgsz)
        hyp = get_cfg(overrides={"imgsz": imgsz, "mosaic": mosaic})
        pipeline = v8_transforms(dataset, imgsz, hyp)
        pipeline.append(Format(bbox_format="xywh", normalize=True, bgr=hyp.bgr))
        if transport:
            before = rng_state()
            pipeline = replace_observation_transforms(pipeline)
            assert_rng_equal(before, rng_state())
        output = pipeline(dataset.get_image_and_label(0))
        return output, rng_state(), dataset.calls
    a, state_a, calls_a = run(False)
    b, state_b, calls_b = run(True)
    assert_rng_equal(state_a, state_b)
    assert calls_a == calls_b
    for key in ("img", "cls", "bboxes", "batch_idx"):
        assert_bits(a[key], b[key])
    assert len(b["obs"]["object_id"]) == len(b["bboxes"])
    np.testing.assert_array_equal(b["obs"]["object_id"], b["cls"].numpy().reshape(-1).astype(int))
    np.testing.assert_array_equal(b["obs"]["human"], b["obs"]["object_id"] % 2 == 0)
    for key in ("affine", "augmentation_clip_window"):
        nominal, bounds = b["obs"][key], b["obs"][key + "_bounds"]
        assert bounds.shape == (len(b["bboxes"]), 4, 2)
        assert bounds.dtype == np.float64 and not np.isnan(bounds).any()
        assert np.all(bounds[..., 0] <= nominal) and np.all(nominal <= bounds[..., 1])
    scale_bounds = b["obs"]["affine_bounds"][:, :2]
    assert not np.any((scale_bounds[..., 0] <= 0) & (scale_bounds[..., 1] >= 0))


def test_replacement_preserves_aliases_p0_nodes_dataset_and_original_graph():
    dataset = SyntheticDataset(True)
    pre = Compose([Mosaic(dataset, imgsz=64, p=0), RandomPerspective()])
    chain = Compose([pre, MixUp(dataset, pre_transform=pre, p=0), CutMix(dataset, pre_transform=pre, p=0),
                     CopyPaste(dataset, p=0), RandomFlip(p=0), Format()])
    before = rng_state()
    replaced = replace_observation_transforms(chain)
    assert_rng_equal(before, rng_state())
    assert len(replaced.transforms) == len(chain.transforms)
    assert replaced.transforms[0] is replaced.transforms[1].pre_transform
    assert replaced.transforms[0] is replaced.transforms[2].pre_transform
    assert type(chain.transforms[0].transforms[0]) is Mosaic
    assert type(replaced.transforms[0].transforms[0]) is ObservationMosaic
    assert replaced.transforms[0].transforms[0].dataset is dataset
    assert replaced.transforms[0].transforms[0].p == 0
    again = replace_observation_transforms(replaced)
    assert isinstance(again.transforms[-1], ObservationFormat)


@pytest.mark.parametrize("kwargs", [{"degrees": 1}, {"shear": .01}, {"perspective": .001}])
def test_reject_unsupported_perspective_configuration(kwargs):
    with pytest.raises(ValueError, match="unsupported"):
        replace_observation_transforms(Compose([RandomPerspective(**kwargs)]))


@pytest.mark.parametrize("M", [
    [[1, .01, 0], [0, 1, 0], [0, 0, 1]],
    [[1, 0, 0], [0, 1, 0], [.001, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[1, 0, float("nan")], [0, 1, 0], [0, 0, 1]],
])
def test_reject_unsupported_runtime_matrix(M):
    params = {"M": np.array(M), "scale": 1, "orig_shape": (60, 80), "size": (80, 60)}
    with pytest.raises(ValueError, match="axis-aligned"):
        ObservationRandomPerspective().apply_instances(labels([[.5, .5, .2, .2]]), params)


def test_fail_closed_spatial_albumentations_and_unknown_nodes_even_under_p0():
    # Deliberately do not import/instantiate an external spatial pipeline.
    spatial = object.__new__(Albumentations)
    spatial.contains_spatial, spatial.p = True, 0
    with pytest.raises(ValueError, match="spatial Albumentations"):
        replace_observation_transforms(Compose([spatial]))
    with pytest.raises(ValueError, match="unsupported transform"):
        replace_observation_transforms(Compose([lambda x: x]))
    for cls in (MixUp, CutMix, CopyPaste):
        with pytest.raises(ValueError, match="unsupported"):
            replace_observation_transforms(cls(SyntheticDataset(True), p=.1))
    with pytest.raises(ValueError, match="rotation"):
        replace_observation_transforms(MixUp(SyntheticDataset(True), p=0,
                                            pre_transform=Compose([RandomPerspective(degrees=1)])))


def test_letterbox_supported_without_extra_random_calls():
    a, b = labels([[.5, .5, .2, .2]], False), labels([[.5, .5, .2, .2]])
    transform = LetterBox(new_shape=(96, 96))
    a = transform(a)
    b = replace_observation_transforms(transform)(b)
    assert_bits(a["img"], b["img"])
    assert_bits(a["instances"].bboxes, b["instances"].bboxes)
    np.testing.assert_allclose(b["instances"].obs["affine"], [[96, 72, 0, 12]])


def test_invalid_schema_and_untracked_updates_fail_closed():
    obj = instances([[.5, .5, .2, .2]])
    with pytest.raises(ValueError, match="row-aligned"):
        ObservationInstances.from_instances(obj, dict(obj.obs, broken=np.zeros((2, 3))))
    with pytest.raises(ValueError, match="schemas differ"):
        other = ObservationInstances.from_instances(obj, dict(obj.obs, extra=np.ones(1)))
        ObservationInstances.concatenate([obj, other])
    with pytest.raises(TypeError, match="every concatenated"):
        ObservationInstances.concatenate([obj, instances([[.5, .5, .2, .2]], False)])
    with pytest.raises(NotImplementedError, match="untracked"):
        obj.update(obj.bboxes)
    with pytest.raises(ValueError, match="keypoints"):
        ObservationInstances(obj.bboxes, keypoints=np.zeros((1, 4, 3)), obs=obj.obs)


def exact_float(value):
    """Exact rational value of a sampled binary floating point coefficient."""
    return Fraction.from_float(float(value))


def assert_outer_enclosure(bounds, exact_values):
    """No float tolerance: compare binary endpoints to exact Fractions."""
    bounds = np.asarray(bounds).reshape(-1, 2)
    assert len(bounds) == len(exact_values)
    for (lo, hi), value in zip(bounds, exact_values):
        if value == -np.inf:
            assert lo == -np.inf
        elif value == np.inf:
            assert hi == np.inf
        else:
            assert lo == -np.inf or exact_float(lo) <= value, (lo, value, hi)
            assert hi == np.inf or value <= exact_float(hi), (lo, value, hi)


def exact_transform(affine, crop, sx, sy, tx=Fraction(0), ty=Fraction(0)):
    """Independent exact-real affine/crop reference; no NumPy arithmetic."""
    transformed = [affine[0] * sx, affine[1] * sy,
                   affine[2] * sx + tx, affine[3] * sy + ty]
    window = list(crop)
    for axis, scale, offset in ((0, sx, tx), (1, sy, ty)):
        lo, hi = (axis, axis + 2) if scale > 0 else (axis + 2, axis)
        window[axis], window[axis + 2] = crop[lo] * scale + offset, crop[hi] * scale + offset
    return transformed, window


def test_initial_bounds_are_exact_and_crop_domain_is_not_source_tile():
    obj = instances([[.5, .5, .2, .2]])
    np.testing.assert_array_equal(obj.obs["affine_bounds"][0], [[1, 1], [1, 1], [0, 0], [0, 0]])
    obj.denormalize(640, 431)
    obj.add_padding(-32, 7)
    np.testing.assert_array_equal(obj.obs["augmentation_clip_window"][0], [-np.inf, -np.inf, np.inf, np.inf])
    np.testing.assert_array_equal(obj.obs["augmentation_clip_window_bounds"][0],
                                  [[-np.inf, -np.inf], [-np.inf, -np.inf], [np.inf, np.inf], [np.inf, np.inf]])
    obj.clip(1280, 1280)
    np.testing.assert_array_equal(obj.obs["augmentation_clip_window"][0], [0, 0, 1280, 1280])
    np.testing.assert_array_equal(obj.obs["augmentation_clip_window_bounds"][0],
                                  [[0, 0], [0, 0], [1280, 1280], [1280, 1280]])
    np.testing.assert_array_equal(obj.obs["current_window"][0], [0, 7, 608, 438])
    before = copy_observations(obj.obs)
    obj.clip(1280, 1280)
    assert_bits(obj.obs["augmentation_clip_window_bounds"], before["augmentation_clip_window_bounds"])


def test_fraction_proves_enclosure_after_composition_normalize_flips_and_clips():
    obj = instances([[.4, .5, .3, .3]])
    source = copy_observations(obj.obs)
    affine = list(map(Fraction, (1, 1, 0, 0)))
    crop = [-np.inf, -np.inf, np.inf, np.inf]

    def check():
        assert_outer_enclosure(obj.obs["affine_bounds"][0], affine)
        assert_outer_enclosure(obj.obs["augmentation_clip_window_bounds"][0], crop)
        for key in ("affine", "augmentation_clip_window"):
            nominal, bounds = obj.obs[key], obj.obs[key + "_bounds"]
            assert np.all(nominal >= bounds[..., 0]) and np.all(nominal <= bounds[..., 1])

    obj.denormalize(613, 397)
    affine, crop = exact_transform(affine, crop, Fraction(613), Fraction(397))
    check()
    obj.add_padding(-17, 23)
    affine, crop = exact_transform(affine, crop, Fraction(1), Fraction(1), Fraction(-17), Fraction(23))
    obj.clip(1280, 1280)
    crop = [max(crop[0], 0), max(crop[1], 0), min(crop[2], 1280), min(crop[3], 1280)]
    check()
    sx, sy = np.float32(.7134567), np.float32(1.2134567)
    obj.scale(sx, sy)
    affine, crop = exact_transform(affine, crop, exact_float(sx), exact_float(sy))
    check()
    obj.add_padding(.123456789, -.314159265)
    affine, crop = exact_transform(affine, crop, Fraction(1), Fraction(1), exact_float(.123456789), exact_float(-.314159265))
    check()
    obj.fliplr(640)
    affine, crop = exact_transform(affine, crop, Fraction(-1), Fraction(1), Fraction(640))
    check()
    obj.flipud(480)
    affine, crop = exact_transform(affine, crop, Fraction(1), Fraction(-1), Fraction(0), Fraction(480))
    obj.clip(640, 480)
    crop = [max(crop[0], 0), max(crop[1], 0), min(crop[2], 640), min(crop[3], 480)]
    check()
    for _ in range(12):
        obj.normalize(641, 479)
        # Exact rational division is deliberately stronger than using the
        # rounded nominal 1/641 and 1/479 as exact reference coefficients.
        affine, crop = exact_transform(affine, crop, Fraction(1, 641), Fraction(1, 479))
        check()
        obj.denormalize(641, 479)
        affine, crop = exact_transform(affine, crop, Fraction(641), Fraction(479))
        check()
    for key in SOURCE_KEYS:
        assert_bits(source[key], obj.obs[key])
    for axis in (0, 1):
        for atom in (-64., 0., 28., 128.):
            inv = outward_inverse_range([atom, atom], obj.obs["affine_bounds"][0, axis],
                                         obj.obs["affine_bounds"][0, axis + 2])
            assert_outer_enclosure(inv, [(exact_float(atom) - affine[axis + 2]) / affine[axis]])


def test_rp_intermediate_crop_has_independent_domain_and_certified_bounds():
    data = labels([[.15, .5, .3, .5]], h=64, w=64)
    obj = data["instances"]
    obj.denormalize(64, 64)
    obj.add_padding(-4, 32)
    obj.clip(128, 128)
    unknown = obj.obs["original_unknown"].copy()
    matrix = np.array([[.5, 0, 6], [0, .5, 0], [0, 0, 1]], dtype=np.float32)
    params = {"M": matrix, "scale": .5, "orig_shape": (128, 128), "size": (64, 64)}
    out = ObservationRandomPerspective().apply_instances(data, params)["instances"]
    np.testing.assert_array_equal(out.obs["augmentation_clip_window"], [[6, 0, 64, 64]])
    np.testing.assert_array_equal(out.obs["current_window"], [[6, 16, 36, 48]])
    np.testing.assert_array_equal(out.obs["original_unknown"], unknown)
    assert_outer_enclosure(out.obs["affine_bounds"][0], list(map(Fraction, [32, 32, 4, 16])))
    assert_outer_enclosure(out.obs["augmentation_clip_window_bounds"][0], list(map(Fraction, [6, 0, 64, 64])))
    # There is a certified positive gap: the old source left boundary lies
    # outside the retained crop. The consumer can resolve U without an epsilon.
    assert out.obs["affine_bounds"][0, 2, 1] < out.obs["augmentation_clip_window_bounds"][0, 0, 0]


def test_same_affine_and_tile_window_can_have_different_clip_domains():
    def path(pad, offset):
        obj = instances([[.2, .5, .4, .2]])
        obj.denormalize(64, 64)
        obj.add_padding(pad, 0)
        obj.clip(128, 128)
        obj._transform_observations(.5, .5, offset, 0)
        obj.clip(64, 64)
        return obj.obs
    first, second = path(28, 6), path(38, 1)
    assert_bits(first["affine"], second["affine"])
    assert_bits(first["current_window"], second["current_window"])
    np.testing.assert_array_equal(first["augmentation_clip_window"], [[6, 0, 64, 64]])
    np.testing.assert_array_equal(second["augmentation_clip_window"], [[1, 0, 64, 64]])
    assert first["augmentation_clip_window_bounds"][0, 0, 0] > 4
    assert second["augmentation_clip_window_bounds"][0, 0, 1] < 4


def test_empty_crop_domain_is_not_resurrected_by_flip_or_repeated_clip():
    obj = instances([[.5, .5, .2, .2]])
    obj.denormalize(10, 10)
    obj.clip(10, 10)
    obj.add_padding(20, 0)
    obj.clip(10, 10)
    assert obj.obs["augmentation_clip_window"][0, 0] > obj.obs["augmentation_clip_window"][0, 2]
    obj.fliplr(10)
    obj.clip(10, 10)
    crop, bounds = obj.obs["augmentation_clip_window"][0], obj.obs["augmentation_clip_window_bounds"][0]
    assert crop[0] > crop[2]
    assert bounds[0, 0] > bounds[2, 1]


def test_letterbox_bounds_cover_actual_sampled_ratio_and_padding():
    obj = labels([[.5, .5, .2, .2]])
    out = replace_observation_transforms(LetterBox(new_shape=(97, 97)))(obj)["instances"]
    ratio = exact_float(97 / 80)
    # Shape 60x80 -> unpadded 73x97, padding top=12, using binary ratio 97/80.
    assert_outer_enclosure(out.obs["affine_bounds"][0],
                           [Fraction(80) * ratio, Fraction(60) * ratio, Fraction(0), Fraction(12)])
    np.testing.assert_array_equal(out.obs["augmentation_clip_window"][0], [-np.inf, -np.inf, np.inf, np.inf])


@pytest.mark.parametrize("scale", [[.13, .17], [-.17, -.13]])
def test_inverse_outer_enclosure_of_all_interval_corners_and_broadcasting(scale):
    coordinates = np.array([[.1, .7], [128., 129.], [-1e-7, 1e-7]])
    offset = np.array([-.31, .19])
    result = outward_inverse_range(coordinates, scale, offset)
    for row in range(3):
        exact = [(exact_float(y) - exact_float(t)) / exact_float(s)
                 for y in coordinates[row] for t in offset for s in scale]
        assert_outer_enclosure(np.repeat(result[row][None, :], len(exact), axis=0), exact)
    assert result.shape == (3, 2)


def test_inverse_supports_half_lines_signed_scale_underflow_and_overflow():
    result = outward_inverse_range([-np.inf, 0], [-3, -2], [1, 2])
    assert_outer_enclosure(result, [Fraction(1, 3)])
    assert result[1] == np.inf
    smallest = np.nextafter(np.float64(0), np.float64(1))
    result = outward_inverse_range([smallest, smallest], [2., 2.], [0., 0.])
    assert_outer_enclosure(result, [exact_float(smallest) / 2])
    largest = np.finfo(np.float64).max
    result = outward_inverse_range([largest, largest], [1., 1.], [-largest, -largest])
    assert_outer_enclosure(result, [2 * exact_float(largest)])
    assert result[1] == np.inf


@pytest.mark.parametrize("scale", [[-1, 1], [0, 1], [-1, 0], [0, 0], [1, np.inf], [np.nan, 1]])
def test_inverse_rejects_zero_touching_crossing_or_nonfinite_scale(scale):
    with pytest.raises(ValueError):
        outward_inverse_range([1., 2.], scale, [0., 0.])


@pytest.mark.parametrize("coordinate,offset", [
    ([2., 1.], [0., 0.]), ([np.nan, 2.], [0., 0.]),
    ([1., 2.], [0., np.inf]), ([1., 2.], [1., 0.]),
])
def test_inverse_rejects_invalid_coordinate_or_offset_intervals(coordinate, offset):
    with pytest.raises(ValueError):
        outward_inverse_range(coordinate, [1., 1.], offset)


def test_new_bounds_survive_selection_concatenation_format_and_empty_shapes():
    obj = instances([[.5, .5, .2, .2], [.3, .5, .1, .1]])
    obj.denormalize(80, 60)
    obj.clip(80, 60)
    obj.fliplr(80)
    selected = ObservationInstances.concatenate([obj[1], obj[:0], obj[0]])
    sample = labels([])
    sample["instances"], sample["cls"] = selected, np.zeros((2, 1), np.float32)
    output = ObservationFormat()(sample)["obs"]
    for key in ("affine_bounds", "augmentation_clip_window", "augmentation_clip_window_bounds"):
        assert_bits(output[key], obj.obs[key][[1, 0]])
        assert not np.shares_memory(output[key], selected.obs[key])
    empty = obj[:0]
    assert empty.obs["affine_bounds"].shape == (0, 4, 2)
    assert empty.obs["augmentation_clip_window_bounds"].shape == (0, 4, 2)


@pytest.mark.parametrize("key", ["affine_bounds", "augmentation_clip_window_bounds"])
def test_invalid_bounds_are_not_accepted_as_certificates(key):
    obj = instances([[.5, .5, .2, .2]])
    bad = copy_observations(obj.obs)
    bad[key][0, 0] = [2., 1.]
    with pytest.raises(ValueError, match="reversed"):
        ObservationInstances.from_instances(obj, bad)
