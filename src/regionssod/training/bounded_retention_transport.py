"""Detection-only observation transport for Ultralytics 8.4.172 (AGPL-3.0).

No dataset, source certificate, loss, or training policy is implemented here.
The caller supplies object/occurrence identities and certified source arrays.
Ordinary boxes follow upstream arithmetic; float64 observations are a sidecar.
The two local instance-building methods follow upstream augment.py, retaining
its order, clipping, and filtering. No random draws are added or removed.

Enclosure argument (IEEE binary64, gradual underflow): each exact basic-operation
result lies between the neighbors of its correctly rounded binary64 value. Interval
addition uses endpoint monotonicity; multiplication/division use all corners
(division excludes zero). Thus each composed affine/crop endpoint encloses its
exact-real value by induction. Crop intersection uses monotone max/min, and
negative scales swap endpoints. These are outer bounds, not an exact inverse
or a joint feasibility certificate; no empirical epsilon enters the argument.
"""
from __future__ import annotations

from copy import copy
from typing import Mapping

import numpy as np
from ultralytics.data.augment import (
    Albumentations, Compose, CopyPaste, CutMix, Format, LetterBox, MixUp,
    Mosaic, RandomFlip, RandomHSV, RandomPerspective,
)
from ultralytics.utils.instance import Instances


def copy_observations(obs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Copy every row-aligned array, including arbitrary caller-owned fields."""
    return {key: np.array(value, copy=True) for key, value in obs.items()}


def _interval(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 0 or value.shape[-1] != 2 or np.isnan(value).any():
        raise ValueError("bounds must have trailing dimension 2 and no NaNs")
    if np.any(value[..., 0] > value[..., 1]):
        raise ValueError("interval endpoints are reversed")
    return value


def _directed(value, direction, infinite_input):
    """One binary64 operation followed by one outward representable neighbor.

    An infinite result from an infinite operand is exact in the extended real
    line. Finite-operand overflow must still step inward on its finite-bound
    side, e.g. a positive overflow has a finite lower bound and +inf upper.
    Underflow and signed zero are expanded too; no empirical tolerance is used.
    """
    if np.isnan(value).any():
        raise ValueError("undefined interval arithmetic (e.g. 0*inf or inf-inf)")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rounded = np.nextafter(value, direction)
    return np.where(infinite_input & np.isinf(value), value, rounded)


def _outward_add(a, b):
    a, b = np.broadcast_arrays(_interval(a), _interval(b))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = a + b
    infinite = np.isinf(a) | np.isinf(b)
    return np.stack((_directed(result[..., 0], -np.inf, infinite[..., 0]),
                     _directed(result[..., 1], np.inf, infinite[..., 1])), axis=-1)


def _outward_multiply(a, b):
    a, b = np.broadcast_arrays(_interval(a), _interval(b))
    left, right = a[..., :, None], b[..., None, :]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        products = left * right
    infinite = np.isinf(left) | np.isinf(right)
    lo = _directed(products, -np.inf, infinite).min(axis=(-2, -1))
    hi = _directed(products, np.inf, infinite).max(axis=(-2, -1))
    return np.stack((lo, hi), axis=-1)


def _outward_divide(a, b):
    a, b = np.broadcast_arrays(_interval(a), _interval(b))
    if not np.isfinite(b).all() or np.any((b[..., 0] <= 0) & (b[..., 1] >= 0)):
        raise ValueError("divisor/scale bounds must be finite and must not contain zero")
    numerator, denominator = a[..., :, None], b[..., None, :]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        quotients = numerator / denominator
    infinite = np.broadcast_to(np.isinf(numerator), quotients.shape)
    lo = _directed(quotients, -np.inf, infinite).min(axis=(-2, -1))
    hi = _directed(quotients, np.inf, infinite).max(axis=(-2, -1))
    return np.stack((lo, hi), axis=-1)


def outward_inverse_range(coordinate_bounds, scale_bounds, offset_bounds):
    """Outer enclosure of (y - t) / s, broadcasting arrays shaped [...,2].

    y may be an exact binary64 atom [y,y], a finite interval, or a half-line.
    s and t are certified affine bounds; s must be finite and exclude zero
    (touching zero is rejected too), and t must be finite. Negative s works.
    Each subtraction/division rounds outwards. The result is an ENCLOSURE,
    not an exact inverse or proof of a joint physical witness: correlations
    between affine coefficients are conservatively forgotten. Callers must
    handle clipping preimages, structural unknowns and oracle semantics.
    """
    offset = _interval(offset_bounds)
    if not np.isfinite(offset).all():
        raise ValueError("inverse offset bounds must be finite")
    negative_offset = np.stack((-offset[..., 1], -offset[..., 0]), axis=-1)
    return _outward_divide(_outward_add(coordinate_bounds, negative_offset), scale_bounds)


def make_observations(*, object_id, occurrence_id, human, original_xywh,
                      original_serialized_intervals, original_unknown,
                      original_area_cap=None, **extra) -> dict[str, np.ndarray]:
    """Build normalized-tile observations; this does NOT certify the inputs.

    Required shapes: IDs/human [N], xywh [N,4], serialized intervals [N,4,2]
    (intervals of normalized x-center/y-center/width/height), unknown [N,4]
    (source L,T,R,B). Optional original_area_cap [N] is normalized tile area.
    Extra arrays may have any trailing dimensions, but must have N rows.
    affine [N,4] is signed sx,sy,tx,ty; current_window and unclipped [N,4]
    are xyxy in the current coordinate frame, initially normalized tile units.
    augmentation_clip_window starts as the entire plane; unlike current_window
    it contains only subsequent augmentation crop domains, not the source tile.
    affine_bounds [N,4,2] and augmentation_clip_window_bounds [N,4,2] enclose
    exact-real compositions of the sampled binary coefficients, with outward
    binary64 arithmetic. normalize additionally encloses exact division by w/h.
    The ordinary affine/window arrays are nominal approximations, not proofs.
    Unknown-side identity stays in the source frame, including after flips.
    """
    xywh = np.array(original_xywh, dtype=np.float64, copy=True)
    if xywh.ndim != 2 or xywh.shape[1:] != (4,):
        raise ValueError("original_xywh must have shape [N,4]")
    n = len(xywh)
    obs = {
        "object_id": np.array(object_id, copy=True),
        "occurrence_id": np.array(occurrence_id, copy=True),
        "human": np.array(human, dtype=bool, copy=True),
        "original_xywh": xywh,
        "original_serialized_intervals": np.array(original_serialized_intervals, dtype=np.float64, copy=True),
        "original_unknown": np.array(original_unknown, dtype=bool, copy=True),
        "affine": np.tile(np.array([1., 1., 0., 0.]), (n, 1)),
        "affine_bounds": np.tile(np.array([[1., 1.], [1., 1.], [0., 0.], [0., 0.]]), (n, 1, 1)),
        "current_window": np.tile(np.array([0., 0., 1., 1.]), (n, 1)),
        "augmentation_clip_window": np.tile(np.array([-np.inf, -np.inf, np.inf, np.inf]), (n, 1)),
        "augmentation_clip_window_bounds": np.tile(
            np.array([[-np.inf, -np.inf], [-np.inf, -np.inf], [np.inf, np.inf], [np.inf, np.inf]]),
            (n, 1, 1)),
        "unclipped": np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2,
                                     xywh[:, :2] + xywh[:, 2:] / 2), axis=1),
    }
    if original_area_cap is not None:
        obs["original_area_cap"] = np.array(original_area_cap, dtype=np.float64, copy=True)
    if obs.keys() & extra.keys():
        raise ValueError("extra arrays cannot replace built-in observation fields")
    obs.update(copy_observations(extra))
    _validate_observations(obs, n)
    return obs


def _validate_observations(obs, n):
    shapes = {"object_id": (), "occurrence_id": (), "human": (),
              "original_xywh": (4,), "original_serialized_intervals": (4, 2),
              "original_unknown": (4,), "affine": (4,),
              "current_window": (4,), "unclipped": (4,), "affine_bounds": (4, 2),
              "augmentation_clip_window": (4,), "augmentation_clip_window_bounds": (4, 2)}
    for key, shape in shapes.items():
        if key not in obs or obs[key].shape != (n, *shape):
            raise ValueError(f"{key} must have shape {(n, *shape)}")
    if "original_area_cap" in obs and obs["original_area_cap"].shape != (n,):
        raise ValueError("original_area_cap must have shape [N]")
    for key, value in obs.items():
        if value.ndim == 0 or len(value) != n:
            raise ValueError(f"observation field {key} is not row-aligned")
    for key in ("original_xywh", "original_serialized_intervals", "affine", "current_window", "unclipped"):
        if obs[key].dtype != np.float64 or not np.isfinite(obs[key]).all():
            raise ValueError(f"{key} must contain finite float64 coordinates")
    if np.any(obs["original_serialized_intervals"][..., 0] > obs["original_serialized_intervals"][..., 1]):
        raise ValueError("serialized intervals are reversed")
    for nominal_key, bounds_key in (("affine", "affine_bounds"),
                                    ("augmentation_clip_window", "augmentation_clip_window_bounds")):
        nominal, bounds = obs[nominal_key], obs[bounds_key]
        if nominal.dtype != np.float64 or bounds.dtype != np.float64 or np.isnan(nominal).any():
            raise ValueError(f"{nominal_key}/{bounds_key} must be float64 without NaNs")
        _interval(bounds)
        if np.any(nominal < bounds[..., 0]) or np.any(nominal > bounds[..., 1]):
            raise ValueError(f"{bounds_key} does not enclose nominal coordinates")


class ObservationInstances(Instances):
    """Instances plus independent row-aligned observations (axis-aligned detect).

    Source fields never change during geometry operations. Only affine/windows/
    bounds/unclipped change. A nominal current cap is original_area_cap *
    abs(sx * sy); certification needs interval arithmetic, not that nominal cap.
    bbox_only=True deliberately leaves observations alone: upstream RP uses it
    for its temporary filtering denominator, not a second geometric transform.
    """

    def __init__(self, bboxes, segments=None, keypoints=None, bbox_format="xywh",
                 normalized=True, *, obs):
        if keypoints is not None or (segments is not None and len(segments)):
            raise ValueError("observation transport supports detection boxes only, not segments/keypoints")
        super().__init__(bboxes, segments, keypoints, bbox_format, normalized)
        self.obs = copy_observations(obs)
        _validate_observations(self.obs, len(self))

    @classmethod
    def from_instances(cls, instances, obs):
        """Wrap a copy without changing bbox format, dtype, or numeric values."""
        return cls(instances.bboxes.copy(), instances.segments.copy(), instances.keypoints,
                   instances._bboxes.format, instances.normalized, obs=obs)

    def _transform_observations(self, sx, sy, tx=0., ty=0., *, _scale_bounds=None):
        transform = np.asarray([sx, sy, tx, ty], dtype=np.float64)
        if not np.isfinite(transform).all() or sx == 0 or sy == 0:
            raise ValueError("observation transform must be finite and invertible")
        # Sampled coefficients are exact binary constants. Normalization passes
        # bounds for its exact rational reciprocal as well as its nominal float.
        scales = np.repeat(transform[:2, None], 2, axis=1) if _scale_bounds is None else _interval(_scale_bounds)
        if (scales.shape != (2, 2) or not np.isfinite(scales).all()
                or np.any((scales[:, 0] <= 0) & (scales[:, 1] >= 0))):
            raise ValueError("transform scale bounds must be finite and exclude zero")
        if np.any(transform[:2] < scales[:, 0]) or np.any(transform[:2] > scales[:, 1]):
            raise ValueError("scale bounds do not enclose nominal coefficients")
        bounds = self.obs["affine_bounds"]
        updated = bounds.copy()
        for axis, offset in enumerate((tx, ty)):
            updated[:, axis] = _outward_multiply(bounds[:, axis], scales[axis])
            updated[:, axis + 2] = _outward_add(
                _outward_multiply(bounds[:, axis + 2], scales[axis]), [offset, offset])
        self.obs["affine_bounds"] = updated
        crop = self.obs["augmentation_clip_window"]
        crop_bounds = self.obs["augmentation_clip_window_bounds"]
        new_crop, new_crop_bounds = crop.copy(), crop_bounds.copy()
        for axis, (scale, offset) in enumerate(((sx, tx), (sy, ty))):
            # Choose endpoints by sign rather than min/max: an empty historical
            # crop (left > right) must stay empty, never be resurrected by a flip.
            order = [axis, axis + 2] if scale > 0 else [axis + 2, axis]
            new_crop[:, [axis, axis + 2]] = crop[:, order] * scale + offset
            new_crop_bounds[:, [axis, axis + 2]] = _outward_add(
                _outward_multiply(crop_bounds[:, order], scales[axis]), [offset, offset])
        self.obs["augmentation_clip_window"] = new_crop
        self.obs["augmentation_clip_window_bounds"] = new_crop_bounds
        affine = self.obs["affine"]
        affine[:, 0] *= sx
        affine[:, 1] *= sy
        affine[:, 2] = affine[:, 2] * sx + tx
        affine[:, 3] = affine[:, 3] * sy + ty
        window = self.obs["current_window"]
        x = window[:, [0, 2]] * sx + tx
        y = window[:, [1, 3]] * sy + ty
        window[:, 0], window[:, 2] = x.min(1), x.max(1)
        window[:, 1], window[:, 3] = y.min(1), y.max(1)
        xywh = self.obs["original_xywh"]
        lo, hi = xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2
        a, b = lo * affine[:, :2] + affine[:, 2:], hi * affine[:, :2] + affine[:, 2:]
        self.obs["unclipped"] = np.concatenate((np.minimum(a, b), np.maximum(a, b)), axis=1)

    def scale(self, scale_w, scale_h, bbox_only=False):
        super().scale(scale_w, scale_h, bbox_only=bbox_only)
        if not bbox_only:
            self._transform_observations(scale_w, scale_h)

    def denormalize(self, w, h):
        was_normalized = self.normalized
        super().denormalize(w, h)
        if was_normalized:
            self._transform_observations(w, h)

    def normalize(self, w, h):
        was_normalized = self.normalized
        super().normalize(w, h)
        if not was_normalized:
            reciprocal_bounds = np.stack((_outward_divide([1., 1.], [w, w]),
                                          _outward_divide([1., 1.], [h, h])))
            self._transform_observations(1 / w, 1 / h, _scale_bounds=reciprocal_bounds)

    def add_padding(self, padw, padh):
        super().add_padding(padw, padh)
        self._transform_observations(1., 1., padw, padh)

    def fliplr(self, w):
        super().fliplr(w)
        self._transform_observations(-1., 1., w, 0.)

    def flipud(self, h):
        super().flipud(h)
        self._transform_observations(1., -1., 0., h)

    def clip(self, w, h, preserve_obb=False):
        if preserve_obb:
            raise ValueError("OBB clipping is unsupported")
        super().clip(w, h, preserve_obb=False)
        # Clip the visible window, never the transported original coordinates.
        window = self.obs["current_window"]
        window[:, [0, 2]] = window[:, [0, 2]].clip(0, w)
        window[:, [1, 3]] = window[:, [1, 3]].clip(0, h)
        # True intersection, not coordinate clamping: preserve an empty domain.
        # max/min of binary constants require no arithmetic rounding envelope.
        crop = self.obs["augmentation_clip_window"]
        bounds = self.obs["augmentation_clip_window_bounds"]
        crop[:, :2] = np.maximum(crop[:, :2], 0.)
        crop[:, 2:] = np.minimum(crop[:, 2:], [w, h])
        bounds[:, :2] = np.maximum(bounds[:, :2], 0.)
        bounds[:, 2:] = np.minimum(bounds[:, 2:], np.array([w, h], dtype=np.float64)[None, :, None])

    def __getitem__(self, index):
        index = [index] if isinstance(index, (int, np.integer)) else index
        ordinary = super().__getitem__(index)
        return type(self).from_instances(ordinary, {key: value[index] for key, value in self.obs.items()})

    def remove_zero_area_boxes(self):
        good = super().remove_zero_area_boxes()
        self.obs = {key: value[good].copy() for key, value in self.obs.items()}
        return good

    @classmethod
    def concatenate(cls, instances_list, axis=0):
        if axis != 0 or not instances_list:
            raise ValueError("observation concatenation requires a nonempty list and axis=0")
        if not all(isinstance(item, ObservationInstances) for item in instances_list):
            raise TypeError("every concatenated instance must carry observations")
        keys = instances_list[0].obs.keys()
        if any(item.obs.keys() != keys for item in instances_list):
            raise ValueError("observation schemas differ")
        ordinary = Instances.concatenate(instances_list, axis=axis)
        obs = {key: np.concatenate([item.obs[key] for item in instances_list], axis=0) for key in keys}
        return cls.from_instances(ordinary, obs)

    def update(self, bboxes, segments=None, keypoints=None):
        # Upstream spatial Albumentations can reorder/filter through update().
        # There is no associated observation transform here; fail closed.
        raise NotImplementedError("untracked Instances.update is unsupported; supply an explicit observation transform")


def _require_observations(instances):
    if not isinstance(instances, ObservationInstances):
        raise TypeError("ObservationInstances required on every row, including non-human/ignore rows")
    return instances


class ObservationMosaic(Mosaic):
    def get_params(self, labels):
        if self.n != 4 or self.preserve_obb:
            raise ValueError("only four-patch axis-aligned Mosaic is supported")
        return super().get_params(labels)

    def _cat_labels(self, mosaic_labels):
        if not mosaic_labels:
            return {}
        cls, instances = [], []
        imgsz = self.imgsz * 2
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(_require_observations(labels["instances"]))
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": ObservationInstances.concatenate(instances, axis=0),
        }
        final_labels["instances"].clip(imgsz, imgsz, preserve_obb=self.preserve_obb)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class ObservationRandomPerspective(RandomPerspective):
    def get_params(self, labels):
        _check_perspective(self)
        return super().get_params(labels)

    def apply_instances(self, labels, params=None):
        _check_perspective(self)
        M = np.asarray(params["M"])
        if (M.shape != (3, 3) or not np.isfinite(M).all()
                or M[0, 1] != 0 or M[1, 0] != 0
                or not np.array_equal(M[2], [0., 0., 1.])
                or M[0, 0] == 0 or M[1, 1] == 0):
            raise ValueError("only finite invertible axis-aligned affine matrices are supported")
        cls = labels["cls"]
        instances = _require_observations(labels.pop("instances"))
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*params["orig_shape"][::-1])
        scale = params["scale"]
        bboxes = self.apply_bboxes(instances.bboxes, M)
        new_instances = ObservationInstances(bboxes, instances.segments, instances.keypoints,
                                             bbox_format="xyxy", normalized=False, obs=instances.obs)
        new_instances._transform_observations(M[0, 0], M[1, 1], M[0, 2], M[1, 2])
        new_instances.clip(*params["size"], preserve_obb=self.preserve_obb)
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        i = self.box_candidates(box1=instances.bboxes.T, box2=new_instances.bboxes.T, area_thr=0.10)
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        return labels


class ObservationFormat(Format):
    """Export labels['obs'] as independent NumPy arrays in the pixel frame.

    Original Format still formats images, classes, boxes and batch_idx. External
    P1 integration must filter obs with ~ignore and perform its own collation.
    """

    def get_params(self, labels):
        if self.return_mask or self.return_keypoint or self.return_obb:
            raise ValueError("observation Format supports detection only")
        _require_observations(labels.get("instances"))
        params = super().get_params(labels)
        labels["obs"] = copy_observations(params["instances"].obs)
        return params


def _check_perspective(transform):
    if transform.degrees != 0 or transform.shear != 0 or transform.perspective != 0 or transform.preserve_obb:
        raise ValueError("rotation, shear, perspective and OBB are unsupported")


def replace_observation_transforms(transform):
    """Return a shallow-cloned transform graph, preserving aliases and RNG state.

    No constructor runs, datasets/buffers remain shared, and p=0 nodes remain.
    Call after each build_transforms(), including the close_mosaic rebuild.
    Existing Albumentations objects (and their internal RNG) are retained.
    Unsupported spatial/custom transforms fail closed, even when hidden inside
    a disabled pre_transform. Nonzero MixUp/CutMix/CopyPaste are unsupported.
    """
    memo = {}

    def replace(node):
        if id(node) in memo:
            return memo[id(node)]
        kind = type(node)
        replacements = {Mosaic: ObservationMosaic, RandomPerspective: ObservationRandomPerspective,
                        Format: ObservationFormat}
        if kind in (Mosaic, ObservationMosaic):
            if node.n != 4 or node.preserve_obb:
                raise ValueError("only four-patch detection Mosaic is supported")
        elif kind in (RandomPerspective, ObservationRandomPerspective):
            _check_perspective(node)
        elif kind in (Format, ObservationFormat):
            if node.return_mask or node.return_keypoint or node.return_obb:
                raise ValueError("only detection Format is supported")
        elif kind in (MixUp, CutMix, CopyPaste):
            if node.p != 0:
                raise ValueError(f"active {kind.__name__} is unsupported")
        elif kind is Albumentations:
            if getattr(node, "contains_spatial", False):
                raise ValueError("spatial Albumentations is unsupported")
            memo[id(node)] = node
            return node
        elif kind not in (Compose, RandomFlip, RandomHSV, LetterBox):
            raise ValueError(f"unsupported transform: {kind.__name__}")
        cloned = copy(node)
        if kind in replacements:
            cloned.__class__ = replacements[kind]
        memo[id(node)] = cloned
        if kind is Compose:
            cloned.transforms = [replace(child) for child in node.transforms]
        if getattr(node, "pre_transform", None) is not None:
            cloned.pre_transform = replace(node.pre_transform)
        return cloned

    return replace(transform)
