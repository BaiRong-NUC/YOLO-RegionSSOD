"""Independent bounded-retention integration COMPONENT; no training gate granted.

Source certification and the physical oracle are supplied by a source factory.
This shell does not infer unknown edges from float32 boxes, certify a source,
or authorize experiments. Importing it performs no data/model access.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from ultralytics.data.utils import img2label_paths

from .bounded_retention_transport import (
    ObservationInstances, make_observations, replace_observation_transforms,
)
from .p1 import P1DetectionLoss, P1DetectionModel, P1Trainer
from .reliable_competitor_negative import SourceDataset

ROOT = Path(__file__).resolve().parents[3]
LEGAL_LABEL_ROOT = ROOT / "data/visdrone_area10/labeled/labels/train"
MODES = {"reference", "candidate", "control"}
OBSERVATIONS = "retention_observations"
RETENTION_GEOMETRY_FIELDS = (
    "affine", "affine_bounds", "original_unknown", "current_window",
    "augmentation_clip_window", "augmentation_clip_window_bounds",
)


class AuditedHumanLabels:
    """Read only allowlisted human text; bind it to a caller-pinned audit SHA.

    The audit's input_sha256 mapping is projected onto the legal directory;
    other entries (including historical manifest paths) are never opened.
    An alternate root is allowed ONLY for explicitly synthetic fixtures outside
    the repository data directory. This is not a physical-source certificate.

    Each label is bound once. counts accumulates raw_rows, retained_rows and
    native_duplicate_rows only after successful native-order matching; all
    three keys exist even at zero. duplicate_aliases is a JSON-serializable
    report of merged groups with original text/tokens and 1-based line numbers.
    Float32-equal rows may merge only if all five exact tokens match.
    """

    def __init__(self, audit_path, audit_sha256, *, label_root=LEGAL_LABEL_ROOT,
                 synthetic_fixture=False):
        self.root = Path(label_root).resolve()
        if self.root != LEGAL_LABEL_ROOT.resolve():
            if not synthetic_fixture or self.root.is_relative_to((ROOT / "data").resolve()):
                raise ValueError("only legal human labels or explicit synthetic fixtures allowed")
        if not isinstance(audit_sha256, str) or len(audit_sha256) != 64:
            raise ValueError("pinned source audit SHA256 required")
        raw = Path(audit_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != audit_sha256:
            raise ValueError("source audit SHA256 mismatch")
        audit = json.loads(raw)
        self.audit_sha256 = audit_sha256
        self.hashes = {Path(p).resolve(): digest for p, digest in audit["input_sha256"].items()
                       if Path(p).resolve().parent == self.root and Path(p).suffix == ".txt"}
        if not self.hashes:
            raise ValueError("source audit contains no permitted human labels")
        self.counts = Counter(raw_rows=0, retained_rows=0, native_duplicate_rows=0)
        # JSON-serializable source aliases, populated only after exact binding.
        # Dataset adds source_object_id to each duplicate group below.
        self.duplicate_aliases = []
        self._bound_paths = set()

    def rows(self, label_path, label):
        path = Path(label_path).resolve()
        if path.parent != self.root or path not in self.hashes:
            raise ValueError("human label outside source audit allowlist")
        if path in self._bound_paths:
            raise ValueError(f"human label rows already bound once: {path}")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != self.hashes[path]:
            raise ValueError("human label SHA256 mismatch")
        rows = [(i, text) for i, text in enumerate(raw.decode("utf-8").splitlines(), 1) if text.strip()]
        fields = [text.split() for _, text in rows]
        if any(len(row) != 5 for row in fields):
            raise ValueError("human labels must be original five-column detection rows")
        parsed = np.asarray(fields, dtype=np.float32).reshape(-1, 5)
        if (not np.isfinite(parsed).all() or (parsed[:, 0] != parsed[:, 0].round()).any()
                or (parsed[:, 1:] < 0).any() or (parsed[:, 1:] > 1).any()
                or (parsed[:, 3:] <= 0).any()):
            raise ValueError("invalid original human labels")
        # Match installed verify_image_label exactly: unique indices are in
        # sorted row order, but native applies them ONLY when duplicates exist.
        _, first, inverse, multiplicities = np.unique(
            parsed, axis=0, return_index=True, return_inverse=True, return_counts=True)
        retained = first if len(first) < len(rows) else np.arange(len(rows))
        duplicates = []
        for group in np.flatnonzero(multiplicities > 1):
            members = np.flatnonzero(inverse == group)
            aliases = [{"line_number": rows[i][0], "text": rows[i][1], "tokens": fields[i]}
                       for i in members]
            if any(fields[i] != fields[first[group]] for i in members):
                raise ValueError(
                    f"ambiguous native float32 duplicate with different exact tokens: {path}; "
                    f"aliases={json.dumps(aliases, ensure_ascii=False)}")
            duplicates.append({"label_path": str(path), "label_sha256": digest,
                "retained_row_index": int(group),  # zero-based native loaded row
                "retained_line_number": rows[first[group]][0], "aliases": aliases})
        expected = parsed[retained]
        actual = np.concatenate((label["cls"], label["bboxes"]), axis=1)
        if (label.get("bbox_format") != "xywh" or not label.get("normalized")
                or expected.shape != actual.shape or not np.array_equal(expected, actual)):
            raise ValueError("raw human rows differ from loaded labels (dedup/order/coordinates)")
        self.counts.update(raw_rows=len(rows), retained_rows=len(retained),
                           native_duplicate_rows=len(rows) - len(retained))
        self.duplicate_aliases.extend(duplicates)
        self._bound_paths.add(path)
        return path, digest, [rows[i] for i in retained]


class BoundedRetentionDataset(SourceDataset):
    """Unchanged P1 pseudo/ignore path plus certified, row-bound human sidecars.

    source_factory is required and called with keyword arguments label_path,
    label_sha256, audit_sha256, line_number (1-based), text (original row), and
    source_object_id. Return a mapping containing physical (oracle object),
    original_xywh [4], original_serialized_intervals [4,2], original_unknown
    bool[4]; physical must expose compatible_ranges. Optional original_area_cap
    is transported without recomputation.
    The factory owns certificate validation and physical semantics.
    """

    def __init__(self, *args, metadata_path, source_audit_path, source_audit_sha256,
                 source_factory: Callable[..., Mapping], label_root=LEGAL_LABEL_ROOT,
                 synthetic_fixture=False, **kwargs):
        if not callable(source_factory):
            raise TypeError("certified source_factory is required")
        self.human_labels = AuditedHumanLabels(
            source_audit_path, source_audit_sha256, label_root=label_root,
            synthetic_fixture=synthetic_fixture)
        self._sources = {}
        self._occurrence = 0
        self._pseudo_label_root = (Path(metadata_path).resolve().parent / "labels/train").resolve()
        super().__init__(*args, metadata_path=Path(metadata_path), **kwargs)
        for label in self.labels:
            key = str(Path(label["im_file"]).resolve())
            if key in self.p1_metadata:
                # Legal human images must not be disguised as pseudo rows.
                path = Path(img2label_paths([key])[0]).resolve()
                if path.parent == self.human_labels.root:
                    raise ValueError("human image appears in pseudo metadata")
                continue
            alias_start = len(self.human_labels.duplicate_aliases)
            path, digest, rows = self.human_labels.rows(img2label_paths([key])[0], label)
            tokens = self.p1_tokens[key].reshape(-1).astype(np.int64)
            duplicate_groups = self.human_labels.duplicate_aliases[alias_start:]
            aliases_by_line = {}
            for group in duplicate_groups:
                group["source_object_id"] = int(tokens[group["retained_row_index"]])
                aliases_by_line[group["retained_line_number"]] = group["aliases"]
            for token, (line_number, text) in zip(tokens, rows, strict=True):
                source = dict(source_factory(label_path=path, label_sha256=digest,
                    audit_sha256=self.human_labels.audit_sha256, line_number=line_number,
                    text=text, source_object_id=int(token)))
                required = {"physical", "original_xywh", "original_serialized_intervals", "original_unknown"}
                if not required.issubset(source) or source["physical"] is None:
                    raise ValueError("source factory lacks certified observation fields")
                if not callable(getattr(source["physical"], "compatible_ranges", None)):
                    raise TypeError("source factory physical oracle must implement compatible_ranges")
                if np.asarray(source["original_unknown"]).dtype != np.bool_:
                    raise ValueError("source unknown mask must be explicit bool")
                # Factory must preserve the original serialized coordinates.
                if not np.array_equal(np.asarray(source["original_xywh"], dtype=np.float64),
                                      np.asarray(text.split()[1:], dtype=np.float64)):
                    raise ValueError("source factory changed original serialized coordinates")
                source["source_aliases"] = deepcopy(aliases_by_line.get(line_number,
                    [{"line_number": line_number, "text": text, "tokens": text.split()}]))
                self._sources[int(token)] = source

    def get_labels(self):
        # Check paths BEFORE upstream cache lookup or verification can open
        # labels. Presence of 'labeled' somewhere in a path is not permission.
        expected = set()
        for image, label in zip(self.im_files, img2label_paths(self.im_files), strict=True):
            key, path = str(Path(image).resolve()), Path(label).resolve()
            expected.add(key)
            if path.parent == self.human_labels.root:
                if key in self.p1_metadata:
                    raise ValueError("human image appears in pseudo metadata")
                if path not in self.human_labels.hashes:
                    raise ValueError("human label outside source audit allowlist")
            elif key not in self.p1_metadata or path.parent != self._pseudo_label_root:
                raise ValueError("training label is neither audited human nor accepted P1 pseudo export")
        labels = super().get_labels()
        if {str(Path(label["im_file"]).resolve()) for label in labels} != expected:
            raise ValueError("upstream label scan dropped or substituted an input image")
        return labels

    def build_transforms(self, hyp=None):
        return replace_observation_transforms(super().build_transforms(hyp))

    def get_image_and_label(self, index):
        label = super().get_image_and_label(index)
        tokens = label["cls"].reshape(-1).astype(np.int64)
        n = len(tokens)
        human = self.p1_lookup[tokens, 3].numpy().astype(bool)
        # Every row has a transport record; non-human observations are inert.
        xywh = label["instances"].bboxes.astype(np.float64, copy=True)
        intervals = np.repeat(xywh[..., None], 2, axis=2)
        unknown = np.zeros((n, 4), dtype=bool)
        physical = np.empty(n, dtype=object)
        physical[:] = None
        caps = np.full(n, np.nan, dtype=np.float64)
        for i in np.flatnonzero(human):
            source = self._sources[int(tokens[i])]
            xywh[i] = source["original_xywh"]
            intervals[i] = source["original_serialized_intervals"]
            unknown[i] = source["original_unknown"]
            physical[i] = source["physical"]
            if "original_area_cap" in source:
                caps[i] = source["original_area_cap"]
        self._occurrence += 1
        obs = make_observations(object_id=tokens, source_object_id=tokens,
            occurrence_id=np.full(n, self._occurrence, dtype=np.int64), human=human,
            original_xywh=xywh, original_serialized_intervals=intervals,
            original_unknown=unknown, original_area_cap=caps, physical=physical)
        label["instances"] = ObservationInstances.from_instances(label["instances"], obs)
        return label

    def __getitem__(self, index):
        # SourceDataset decodes and filters original metadata using the same
        # augmented scalar tokens. obs survives that path unfiltered on purpose.
        sample = super().__getitem__(index)
        obs = sample.pop("obs")
        tokens = obs["object_id"]
        keep = ~self.p1_lookup[torch.as_tensor(tokens), 2].bool().numpy()
        obs = {key: values[keep] for key, values in obs.items()}
        if (not np.array_equal(obs["object_id"], sample["object_id"].numpy())
                or not np.array_equal(obs["human"], sample["object_is_human"].numpy())):
            raise ValueError("observation/object provenance binding changed")
        missing = set(RETENTION_GEOMETRY_FIELDS) - obs.keys()
        if missing:
            raise ValueError(f"observation transport lacks required geometry fields: {sorted(missing)}")
        sample["obs"] = obs
        sample[OBSERVATIONS] = [None if not is_human else {
            "physical": obs["physical"][i],
            **{key: obs[key][i].copy() for key in RETENTION_GEOMETRY_FIELDS},
            "source_object_id": int(obs["source_object_id"][i]),
            "occurrence_id": int(obs["occurrence_id"][i]),
            # Resolve through the transported source token AFTER filtering;
            # never associate aliases with an augmented row index.
            "source_aliases": deepcopy(self._sources[int(obs["source_object_id"][i])]["source_aliases"]),
        } for i, is_human in enumerate(obs["human"])]
        return sample

    @staticmethod
    def collate_fn(batch):
        ordinary = [{key: value for key, value in sample.items()
                     if key not in {"obs", OBSERVATIONS}} for sample in batch]
        result = SourceDataset.collate_fn(ordinary)
        keys = batch[0]["obs"].keys()
        if any(sample["obs"].keys() != keys for sample in batch):
            raise ValueError("observation schemas differ")
        for sample in batch:
            n = len(sample["cls"])
            if len(sample[OBSERVATIONS]) != n or any(len(v) != n for v in sample["obs"].values()):
                raise ValueError("observation row count differs from positives")
        result["obs"] = {key: np.concatenate([s["obs"][key] for s in batch]) for key in keys}
        result[OBSERVATIONS] = [item for sample in batch for item in sample[OBSERVATIONS]]
        return result


class BoundedRetentionModel(P1DetectionModel):
    def init_criterion(self):
        # Missing objective is a genuine dependency error, never a fallback.
        from .bounded_retention_objective import BoundedRetentionLoss
        return BoundedRetentionLoss(self)

    def loss(self, batch, preds=None):
        if self.training:
            return super().loss(batch, preds)
        if preds is None:
            preds = self.forward(batch["img"])
        return P1DetectionLoss(self)(preds, batch)


class BoundedRetentionTrainer(P1Trainer):
    def __init__(self, *args, mode, source_factory, source_audit_path,
                 source_audit_sha256, label_root=LEGAL_LABEL_ROOT,
                 synthetic_fixture=False, **kwargs):
        import ultralytics
        if ultralytics.__version__ != "8.4.172" or mode not in MODES:
            raise ValueError("pinned Ultralytics and legal retention mode required")
        if not callable(source_factory):
            raise TypeError("certified source_factory is required")
        overrides = kwargs.get("overrides") or {}
        if overrides.get("resume") or overrides.get("workers", 0) != 0:
            raise ValueError("resume is forbidden and workers=0 required")
        kwargs["overrides"] = {**overrides, "workers": 0}
        self.mode = mode
        self.source_options = dict(source_factory=source_factory,
            source_audit_path=source_audit_path, source_audit_sha256=source_audit_sha256,
            label_root=label_root, synthetic_fixture=synthetic_fixture)
        super().__init__(*args, **kwargs)
        if self.args.cls_pw != 0 or self.args.workers != 0 or self.args.resume:
            raise ValueError("original P1 cls_pw, workers=0 and no resume required")

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(BoundedRetentionModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose))
        model.p1_options = self.p1_options
        model.bounded_retention_mode = self.mode
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        if self.args.single_cls or self.args.classes is not None or self.args.fraction != 1:
            raise ValueError("all classes and fraction=1 required")
        return BoundedRetentionDataset(img_path=img_path,
            metadata_path=Path(self.data["p1_metadata"]), **self.source_options,
            imgsz=self.args.imgsz, batch_size=batch, augment=True, hyp=self.args,
            rect=self.args.rect, cache=self.args.cache or None, single_cls=False,
            stride=max(int(self.model.stride.max()), 32), pad=0., prefix="train: ",
            task="detect", data=self.data, fraction=1.)


def run_training(config_path, *, source_factory, engineering_gates_passed=False):
    """Future explicit worker; caller must establish all gates before invoking.

    This component and synthetic tests do not establish those gates. Never
    resume or reuse an output directory; keep new artifacts under outputs/.
    """
    from .runner import load_experiment_config
    config = load_experiment_config(Path(config_path).resolve())
    settings = config["bounded_retention"]
    if config.get("resume") or config["train"].get("resume"):
        raise ValueError("resume is forbidden")
    name = config["name"]
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("run name must be a single fresh directory name")
    run = Path(config["project"]).resolve() / name
    if not run.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("independent run must be inside ignored outputs")
    if run.exists():
        raise FileExistsError(run)
    if not engineering_gates_passed:
        raise RuntimeError("all source/transport/loss/control/integration gates required")
    if settings["mode"] not in MODES or config["train"].get("workers", 0) != 0:
        raise ValueError("legal mode and workers=0 required")
    # Fail before model loading if required source/objective APIs are absent.
    if not callable(source_factory):
        raise TypeError("certified source_factory is required")
    from .bounded_retention_objective import BoundedRetentionLoss  # noqa: F401
    from ultralytics import YOLO
    trainer = partial(BoundedRetentionTrainer, mode=settings["mode"],
        source_factory=source_factory, source_audit_path=settings["source_audit_path"],
        source_audit_sha256=settings["source_audit_sha256"], p1_options=config["p1"])
    train_args = {**config["train"], "workers": 0}
    result = YOLO(config["model"]).train(data=config["data"], name=name,
        project=str(run.parent), exist_ok=False, trainer=trainer, **train_args)
    return Path(result.save_dir).resolve()
