"""Synthetic integration COMPONENT checks, not release of scientific gates.

No real labels, manifests, quality caches, weights, or optimizer are accessed.
The injected source factory below is a deliberately synthetic oracle stub.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset

from regionssod.training import bounded_retention_training as integration
from regionssod.training.reliable_competitor_negative import validate_batch_provenance


class SyntheticPhysical(dict):
    """Transport-only fixture. Never claim physical membership certification."""
    def compatible_ranges(self, ranges):
        raise AssertionError("transport fixture is not a certified physical oracle")


def fixture_factory(**binding):
    xywh = np.asarray(binding["text"].split()[1:], dtype=np.float64)
    return {"physical": SyntheticPhysical(binding), "original_xywh": xywh,
            "original_serialized_intervals": np.repeat(xywh[:, None], 2, axis=1),
            "original_unknown": np.array([True, False, False, False]),
            "original_area_cap": 2.0}


@pytest.fixture
def synthetic(tmp_path):
    labeled = tmp_path / "labeled"
    pseudo = tmp_path / "pseudo"
    val = tmp_path / "val"
    human_text = "0 .25 .5 .5 .4\n1 .75 .6 .2 .3\n"
    for folder in (labeled, pseudo, val):
        (folder / "images/train").mkdir(parents=True)
        (folder / "labels/train").mkdir(parents=True)
        Image.fromarray(np.full((64, 96, 3), 115, dtype=np.uint8)).save(folder / "images/train/one.png")
    human_label = labeled / "labels/train/one.txt"
    human_label.write_text(human_text, encoding="utf-8")
    (pseudo / "labels/train/one.txt").write_text("1 .5 .5 .3 .3\n", encoding="utf-8")
    (val / "labels/train/one.txt").write_text("0 .5 .5 .4 .4\n", encoding="utf-8")
    metadata = pseudo / "metadata.jsonl"
    metadata.write_text(json.dumps({"image": str(pseudo / "images/train/one.png"),
        "positive": [{"class_id": 1, "xywhn": [.5, .5, .3, .3], "confidence": .8}],
        "ignore": [{"class_id": 0, "xywhn": [.2, .2, .15, .15], "confidence": .3}]}) + "\n", encoding="utf-8")
    train = tmp_path / "train.txt"
    train.write_text("\n".join(str(p / "images/train/one.png") for p in (labeled, pseudo)) + "\n", encoding="utf-8")
    audit = tmp_path / "source_audit.json"
    audit.write_text(json.dumps({"input_sha256": {
        str(human_label): hashlib.sha256(human_label.read_bytes()).hexdigest(),
        str(tmp_path / "forbidden_manifest.jsonl"): "not-opened",
    }}), encoding="utf-8")
    data = {"path": str(tmp_path), "train": str(train), "val": str(val / "images/train"),
            "names": {0: "a", 1: "b"}, "nc": 2, "channels": 3, "p1_metadata": str(metadata)}
    data_path = tmp_path / "synthetic.yaml"
    data_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return SimpleNamespace(root=tmp_path, train=train, label=human_label, audit=audit,
        metadata=metadata, data=data, data_path=data_path,
        source_options=dict(source_audit_path=audit,
            source_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
            source_factory=fixture_factory, label_root=human_label.parent, synthetic_fixture=True))


def dataset(fixture, *, augment=True):
    return integration.BoundedRetentionDataset(img_path=str(fixture.train),
        metadata_path=fixture.metadata, **fixture.source_options, imgsz=64,
        batch_size=2, augment=augment, hyp=get_cfg(), rect=False, cache=False,
        single_cls=False, stride=32, pad=0., task="detect", data=fixture.data)


def test_raw_hash_and_physical_binding_and_unique_occurrences(synthetic):
    ds = dataset(synthetic)
    first = ds.get_image_and_label(0)
    second = ds.get_image_and_label(0)
    a, b = first["instances"].obs, second["instances"].obs
    assert len(set(a["source_object_id"])) == 2
    assert np.array_equal(a["source_object_id"], b["source_object_id"])
    assert (a["occurrence_id"] != b["occurrence_id"]).all()
    assert a["physical"][0]["text"] == "0 .25 .5 .5 .4"
    assert a["physical"][1]["line_number"] == 2
    assert a["physical"][0]["audit_sha256"] == synthetic.source_options["source_audit_sha256"]
    assert a["original_unknown"][1, 0]  # supplied by factory, NOT float32 boundary inference
    # Original tile visibility must not be mistaken for an augmentation clip.
    np.testing.assert_array_equal(a["current_window"], np.tile([0., 0., 1., 1.], (2, 1)))
    np.testing.assert_array_equal(a["augmentation_clip_window"],
                                  np.tile([-np.inf, -np.inf, np.inf, np.inf], (2, 1)))
    assert a["affine_bounds"].shape == (2, 4, 2)
    assert a["augmentation_clip_window_bounds"].shape == (2, 4, 2)


def test_real_augmentation_filter_and_collate(synthetic):
    ds = dataset(synthetic)
    assert ds.augment
    assert ds.transforms.transforms[0].transforms[0].p == 1.0
    # Default augmentation remains active, including random mosaic and flips.
    samples = [ds[i % 2] for i in range(8)]
    geometry_shapes = {"affine": (4,), "original_unknown": (4,),
        "affine_bounds": (4, 2), "current_window": (4,),
        "augmentation_clip_window": (4,), "augmentation_clip_window_bounds": (4, 2)}
    for sample in samples:
        for i, record in enumerate(sample["retention_observations"]):
            if record is None:
                continue
            for key, shape in geometry_shapes.items():
                assert record[key].shape == shape
                np.testing.assert_array_equal(record[key], sample["obs"][key][i])
                assert not np.shares_memory(record[key], sample["obs"][key][i])
    batch = ds.collate_fn(samples)
    validate_batch_provenance(batch, 2)
    assert len(batch["retention_observations"]) == len(batch["cls"])
    assert np.array_equal(batch["obs"]["object_id"], batch["object_id"].numpy())
    for i, record in enumerate(batch["retention_observations"]):
        if batch["object_is_human"][i]:
            assert record["physical"]["source_object_id"] == batch["object_id"][i]
            for key in geometry_shapes:
                np.testing.assert_array_equal(record[key], batch["obs"][key][i])
            assert record["source_object_id"] == batch["object_id"][i]
        else:
            assert record is None
    ds.close_mosaic(get_cfg())
    assert ds.transforms.transforms[-1].__class__.__name__ == "ObservationFormat"
    validate_batch_provenance(ds.collate_fn([ds[0], ds[1]]), 2)


@pytest.mark.parametrize("missing", ["affine_bounds", "augmentation_clip_window",
                                     "augmentation_clip_window_bounds", "current_window"])
def test_missing_crop_or_bounds_field_is_rejected_not_reconstructed(synthetic, monkeypatch, missing):
    ds = dataset(synthetic, augment=False)
    original = integration.SourceDataset.__getitem__
    def damaged(self, index):
        sample = original(self, index)
        sample["obs"].pop(missing)
        return sample
    monkeypatch.setattr(integration.SourceDataset, "__getitem__", damaged)
    with pytest.raises(ValueError, match="required geometry fields"):
        ds[0]


def test_pseudo_ignore_stays_original_p1(synthetic):
    ds = dataset(synthetic, augment=False)
    pseudo = ds[1]
    assert pseudo["retention_observations"] == [None]
    torch.testing.assert_close(pseudo["pseudo_weights"], torch.tensor([.8]))
    assert len(pseudo["ignore_bboxes"]) == 1
    assert len(pseudo["obs"]["physical"]) == 1


def rebind_fixture_text(synthetic, text):
    synthetic.label.write_text(text, encoding="utf-8")
    synthetic.audit.write_text(json.dumps({"input_sha256": {
        str(synthetic.label): hashlib.sha256(synthetic.label.read_bytes()).hexdigest(),
    }}), encoding="utf-8")
    synthetic.source_options["source_audit_sha256"] = hashlib.sha256(synthetic.audit.read_bytes()).hexdigest()


def test_empty_human_keeps_empty_observations_collatable(synthetic):
    rebind_fixture_text(synthetic, "")
    ds = dataset(synthetic, augment=False)
    sample = ds[0]
    assert dict(ds.human_labels.counts) == {
        "raw_rows": 0, "retained_rows": 0, "native_duplicate_rows": 0}
    assert ds.human_labels.duplicate_aliases == []
    assert sample["retention_observations"] == []
    assert sample["obs"]["affine"].shape == (0, 4)
    batch = ds.collate_fn([sample, ds[1]])
    validate_batch_provenance(batch, 2)
    assert batch["retention_observations"] == [None]
    assert batch["batch_idx"].tolist() == [1.0]


@pytest.mark.parametrize("augment", [False, True])
def test_native_duplicate_order_first_source_and_aliases(synthetic, augment):
    # Native sorts unique rows only if a duplicate is present. Blank lines and
    # different whitespace must not lose original line numbers/text.
    rebind_fixture_text(synthetic,
        "\n1 .75 .6 .2 .3\n0 .25 .5 .5 .4\n\n1  .75 .6 .2 .3\n0 .25 .5 .5 .4\n")
    ds = dataset(synthetic, augment=augment)
    assert isinstance(ds.human_labels.counts, Counter)
    assert dict(ds.human_labels.counts) == {
        "raw_rows": 4, "retained_rows": 2, "native_duplicate_rows": 2}
    assert len(ds._sources) == 2
    assert [s["physical"]["line_number"] for s in ds._sources.values()] == [3, 2]
    report = json.loads(json.dumps(ds.human_labels.duplicate_aliases))
    assert [g["retained_row_index"] for g in report] == [0, 1]
    assert [g["retained_line_number"] for g in report] == [3, 2]
    assert [[a["line_number"] for a in g["aliases"]] for g in report] == [[3, 6], [2, 5]]
    for group in report:
        assert group["label_path"] == str(synthetic.label.resolve())
        assert group["label_sha256"] == hashlib.sha256(synthetic.label.read_bytes()).hexdigest()
        assert ds._sources[group["source_object_id"]]["source_aliases"] == group["aliases"]
    # Exercise transport, ignored pseudo filtering, and collation with aliases
    # looked up from the transported identity, including mosaic repetitions.
    batch = ds.collate_fn([ds[i % 2] for i in range(8)])
    validate_batch_provenance(batch, 2)
    for i, record in enumerate(batch["retention_observations"]):
        if record is None:
            assert not batch["object_is_human"][i]
            continue
        group = next(g for g in report if g["source_object_id"] == record["source_object_id"])
        assert record["source_aliases"] == group["aliases"]
        assert record["physical"]["line_number"] == group["retained_line_number"]
        assert record["source_object_id"] == batch["object_id"][i]
    assert dict(ds.human_labels.counts) == {
        "raw_rows": 4, "retained_rows": 2, "native_duplicate_rows": 2}


def test_no_duplicate_keeps_unsorted_native_order_and_zero_count(synthetic):
    rebind_fixture_text(synthetic, "1 .75 .6 .2 .3\n0 .25 .5 .5 .4\n")
    ds = dataset(synthetic, augment=False)
    assert ds[0]["cls"].reshape(-1).tolist() == [1, 0]
    assert [s["physical"]["line_number"] for s in ds._sources.values()] == [1, 2]
    assert dict(ds.human_labels.counts) == {
        "raw_rows": 2, "retained_rows": 2, "native_duplicate_rows": 0}
    assert ds.human_labels.duplicate_aliases == []
    with pytest.raises(ValueError, match="already bound once"):
        ds.human_labels.rows(synthetic.label, ds.labels[0])
    assert ds.human_labels.counts["raw_rows"] == 2


@pytest.mark.parametrize("other", ["0 .2500000001 .5 .5 .4", "0 0.25 .5 .5 .4",
                                   "0.0 .25 .5 .5 .4"])
def test_float32_duplicate_different_exact_tokens_rejected(synthetic, other):
    original = "0 .25 .5 .5 .4"
    np.testing.assert_array_equal(np.array(original.split(), dtype=np.float32),
                                  np.array(other.split(), dtype=np.float32))
    rebind_fixture_text(synthetic, original + "\n\n" + other + "\n")
    calls = []
    def factory(**binding):
        calls.append(binding)
        return fixture_factory(**binding)
    synthetic.source_options["source_factory"] = factory
    with pytest.raises(ValueError, match="different exact tokens") as error:
        dataset(synthetic)
    message = str(error.value)
    assert str(synthetic.label.resolve()) in message
    assert original in message and other in message
    assert '"line_number": 1' in message and '"line_number": 3' in message
    assert calls == []


def test_duplicate_still_rejects_non_native_loaded_order(synthetic, monkeypatch):
    rebind_fixture_text(synthetic, "1 .75 .6 .2 .3\n0 .25 .5 .5 .4\n1 .75 .6 .2 .3\n")
    original = integration.SourceDataset.__init__
    def altered(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for key in ("cls", "bboxes"):
            self.labels[0][key] = self.labels[0][key][::-1].copy()
    monkeypatch.setattr(integration.SourceDataset, "__init__", altered)
    with pytest.raises(ValueError, match="raw human rows"):
        dataset(synthetic)


def test_blank_lines_preserve_physical_source_line_number(synthetic):
    rebind_fixture_text(synthetic, "\n0 .25 .5 .5 .4\n\n1 .75 .6 .2 .3\n")
    ds = dataset(synthetic, augment=False)
    records = ds[0]["retention_observations"]
    assert [r["physical"]["line_number"] for r in records] == [2, 4]


def test_human_cannot_be_reclassified_as_pseudo(synthetic):
    metadata = json.loads(synthetic.metadata.read_text())
    human = {"image": str(synthetic.root / "labeled/images/train/one.png"),
        "positive": [{"class_id": 0, "xywhn": [.25, .5, .5, .4], "confidence": 1.},
                     {"class_id": 1, "xywhn": [.75, .6, .2, .3], "confidence": 1.}], "ignore": []}
    synthetic.metadata.write_text(json.dumps(metadata) + "\n" + json.dumps(human) + "\n")
    with pytest.raises(ValueError, match="human image appears"):
        dataset(synthetic)


@pytest.mark.parametrize("mutation", ["hash", "reorder", "dedup", "coordinate"])
def test_refuse_misbinding(synthetic, monkeypatch, mutation):
    if mutation == "hash":
        synthetic.label.write_text("0 .5 .5 .3 .3\n", encoding="utf-8")
    else:
        original = integration.SourceDataset.__init__
        def altered(self, *args, **kwargs):
            original(self, *args, **kwargs)
            label = self.labels[0]
            if mutation == "reorder":
                label["cls"] = label["cls"][::-1].copy()
                label["bboxes"] = label["bboxes"][::-1].copy()
            elif mutation == "dedup":
                label["cls"] = label["cls"][:1]
                label["bboxes"] = label["bboxes"][:1]
            else:
                label["bboxes"][0, 0] += .001
        monkeypatch.setattr(integration.SourceDataset, "__init__", altered)
    with pytest.raises(ValueError, match="SHA256|raw human rows"):
        dataset(synthetic)


def test_audit_sha_and_allowlist_reject_before_label_read(synthetic):
    with pytest.raises(ValueError, match="audit SHA256"):
        integration.AuditedHumanLabels(synthetic.audit, "0" * 64,
            label_root=synthetic.label.parent, synthetic_fixture=True)
    reader = integration.AuditedHumanLabels(synthetic.audit,
        synthetic.source_options["source_audit_sha256"], label_root=synthetic.label.parent,
        synthetic_fixture=True)
    with pytest.raises(ValueError, match="allowlist"):
        reader.rows(synthetic.root / "forbidden_manifest.jsonl", {})
    with pytest.raises(ValueError, match="explicit synthetic"):
        integration.AuditedHumanLabels(synthetic.audit,
            synthetic.source_options["source_audit_sha256"], label_root=synthetic.label.parent)


@pytest.mark.parametrize("forged_metadata", [False, True])
def test_disallowed_training_paths_rejected_before_upstream_label_access(tmp_path, monkeypatch, forged_metadata):
    ds = object.__new__(integration.BoundedRetentionDataset)
    image = str(tmp_path / "hidden/images/train/no-file.png")
    ds.im_files = [image]
    ds.human_labels = SimpleNamespace(root=tmp_path / "legal/labels/train", hashes={})
    ds._pseudo_label_root = tmp_path / "accepted/labels/train"
    ds.p1_metadata = {str(Path(image).resolve()): {}} if forged_metadata else {}
    def forbidden(self):
        pytest.fail("upstream must not access disallowed labels/cache")
    monkeypatch.setattr(YOLODataset, "get_labels", forbidden)
    with pytest.raises(ValueError, match="neither audited human nor accepted"):
        ds.get_labels()


def test_factory_cannot_omit_physical_or_change_original(synthetic):
    for mutation in ("physical", "coordinates"):
        def invalid(**binding):
            result = fixture_factory(**binding)
            if mutation == "physical":
                result.pop("physical")
            else:
                result["original_xywh"][0] += .01
            return result
        options = dict(synthetic.source_options, source_factory=invalid)
        fixture = SimpleNamespace(**{**vars(synthetic), "source_options": options})
        with pytest.raises(ValueError, match="factory"):
            dataset(fixture)


@pytest.mark.parametrize("mode", ["reference", "candidate", "control"])
def test_real_trainer_construction_train_val_loaders_zero_optimizer(synthetic, mode):
    trainer = integration.BoundedRetentionTrainer(mode=mode, **synthetic.source_options,
        overrides={"model": "yolo11n.yaml", "data": str(synthetic.data_path),
            "project": str(synthetic.root / "outputs"), "name": mode, "imgsz": 64,
            "batch": 2, "workers": 0, "device": "cpu", "plots": False, "save": False},
        _callbacks={})
    # No real model, weights, forward, setup_train, or optimizer. This shell
    # only needs stride to exercise actual DetectionTrainer loader construction.
    trainer.model = SimpleNamespace(stride=torch.tensor([8, 16, 32]))
    train_loader = trainer.get_dataloader(trainer.data["train"], batch_size=2, rank=-1, mode="train")
    val_loader = trainer.get_dataloader(trainer.data["val"], batch_size=1, rank=-1, mode="val")
    train_batch, val_batch = next(iter(train_loader)), next(iter(val_loader))
    validate_batch_provenance(train_batch, 2)
    assert isinstance(train_loader.dataset, integration.BoundedRetentionDataset)
    assert type(val_loader.dataset) is YOLODataset
    assert "retention_observations" in train_batch
    assert "retention_observations" not in val_batch
    assert trainer.args.workers == 0 and trainer.args.mosaic == 1.0
    assert trainer.optimizer is None


def test_get_model_names_options_mode_and_loading_order(monkeypatch):
    events = []
    class Stub:
        def __init__(self, cfg, nc, ch, verbose):
            events.append(("create", cfg, nc, ch))
        def load(self, weights):
            assert self.named and self.bounded_retention_mode == "control"
            events.append(("load", weights))
    monkeypatch.setattr(integration, "BoundedRetentionModel", Stub)
    trainer = object.__new__(integration.BoundedRetentionTrainer)
    trainer.data = {"nc": 2, "channels": 3}
    trainer.p1_options, trainer.mode = {"soft_weights": True, "uncertain_ignore": True}, "control"
    def names(model):
        model.named = True
        events.append(("names",))
        return model
    trainer.set_model_names_for_load = names
    model = trainer.get_model("synthetic", "synthetic-weights", verbose=False)
    assert model.p1_options == trainer.p1_options
    assert events == [("create", "synthetic", 2, 3), ("names",), ("load", "synthetic-weights")]


def test_eval_loss_always_dispatches_original_p1_without_train_metadata(monkeypatch):
    calls = []
    class Loss:
        def __init__(self, model):
            calls.append(model)
        def __call__(self, preds, batch):
            return preds, batch
    monkeypatch.setattr(integration, "P1DetectionLoss", Loss)
    model = SimpleNamespace(training=False, criterion=object())
    batch = {"cls": torch.zeros((0, 1)), "bboxes": torch.zeros((0, 4))}
    assert integration.BoundedRetentionModel.loss(model, batch, "preds") == ("preds", batch)
    assert calls == [model]


@pytest.mark.parametrize("mode", ["reference", "candidate", "control"])
def test_criterion_connects_real_objective_without_constructing_detector(mode):
    from regionssod.training.bounded_retention_objective import BoundedRetentionLoss
    model = torch.nn.Module()
    model.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1)))
    model.model = [SimpleNamespace(stride=torch.tensor([8.]), nc=2, reg_max=16)]
    model.args = SimpleNamespace(box=7.5, cls=.5, dfl=1.5)
    model.p1_options = {"soft_weights": True, "uncertain_ignore": True}
    model.bounded_retention_mode = mode
    criterion = integration.BoundedRetentionModel.init_criterion(model)
    assert type(criterion) is BoundedRetentionLoss and criterion.mode == mode


def test_training_loss_propagates_whole_object_rejection():
    from regionssod.training.bounded_retention import UnsupportedObservation
    # Allocate only the Module shell; no detector constructor/forward/weights.
    model = object.__new__(integration.BoundedRetentionModel)
    torch.nn.Module.__init__(model)
    def rejected(preds, batch):
        raise UnsupportedObservation("partially unknown object has no legal discrete anchor event")
    model.criterion = rejected
    with pytest.raises(UnsupportedObservation, match="no legal discrete anchor event"):
        model.loss({}, preds={"synthetic": True})


@pytest.mark.parametrize("overrides", [{"resume": True}, {"workers": 1}])
def test_trainer_rejects_resume_and_workers_before_construction(synthetic, overrides):
    with pytest.raises(ValueError, match="resume|workers"):
        integration.BoundedRetentionTrainer(mode="candidate", **synthetic.source_options,
            overrides=overrides)


def test_run_training_refuses_existing_or_ungated_output(tmp_path, monkeypatch):
    # Use a fresh path name under the repository's ignored outputs, but never
    # create it. Monkeypatch exists for the existing-run branch only.
    config = {"name": "synthetic-only-no-run", "project": str(integration.ROOT / "outputs"),
        "data": "not-opened", "model": "not-opened", "p1": {}, "train": {},
        "bounded_retention": {"mode": "candidate"}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="gates required"):
        integration.run_training(path, source_factory=fixture_factory)
    run = Path(config["project"]) / config["name"]
    original_exists = Path.exists
    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", lambda p: True if p == run else original_exists(p))
        with pytest.raises(FileExistsError):
            integration.run_training(path, source_factory=fixture_factory, engineering_gates_passed=True)
    config["name"] = "../old-run"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="single fresh"):
        integration.run_training(path, source_factory=fixture_factory)
