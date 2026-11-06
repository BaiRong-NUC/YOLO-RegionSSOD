"""Read-only qualification of completed pressure artifacts; no model or loss execution.

Example (run only after the owning session authorizes the audit):
  python -B scripts/audit_bounded_retention_point_pressure.py --run \
    outputs/bounded_retention_pressure_20261106/audit_02 --output \
    outputs/bounded_retention_pressure_20261106/qualification_01

Only the fresh qualification directory is written. An absent/failed terminal
returns STOP before importing torch or loading any tensor/checkpoint. Local
pickle artifacts are trusted only within this explicitly selected experiment.
Gradient finiteness and restoration are checked against saved receipts, not
recomputed gradients. Missing evidence never grants qualification or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs/bounded_retention_pressure_20261106"
PREFIX = ROOT / "outputs/bounded_retention_point_review_20261106/real_03"
GEOMETRY = ROOT / "outputs/bounded_retention_continuous_20261105/native_e_integration_01/result.json"
AMP = ROOT / "outputs/bounded_retention_continuous_20261105/native_amp_audit_01/result.json"
EPOCH9 = ROOT / "outputs/bounded_retention_continuous_three30_s0/candidate/weights/last.pt"
INITIAL_WEIGHTS = Path("D:/Python/YOLO/yolo11s.pt")
PRESSURE = ROOT / "scripts/check_bounded_retention_point_pressure.py"
PROTOCOL = ROOT / "docs/paper/candidates/bounded_retention/point_pressure_protocol.md"
ARMS = ("p1", "new_candidate", "new_control")
DFL_PARAMETER = "model.23.dfl.conv.weight"
LEGACY_PARAMETERS = "ef8b44bff4cfb9cb5dc7cb0e575a9d6f37f58ed9d8719f0609e0bd90e8481f39"
PREFIX_PAIRS_SHA = "a551700a05ae2dc4c920a21ceac49d01bb523449792927ad9e1b28a656960c15"
# These bytes were inspected before pressure execution, not learned from its result.
PINS = {
    GEOMETRY: "39c03aa3a4243a0f3169abe174c4701d964f905df749c208148fc0bccee7d259",
    AMP: "60ef98ace4414b1636baa04afd56d1488d934f93edf833c16c4e74c92a6b6504",
    EPOCH9: "8b27ba426c82315c62d1120accdb05921a2d8af0427012a84ef277a963a8b56e",
    INITIAL_WEIGHTS: "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
    PREFIX / "terminal.json": "037d98e926244b84e2a4dc7addded2a29df1d29c67e67b3b4c3ac148cb6511ee",
    PREFIX / "manifest.json": "a41520619769ca140d57e916f81c9bfd43902a3a6154b87a491c0d352f9b67e8",
    PREFIX / "worker_result.json": "7e9080d7930212a179457096667d8155887662828f0285516a69c617a0a44bff",
    PREFIX / "initial_state.pt": "ce3812427bbdb91dfa2c9653436733a1eef298ec77b46fc78dd8cba0dab05465",
    PREFIX / "source_factory.json": "e22d003fee604eb0413ef265d2fcdbc355eda3fedaf1af4de97810baebb3c501",
    PROTOCOL: "e77038b0d24a39639878a1d6dd49a454a7abd375a18b883e282e0dbc4fb1e478",
}


class QualificationStop(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise QualificationStop(message)


def fixed_cases():
    return [("initial", "pressure"), ("epoch9", "pressure")] + [
        ("epoch9", f"{phase}_{i:02d}") for phase in ("open", "closed") for i in range(16)]


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Evidence:
    """Capture consumed bytes and reject changes, including changes during torch.load."""

    def __init__(self):
        self.hashes = {}

    def pin(self, path, expected=None):
        path = Path(path).resolve()
        digest = file_sha(path)
        require(expected is None or digest == expected, f"SHA mismatch: {path}")
        require(str(path) not in self.hashes or self.hashes[str(path)] == digest,
                f"evidence changed during audit: {path}")
        self.hashes[str(path)] = digest
        return digest

    def json(self, path):
        self.pin(path)
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        self.pin(path)
        return result

    def tensor(self, path):
        self.pin(path)
        # Do not import the pressure runner or construct a Trainer/Model.
        import torch
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        result = torch.load(path, map_location="cpu", weights_only=False)
        self.pin(path)
        return result

    def unchanged(self):
        for path, digest in self.hashes.items():
            require(file_sha(path) == digest, f"evidence changed during audit: {path}")


def normalize_hashes(mapping):
    result = {}
    for name, digest in mapping.items():
        path = Path(name).resolve()
        require(path not in result and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid input SHA map")
        result[path] = digest
    return result


def pointer(evidence, value, path):
    require(Path(value["path"]).resolve() == Path(path).resolve(), f"artifact pointer substituted: {path}")
    evidence.pin(path, value["sha256"])
    if "bytes" in value:
        require(Path(path).stat().st_size == value["bytes"], f"artifact size changed: {path}")


def prefix_pairs_digest(pairs):
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def verify_history(evidence, manifest, run):
    for path, digest in PINS.items():
        evidence.pin(path, digest)
    amp = evidence.json(AMP)
    old_manifest = evidence.json(PREFIX / "manifest.json")
    old_worker = evidence.json(PREFIX / "worker_result.json")
    old_terminal = evidence.json(PREFIX / "terminal.json")
    require(old_terminal["status"] == "PASSED" and old_terminal["worker_result"] == old_worker,
            "historical terminal is inconsistent")
    expected = normalize_hashes(old_manifest["source_sha256"])
    # The pressure entry adds only these inputs to the immutable real_03 source map.
    additions = [PRESSURE, PROTOCOL, GEOMETRY, AMP, EPOCH9, INITIAL_WEIGHTS,
                 PREFIX / "terminal.json", PREFIX / "manifest.json"]
    pairs = []
    for phase in ("open", "closed"):
        for i in range(16):
            case = f"{phase}_{i:02d}"
            directory = PREFIX / case
            receipt = evidence.json(directory / "result.json")
            require(receipt["status"] == "COMPLETED" and receipt["state_restored"] is True,
                    f"historical batch incomplete: {case}")
            pointer(evidence, receipt["raw_batch"], directory / "batch.pt")
            pairs.append([case, evidence.pin(directory / "result.json"), receipt["raw_batch"]["sha256"]])
            additions.extend([directory / "result.json", directory / "batch.pt"])
    require(prefix_pairs_digest(pairs) == PREFIX_PAIRS_SHA, "historical 32 receipt/batch roots changed")
    expected.update({p.resolve(): evidence.pin(p) for p in additions})
    actual = normalize_hashes(manifest["source_sha256"])
    require(actual == expected, "pressure original input map differs from frozen historical sources")
    for path, digest in expected.items():
        evidence.pin(path, digest)
    # Inspect only the already pinned AMP input allowlist; never follow nested data manifests.
    for path, digest in normalize_hashes(amp["input_sha256"]).items():
        require(not path.is_relative_to(ROOT / "data"), "unexpected raw-data input in AMP receipt")
        evidence.pin(path, digest)
    for path, digest in normalize_hashes(old_manifest["source_sha256"]).items():
        relative = (Path("repo") / path.relative_to(ROOT) if path.is_relative_to(ROOT)
                    else Path("ultralytics") / Path(*path.parts[path.parts.index("ultralytics") + 1:]))
        evidence.pin(run / "source_snapshot" / relative, digest)
    for path in (PRESSURE, PROTOCOL):
        evidence.pin(run / "source_snapshot/repo" / path.relative_to(ROOT), actual[path.resolve()])
    old_source = evidence.json(PREFIX / "source_factory.json")
    source = evidence.json(run / "source_factory.json")
    require(all(source[k] == old_source[k] for k in ("reads", "bindings", "counts")),
            "legal source binding differs from frozen real_03")
    for path, digest in normalize_hashes(old_source["reads"]).items():
        evidence.pin(path, digest)
    return amp, old_worker


def tensor_hash(value):
    import torch
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(repr((str(value.dtype), tuple(value.shape))).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def named_hash(items):
    h = hashlib.sha256()
    for name, value in items:
        h.update(name.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


def verify_state_equal(actual, expected):
    import torch
    require(list(actual) == list(expected), "state key/order mismatch")
    for name, value in actual.items():
        other = expected[name]
        require(isinstance(value, torch.Tensor) and isinstance(other, torch.Tensor), "non-tensor state")
        require(value.dtype == other.dtype and value.shape == other.shape and torch.equal(value, other),
                f"state tensor mismatch: {name}")


def parameter_names(amp):
    attempts = amp["amp_backward_audit"]["attempts"]
    selected = [a for a in attempts if a["scale"] == 1024]
    require(len(selected) == 1, "historical scale1024 attempt missing/ambiguous")
    names = list(selected[0]["arms"]["candidate"]["parameters"]["by_name"])
    require(len(names) == 255 and DFL_PARAMETER not in names, "unexpected historical parameter schema")
    for arm in ("reference", "candidate", "control"):
        require(list(selected[0]["arms"][arm]["parameters"]["by_name"]) == names,
                "historical parameter orders differ")
    return names


def state_hashes(state, trainable):
    parameters = set(trainable) | {DFL_PARAMETER}
    require(parameters <= set(state), "state missing model parameter")
    buffers = [(n, v) for n, v in state.items() if n not in parameters]
    require(all(n.endswith(("running_mean", "running_var", "num_batches_tracked")) for n, _ in buffers),
            "unrecognized non-parameter state; cannot certify buffers")
    return {"parameters": named_hash((n, v) for n, v in state.items() if n in parameters),
            "buffers": named_hash(buffers), "bn_buffers": named_hash(buffers),
            "state_dict": named_hash(state.items())}


def legacy_digest(state, trainable):
    names = trainable + [DFL_PARAMETER]
    require([n for n in state if n in set(names)] == names, "parameter traversal order changed")
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode("utf-8"))
        h.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def verify_states(evidence, run, amp, old_worker):
    import torch
    trainable = parameter_names(amp)
    historical = evidence.tensor(PREFIX / "initial_state.pt")
    initial = evidence.tensor(run / "initial_state.pt")
    require(legacy_digest(historical["state_dict"], trainable) == LEGACY_PARAMETERS ==
            amp["parameters_before_forward_sha256"], "historical initial parameter digest mismatch")
    require(state_hashes(historical["state_dict"], trainable) == historical["hashes"] ==
            old_worker["initial_hashes"], "historical full-state hashes mismatch")
    verify_state_equal(initial["state_dict"], historical["state_dict"])
    require(initial["model_yaml"] == historical["model_yaml"], "initial architecture changed")
    epoch9 = evidence.tensor(run / "epoch9_state.pt")
    checkpoint = evidence.tensor(EPOCH9)
    require(checkpoint.get("epoch") == 8 and checkpoint.get("ema") is not None,
            "epoch-nine saved EMA missing")
    ema = checkpoint["ema"]
    require(ema.nc == 10 and ema.model[-1].reg_max == 16 and not ema.model[-1].end2end,
            "unexpected EMA architecture")
    require(ema.names == dict(enumerate(("pedestrian", "people", "bicycle", "car", "van", "truck",
                                        "tricycle", "awning-tricycle", "bus", "motor"))),
            "EMA class order differs from frozen VisDrone architecture")
    require(ema.yaml == initial["model_yaml"] == epoch9["model_yaml"], "EMA architecture mismatch")
    # Conversion of saved values only; no model construction, .forward(), .train(), or optimizer access.
    ema_state = {n: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
                 for n, v in ema.state_dict().items()}
    verify_state_equal(epoch9["state_dict"], ema_state)
    require(all(torch.isfinite(v).all().item() for s in (initial, epoch9)
                for v in s["state_dict"].values() if v.is_floating_point()), "nonfinite saved model state")
    hashes = {"initial": state_hashes(initial["state_dict"], trainable),
              "epoch9": state_hashes(epoch9["state_dict"], trainable)}
    shapes = {n: list(initial["state_dict"][n].shape) for n in trainable}
    return trainable, shapes, hashes


def positive_3u_exposure(payload, batch=None):
    """Join posterior events to support identity, local row, global anchor and TAL weight."""
    import torch
    geometry, certificates = payload["geometry"], payload["certificates"]
    assignment = payload["assignment"]
    foreground, gt, scores = (assignment[k] for k in ("foreground", "gt_index", "scores"))
    require(foreground.ndim == gt.ndim == 2 and scores.ndim == 3 and
            foreground.shape == gt.shape == scores.shape[:2], "assignment shapes inconsistent")
    require(foreground.dtype == torch.bool and gt.dtype in (torch.int32, torch.int64),
            "assignment mask/index dtype invalid")
    supports = {}
    for key, certificate in certificates.items():
        require(re.fullmatch(r"\d+:\d+", key), "invalid support key")
        identity = tuple(certificate["identity"])
        require(identity not in supports, "ambiguous source/occurrence identity")
        supports[identity] = key
    exposures, seen = [], set()
    events = payload["criterion_diagnostics"].get("posterior_event_details", [])
    for event in events:
        identity = tuple(event["identity"])
        require(identity in supports, "posterior event has no support certificate")
        key = supports[identity]
        mask, indices = geometry[key + "/mask"], geometry[key + "/indices"]
        require(mask.dtype == torch.bool and tuple(mask.shape) == (4,), "invalid support mask")
        row = event["anchor_row"]
        require(isinstance(row, int) and not isinstance(row, bool) and 0 <= row < len(indices),
                "posterior anchor_row is out of range")
        require((identity, row) not in seen, "duplicate posterior event")
        seen.add((identity, row))
        if int(mask.sum()) != 3:
            continue
        image, target = map(int, key.split(":"))
        require(indices.ndim == 1 and indices.dtype in (torch.int32, torch.int64), "invalid anchor indices")
        anchor = int(indices[row])
        require(0 <= image < foreground.shape[0] and 0 <= anchor < foreground.shape[1],
                "3U anchor is out of assignment bounds")
        require(bool(foreground[image, anchor]) and int(gt[image, anchor]) == target,
                "3U posterior was not assigned to its source target")
        if batch is not None:
            targets = (batch["batch_idx"].reshape(-1) == image).nonzero().reshape(-1)
            require(target < len(targets), "3U target absent from saved batch")
            record = batch["retention_observations"][int(targets[target])]
            require(record is not None and (record["source_object_id"], record["occurrence_id"]) == identity,
                    "3U certificate identity differs from saved target")
        values = scores[image, anchor]
        require(bool(torch.isfinite(values).all()) and bool((values >= 0).all()), "invalid 3U TAL scores")
        weight = float(values.sum())
        require(math.isfinite(weight), "nonfinite 3U TAL weight")
        if weight > 0:
            exposures.append({"support": key, "identity": list(identity), "anchor_row": row,
                              "anchor_index": anchor, "tal_weight": weight})
    require(exposures, "no posterior-consumed 3U anchor with finite positive TAL weight")
    return exposures


def verify_arm_receipt(receipt, arm, ordinal, parameter_shapes):
    require(receipt["status"] == "COMPLETED" and receipt["arm"] == arm, "arm incomplete/substituted")
    require(receipt["gradient_scale"] == 1024 and receipt["posterior_call_number"] == ordinal,
            "scale or posterior call changed")
    require(receipt["unused_parameters"] == [] and receipt["gradient_finite"] is True and
            receipt["model_grad_storage_untouched"] is True, "unused/nonfinite/accumulated gradients")
    gradients = receipt["gradients"]
    shapes = {"head/boxes": [8, 64, 8400], "head/scores": [8, 10, 8400],
              **{f"parameter/{k}": v for k, v in parameter_shapes.items()}}
    require(set(gradients) == set(shapes), "full-model gradient coverage mismatch")
    for name, stats in gradients.items():
        require(stats["status"] == "OBSERVED" and stats["finite"] is True and stats["nonfinite"] == 0,
                f"gradient not observed/finite: {name}")
        require(stats["shape"] == shapes[name] and stats["numel"] == math.prod(shapes[name]) and
                stats["dtype"] == "torch.float64" and re.fullmatch(r"[0-9a-f]{64}", stats["sha256"]),
                f"gradient metadata mismatch: {name}")
    require(receipt["full_loss_stats"]["finite"] is True and
            len(receipt["full_loss"]) == 3 and all(math.isfinite(v) for v in receipt["full_loss"]),
            "nonfinite full loss")
    for name in ("loss_seconds", "backward_seconds"):
        require(math.isfinite(receipt[name]) and 0 <= receipt[name] <= 120, f"deadline exceeded: {name}")


def tensor_mapping_equal(a, b):
    import torch
    return set(a) == set(b) and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and
                                   torch.equal(a[k], b[k]) for k in a)


def amp_fingerprints(mapping):
    import torch
    return {n: hashlib.sha256(str((v.dtype, tuple(v.shape))).encode() +
                             v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, v in mapping.items() if isinstance(v, torch.Tensor)}


def saved_tree_equal(actual, expected):
    """Compare saved metadata as well as tensors; no reconstruction or transform."""
    import numpy as np
    import torch
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, torch.Tensor):
        return actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(actual, expected)
    if isinstance(expected, np.ndarray):
        return actual.dtype == expected.dtype and np.array_equal(actual, expected)
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(saved_tree_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(saved_tree_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def verify_request_batch(actual, raw):
    import torch
    require(set(actual) == set(raw), "request batch keys changed")
    for name, value in raw.items():
        if not isinstance(value, torch.Tensor):
            continue
        other = actual[name]
        require(other.shape == value.shape, f"request batch shape changed: {name}")
        if name == "img":
            require(value.dtype == torch.uint8 and other.dtype == torch.float32 and
                    torch.allclose(other.double(), value.double() / 255, rtol=0, atol=1.2e-7),
                    "request image differs from normalized historical input")
        else:
            require(other.dtype == value.dtype and torch.equal(other, value), f"request batch changed: {name}")
    identity = lambda b: [None if r is None else (r["source_object_id"], r["occurrence_id"])
                          for r in b["retention_observations"]]
    require(identity(actual) == identity(raw), "request observation identities changed")
    for name, value in raw.items():
        if not isinstance(value, torch.Tensor):
            require(saved_tree_equal(actual[name], value), f"request metadata changed: {name}")


def audit_completed(evidence, run, terminal):
    import torch
    result, manifest = evidence.json(run / "result.json"), evidence.json(run / "manifest.json")
    require(terminal["returncode"] == 0 and terminal["source_verified_after"] is True and
            terminal["training_started"] is False and terminal["worker_result"] == result,
            "terminal/worker receipt mismatch")
    require(result["status"] == "PASSED_PRESSURE_ZERO_TRAINING" and
            result["completed_cases"] == result["explicit_forwards"] == 34 and
            result["optimizer_steps"] == result["validation_inference_calls"] == 0 and
            result["training_started"] is False and result["diagnostic_scale"] == 1024 and
            all(result[k] is True for k in ("all_sources_unchanged", "all_state_restored",
                                            "original_initial_state_restored")), "incomplete pressure run")
    require(manifest["cases"] == [list(c) for c in fixed_cases()] and manifest["scale"] == 1024 and
            manifest["timeout"] == 120, "fixed protocol population changed")
    amp, old_worker = verify_history(evidence, manifest, run)
    trainable, shapes, hashes = verify_states(evidence, run, amp, old_worker)
    require(result["legacy_initial_parameter_sha256"] == LEGACY_PARAMETERS,
            "run initial digest differs from historical AMP")
    replay = evidence.json(run / "replay.json")
    require(replay["counts"] == {"mosaic_open": 1414, "mosaic_closed": 1414} and
            replay["position"] == amp["forward_batch_selection"] and
            replay["tensor_sha256"] == amp["batch_tensor_sha256"] and replay["identity"] == amp["batch_identity"],
            "historical pressure replay receipt mismatch")
    pointer(evidence, replay["batch"], run / "pressure_batch.pt")
    pressure = evidence.tensor(run / "pressure_batch.pt")
    require(amp_fingerprints(pressure) == amp["batch_tensor_sha256"] and len(pressure["cls"]) == 63,
            "saved pressure batch differs from frozen AMP")
    require([None if r is None else [r["source_object_id"], r["occurrence_id"]]
             for r in pressure["retention_observations"]] == amp["batch_identity"],
            "saved pressure identities differ from frozen AMP")
    require(len(result["cases"]) == 34, "case list incomplete")
    exposure = {}
    for ordinal, (weights, case) in enumerate(fixed_cases(), 1):
        directory = run / f"{ordinal:02d}_{weights}_{case}"
        entry = result["cases"][ordinal - 1]
        require((entry["weights"], entry["case"], entry["ordinal"]) == (weights, case, ordinal) and
                Path(entry["result"]).resolve() == directory / "result.json", "case order/path changed")
        row, state = evidence.json(directory / "result.json"), evidence.json(directory / "state.json")
        require((row["weights"], row["case"], row["ordinal"]) == (weights, case, ordinal) and
                row["state_restored"] is True and state["restored"] is True and
                state["before"] == state["after"] == hashes[weights], "case parameter/BN restoration mismatch")
        request = evidence.tensor(directory / "request.pt")
        require(request["before"] == hashes[weights] and request["state"] == weights and
                request["diagnostic_scale"] == 1024 and request["source_sha256"] == manifest["source_sha256"],
                "request state/input binding mismatch")
        raw = pressure if case == "pressure" else evidence.tensor(PREFIX / case / "batch.pt")
        verify_request_batch(request["cpu_batch"], raw)
        for name in ("boxes", "scores"):
            require(torch.isfinite(request["predictions"][name]).all().item(), "nonfinite saved prediction")
        require(set(row["arms"]) == set(ARMS), "arm population changed")
        payloads = {}
        for arm in ARMS:
            receipt = evidence.json(directory / f"{arm}.json")
            require(receipt == row["arms"][arm], "case/arm receipt disagreement")
            verify_arm_receipt(receipt, arm, ordinal, shapes)
            pointer(evidence, receipt["observation_artifact"], directory / f"{arm}.loss.pt")
            payload = evidence.tensor(directory / f"{arm}.loss.pt")
            require(torch.isfinite(payload["full_loss"]).all().item() and
                    payload["full_loss"].tolist() == receipt["full_loss"], "saved loss differs from receipt")
            payloads[arm] = payload
        a, b = payloads["new_candidate"], payloads["new_control"]
        require(tensor_mapping_equal(a["assignment"], b["assignment"]) and
                tensor_mapping_equal(a["geometry"], b["geometry"]) and
                torch.equal(a["full_loss"][:2], b["full_loss"][:2]), "saved shared TAL/geometry/loss differ")
        if case == "pressure":
            for arm in ("new_candidate", "new_control"):
                require(row["arms"][arm]["point_search"]["by_dimension"].get("3", {}).get("calls", 0) > 0,
                        "pressure case has no actual 3U point request")
            exposure[weights] = positive_3u_exposure(a, request["cpu_batch"])
            ga, gb = (row["arms"][arm]["gradients"] for arm in ("new_candidate", "new_control"))
            require(any(ga[n]["sha256"] != gb[n]["sha256"] for n in ga), "pressure gradients are identical")
            require(row["comparisons"]["gradient"]["equal"] is False, "pressure gradient comparison inconsistent")
        del request, raw, payloads, a, b, payload
    require(set(exposure) == {"initial", "epoch9"}, "both pressure states need positive 3U exposure")
    return {"checked_cases": 34, "checked_arms": 102, "trainable_parameter_tensors": len(trainable),
            "positive_3u_exposure": exposure, "legacy_initial_parameter_sha256": LEGACY_PARAMETERS,
            "state_hashes": hashes, "historical_32_pairs_sha256": PREFIX_PAIRS_SHA}


def audit_run(run):
    run = Path(run).resolve()
    evidence = Evidence()
    report = {"status": "STOP", "accepted": False, "run": str(run), "training_permit": False,
              "scope": "saved artifacts only; no model/forward/loss/backward/optimizer execution",
              "limitations": ["Gradient finiteness and restoration use saved execution receipts; gradients are not recomputed.",
                              "34 explicit forwards exclude constructor/THOP probes.",
                              "Qualification is not a performance result, future reliability guarantee, or training permission."]}
    try:
        require((run / "terminal.json").is_file(), "terminal missing: run unfinished; no tensor artifacts loaded")
        terminal = evidence.json(run / "terminal.json")
        require(terminal.get("status") == "PASSED", "terminal did not pass; no tensor artifacts loaded")
        report["checks"] = audit_completed(evidence, run, terminal)
        evidence.unchanged()
        report.update(status="QUALIFIED_SAVED_PRESSURE_EVIDENCE", accepted=True)
    except Exception as exc:
        report["reason"] = f"{type(exc).__name__}: {exc}"
    report["evidence_sha256"] = evidence.hashes
    return report


def validate_paths(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    require(run.parent == OUT_ROOT and re.fullmatch(r"audit_\d{2}", run.name), "run must be audit_NN under fixed root")
    require(output.parent == OUT_ROOT and re.fullmatch(r"qualification_\d{2}", output.name),
            "output must be a fresh qualification_NN sibling under ignored fixed root")
    require(not output.exists(), "refusing to overwrite qualification output")
    return run, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run, output = validate_paths(args.run, args.output)
    auditor_sha = file_sha(__file__)
    output.mkdir(parents=True, exist_ok=False)
    report = audit_run(run)
    report["auditor_sha256"] = auditor_sha
    if file_sha(__file__) != auditor_sha:
        report.update(status="STOP", accepted=False, reason="auditor changed during audit")
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "accepted": report["accepted"],
                      "report": str(output / "report.json")}, ensure_ascii=False))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
