"""Independent cached pressure entry; NEVER train or resume.

Copied from the protected point-pressure runner. Only the solver import, output root,
protocol and explicit source bindings differ; numerical guards are inherited.
Uses immutable historical batch receipts, not hidden training GT. Every loss and
backward has a 120s deadline. The parent only owns its diagnostic child process.
"""
from __future__ import annotations
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import secrets
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import check_bounded_retention_point_real as base
from check_bounded_retention_point_real import (
    ROOT, CONFIG, INITIAL_WEIGHTS, require, sha, write_json, exception_info,
    Heartbeat, latest_heartbeat, verify_sources, cpu_tree, save_tensor_file,
    model_hashes, clear_numerical_caches, finite_stats, geometry_snapshot,
    mapping_difference, tensor_difference,
)

OUT_ROOT = ROOT / 'outputs/bounded_retention_cache_20261106'
PROTOCOL = ROOT / 'docs/paper/candidates/bounded_retention/point_geometry_cache_protocol.md'
# Explicit additions only; historical sources remain hash-bound and unchanged.
LEGACY_PRESSURE = ROOT / 'scripts/check_bounded_retention_point_pressure.py'
LEGACY_AUDITOR = ROOT / 'scripts/audit_bounded_retention_point_pressure.py'
LEGACY_PROTOCOL = ROOT / 'docs/paper/candidates/bounded_retention/point_pressure_protocol.md'
REPLAY_ADDENDUM = ROOT / 'docs/paper/candidates/bounded_retention/point_pressure_replay_addendum.md'
ACCEPTANCE_ADDENDUM = ROOT / 'docs/paper/candidates/bounded_retention/point_pressure_acceptance_addendum.md'
CACHE_SOLVER = ROOT / 'src/regionssod/training/bounded_retention_point_geometry_cache.py'
CACHED_PRESSURE = ROOT / 'scripts/check_bounded_retention_cached_pressure.py'
CACHED_AUDITOR = ROOT / 'scripts/audit_bounded_retention_cached_pressure.py'
BINDING_SOURCES = (CACHED_PRESSURE, CACHED_AUDITOR, LEGACY_PRESSURE, LEGACY_AUDITOR,
                   LEGACY_PROTOCOL, REPLAY_ADDENDUM, ACCEPTANCE_ADDENDUM, PROTOCOL, CACHE_SOLVER)
GEOMETRY = ROOT / 'outputs/bounded_retention_continuous_20261105/native_e_integration_01/result.json'
AMP = ROOT / 'outputs/bounded_retention_continuous_20261105/native_amp_audit_01/result.json'
PREFIX = ROOT / 'outputs/bounded_retention_point_review_20261106/real_03'
EPOCH9 = ROOT / 'outputs/bounded_retention_continuous_three30_s0/candidate/weights/last.pt'
EPOCH9_SHA = '8b27ba426c82315c62d1120accdb05921a2d8af0427012a84ef277a963a8b56e'
ARMS = ('p1', 'new_candidate', 'new_control')
SCALE = 1024.
TIMEOUT = 120
TOKEN = 'REGIONSSOD_PRESSURE_WORKER_TOKEN'


def fixed_cases():
    return [('initial', 'pressure'), ('epoch9', 'pressure')] + [
        ('epoch9', f'{phase}_{i:02d}') for phase in ('open', 'closed') for i in range(16)]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_epoch_checkpoint(checkpoint):
    require(checkpoint.get('epoch') == 8, 'checkpoint must be saved epoch nine, zero-based eight')
    require(checkpoint.get('ema') is not None, 'saved EMA is required; no model/optimizer fallback')
    return checkpoint['ema']


def tensor_fingerprints(mapping):
    # Match the exact historical AMP receipt representation.
    result = {}
    for name, value in mapping.items():
        if isinstance(value, torch.Tensor):
            h = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
            result[name] = h.hexdigest()
    return result


def supervision_profile(criterion):
    profile = {}
    for support in getattr(criterion, 'supports', {}).values():
        dimension = str(support['domain'].dimension)
        count = len(support.get('density_diagnostics', {}))
        profile[dimension] = profile.get(dimension, 0) + count
    return profile


def shared_comparison(results):
    a, b = results['new_candidate'], results['new_control']
    shared = {name: mapping_difference(a[name], b[name]) for name in ('assignment', 'geometry')}
    shared['ciou_classification'] = tensor_difference(a['loss'][:2], b['loss'][:2])
    shared['gradient'] = mapping_difference(a['gradients'], b['gradients'])
    require(all(shared[k]['equal'] for k in ('assignment', 'geometry', 'ciou_classification')),
            'candidate/control do not share geometry/TAL/CIoU/classification')
    return shared


def replay_pressure(loader, trainer, receipt, amp, beat, output):
    from regionssod.training.bounded_retention_objective_v6 import current_unknown
    expected = iter(receipt['geometry_batch_checks'])
    selected, maximum, position = None, -1, None
    counts = {}
    for phase in ('mosaic_open', 'mosaic_closed'):
        if phase == 'mosaic_closed':
            trainer.train_loader = loader
            trainer._close_dataloader_mosaic()
            loader.reset()
        count = 0
        for index, batch in enumerate(loader):
            beat('transport_replay', phase=phase, batch_index=index)
            count += 1
            height, width = batch['img'].shape[-2:]
            dims = Counter()
            for record in batch['retention_observations']:
                if record is not None:
                    d = int(current_unknown(record, width, height)[0].sum())
                    if d:
                        dims[str(d)] += 1
            row = next(expected)
            if not (row['phase'] == phase and dict(dims) == row['active_dimensions']
                    and sum(dims.values()) == row['partial_objects_checked']):
                write_json(output / 'replay_mismatch.json', dict(phase=phase, batch_index=index,
                    actual_dimensions=dict(dims), expected=row,
                    identity=[None if r is None else [r['source_object_id'], r['occurrence_id']]
                              for r in batch['retention_observations']]))
                save_tensor_file(output / 'replay_mismatch.pt', cpu_tree(batch))
            require(row['phase'] == phase and dict(dims) == row['active_dimensions']
                    and sum(dims.values()) == row['partial_objects_checked'],
                    f'transport replay mismatch: {phase}/{index}')
            d = max(map(int, dims), default=0)
            if d > maximum:
                selected, maximum = cpu_tree(batch), d
                position = dict(rule='first_maximum_active_dimension', phase=phase,
                                batch_index=index, active_dimension=d)
        counts[phase] = count
    require(next(expected, None) is None and counts == {'mosaic_open': 1414, 'mosaic_closed': 1414},
            'incomplete frozen transport replay')
    require(position == receipt['forward_batch_selection'] == amp['forward_batch_selection']
            and maximum == 3, 'fixed pressure batch selection changed')
    require(tensor_fingerprints(selected) == amp['batch_tensor_sha256'], 'pressure batch tensors changed')
    identity = [None if r is None else [r['source_object_id'], r['occurrence_id']]
                for r in selected['retention_observations']]
    require(identity == amp['batch_identity'] and len(selected['cls']) == 63, 'pressure identity changed')
    pointer = save_tensor_file(output / 'pressure_batch.pt', selected)
    write_json(output / 'replay.json', dict(counts=counts, position=position,
               identity=identity, batch=pointer, tensor_sha256=tensor_fingerprints(selected)))
    return selected


def worker(output):
    global torch, np
    beat = Heartbeat(output)
    beat('authenticate')
    manifest = read_json(output / 'manifest.json')
    token = os.environ.pop(TOKEN, '')
    require(os.getppid() == manifest['parent_pid'] and token and
            hashlib.sha256(token.encode()).hexdigest() == manifest['token_sha256'], 'unauthenticated worker')
    report = dict(status='RUNNING', training_started=False, optimizer_steps=0,
                  validation_inference_calls=0, explicit_forwards=0, completed_cases=0,
                  diagnostic_scale=SCALE, cases=[], initial_and_epoch9_only=True)
    start = time.perf_counter()
    code = 2
    try:
        beat('imports')
        sys.path.insert(0, str(ROOT / 'src'))
        import torch
        import numpy as np
        import yaml
        import ultralytics
        base.torch, base.np = torch, np
        from ultralytics.data import dataset as dataset_module
        from regionssod.training.bounded_retention_training_v6 import ContinuousNativeETrainer
        from regionssod.training.bounded_retention_source import BoundSourceFactory, AUDIT, AUDIT_SHA
        verify_sources(manifest)
        require(torch.cuda.is_available(), 'CUDA required')
        report['runtime'] = dict(torch=torch.__version__, ultralytics=ultralytics.__version__,
            gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
            threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads())
        config = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
        parent, amp = read_json(GEOMETRY), read_json(AMP)
        beat('source_factory')
        factory = BoundSourceFactory()

        def progress_factory(**kwargs):
            value = factory(**kwargs)
            if factory.counts['objects'] % 5000 == 0:
                beat('source_binding', objects=factory.counts['objects'])
            return value

        class AuditTrainer(ContinuousNativeETrainer):
            def train(self, *args, **kwargs):
                raise RuntimeError('training forbidden')

            def build_optimizer(self, *args, **kwargs):
                raise RuntimeError('optimizer forbidden')

            def get_validator(self, *args, **kwargs):
                raise RuntimeError('validation inference forbidden')

        beat('trainer_setup')
        # Preserve native registration after init_seeds: events module import
        # consumes Python RNG, so suppressing it changes historical augmentation.
        trainer = AuditTrainer(mode='candidate', source_factory=progress_factory,
            source_audit_path=AUDIT, source_audit_sha256=AUDIT_SHA, p1_options=config['p1'],
            overrides={**config['train'], 'model': config['model'],
                'data': str(ROOT / config['data']), 'project': str(output), 'name': 'trainer',
                'save': False, 'plots': False, 'val': True, 'resume': False}, _callbacks={})
        trainer.setup_model()
        trainer.model = trainer.model.to(trainer.device)
        trainer.set_model_attributes()
        model = trainer.model
        require(trainer.device.type == 'cuda' and trainer.optimizer is None, 'wrong device/optimizer')
        initial = cpu_tree(model.state_dict())
        initial_hash = model_hashes(model)
        legacy_hash = hashlib.sha256()
        for name, parameter in model.named_parameters():
            legacy_hash.update(name.encode('utf-8'))
            legacy_hash.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        report['legacy_initial_parameter_sha256'] = legacy_hash.hexdigest()
        require(legacy_hash.hexdigest() == amp['parameters_before_forward_sha256'] ==
                'ef8b44bff4cfb9cb5dc7cb0e575a9d6f37f58ed9d8719f0609e0bd90e8481f39',
                'historical initial model parameters changed')
        beat('loader_setup')
        original_save = dataset_module.save_dataset_cache_file

        def redirected_cache(prefix, path, data, version):
            destination = output / 'dataset_cache' / (hashlib.sha256(str(path).encode()).hexdigest() + '.cache')
            destination.parent.mkdir(exist_ok=True)
            return original_save(prefix, destination, data, version)

        with patch.object(dataset_module, 'save_dataset_cache_file', redirected_cache):
            loader = trainer.get_dataloader(trainer.data['train'], batch_size=8, rank=-1, mode='train')
            # The historical pressure replay created this loader before iterating.
            # Preserve constructor/RNG order; no validation iteration or inference.
            val_loader = trainer.get_dataloader(trainer.data['val'], batch_size=8, rank=-1, mode='val')
        require(len(loader.dataset) == 11310 and len(loader) == 1414 and len(val_loader.dataset) == 548,
                'loader population mismatch')
        del val_loader
        require(dict(factory.counts) == parent['source_factory_counts'], 'source population mismatch')
        trainer.train_loader = loader
        write_json(output / 'source_factory.json', dict(reads=factory.reads, bindings=factory.bindings,
                                                       counts=dict(factory.counts)))
        pressure = replay_pressure(loader, trainer, parent, amp, beat, output)
        beat('load_ema', phase=None, batch_index=None)
        require(sha(EPOCH9) == EPOCH9_SHA, 'epoch-nine checkpoint changed')
        checkpoint = torch.load(EPOCH9, map_location='cpu', weights_only=False)
        ema = verify_epoch_checkpoint(checkpoint)
        require(ema.names == model.names, 'EMA class names differ')
        ema_state = cpu_tree(ema.float().state_dict())
        report['checkpoint'] = dict(path=str(EPOCH9), sha256=EPOCH9_SHA, epoch=8,
                                    model='saved_half_EMA_cast_float_not_online_student')
        del checkpoint, ema
        states = dict(initial=initial, epoch9=ema_state)
        for name, state in states.items():
            save_tensor_file(output / f'{name}_state.pt', dict(state_dict=state, model_yaml=model.yaml))
        for ordinal, (state_name, case) in enumerate(fixed_cases(), 1):
            beat('case_setup', case=case, weights=state_name, ordinal=ordinal, arm=None)
            verify_sources(manifest)
            model.load_state_dict(states[state_name], strict=True)
            model.train()
            model.zero_grad(set_to_none=True)
            raw = cpu_tree(pressure) if case == 'pressure' else torch.load(
                PREFIX / case / 'batch.pt', map_location='cpu', weights_only=False)
            directory = output / f'{ordinal:02d}_{state_name}_{case}'
            directory.mkdir()
            before = model_hashes(model)
            buffers = {n: v.detach().clone() for n, v in model.named_buffers()}
            batch = trainer.preprocess_batch(raw)
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
            results = {}
            try:
                beat('forward')
                torch.cuda.synchronize()
                tick = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.float16):
                    preds = model(batch['img'])
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter() - tick
                report['explicit_forwards'] += 1
                beat('save_request')
                save_tensor_file(directory / 'request.pt', dict(cpu_batch=cpu_tree(batch),
                    predictions=cpu_tree(preds), rng=rng, before=before, state=state_name,
                    source_sha256=manifest['source_sha256'], diagnostic_scale=SCALE))
                for arm in ARMS:
                    random.setstate(rng[0]); np.random.set_state(rng[1])
                    torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
                    results[arm] = run_arm(arm, model, preds, batch, ordinal, directory, beat, manifest)
                beat('compare')
                comparisons = shared_comparison(results)
                if case == 'pressure':
                    for arm in ('new_candidate', 'new_control'):
                        require(results[arm]['report']['point_search']['by_dimension'].get('3', {}).get('calls', 0) > 0,
                                'pressure case never consumed 3U point geometry')
                    require(results['new_candidate']['report']['supervision_by_dimension'].get('3', 0) > 0,
                            'pressure candidate never consumed 3U supervision')
                    require(not comparisons['gradient']['equal'], 'pressure candidate/control gradients identical')
                row = dict(case=case, weights=state_name, ordinal=ordinal, forward_seconds=forward_seconds,
                           comparisons=comparisons, arms={a: r['report'] for a, r in results.items()})
            finally:
                # Restore before I/O: heartbeat failures must not skip restoration.
                with torch.no_grad():
                    for n, v in model.named_buffers():
                        v.copy_(buffers[n])
                torch.cuda.synchronize()
                after = model_hashes(model)
                write_json(directory / 'state.json', dict(before=before, after=after, restored=after == before))
            require(before == after and all(p.grad is None for p in model.parameters()), 'model state drift')
            row['state_restored'] = True
            write_json(directory / 'result.json', row)
            report['cases'].append(dict(case=case, weights=state_name, ordinal=ordinal, result=str(directory / 'result.json')))
            report['completed_cases'] += 1
            del preds, results, batch, raw
        beat('final_verification', arm=None)
        model.load_state_dict(initial, strict=True)
        require(model_hashes(model) == initial_hash, 'final state mismatch')
        require(trainer.optimizer is None and trainer.ema is None, 'optimizer or training EMA constructed')
        require(report['completed_cases'] == 34, 'fixed 34 cases incomplete')
        verify_sources(manifest)
        for path, digest in factory.reads.items():
            require(sha(path) == digest, f'legal source changed: {path}')
        report.update(status='PASSED_PRESSURE_ZERO_TRAINING', all_sources_unchanged=True,
                      all_state_restored=True, original_initial_state_restored=True)
        code = 0
    except BaseException as exc:
        report.update(status='STOPPED_NO_RETRY', error=exception_info(exc), last_stage=beat.last)
        (output / 'worker_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    report['elapsed_seconds'] = time.perf_counter() - start
    write_json(output / 'result.json', report)
    return code


def parent(output):
    output = output.resolve()
    require(output.parent == OUT_ROOT and re.fullmatch('audit_[0-9]{2}', output.name), 'fresh audit_NN under fixed root required')
    output.mkdir(parents=True, exist_ok=False)
    token = secrets.token_hex(32)
    hashes = base.snapshot_sources(output)
    inputs = [*BINDING_SOURCES, GEOMETRY, AMP, EPOCH9, INITIAL_WEIGHTS,
              PREFIX / 'terminal.json', PREFIX / 'manifest.json']
    terminal = read_json(PREFIX / 'terminal.json')
    require(terminal['status'] == 'PASSED' and terminal['complete_32_passed'], 'real_03 did not pass')
    require(sha(GEOMETRY) == '39c03aa3a4243a0f3169abe174c4701d964f905df749c208148fc0bccee7d259', 'geometry receipt changed')
    require(sha(EPOCH9) == EPOCH9_SHA, 'epoch-nine checkpoint SHA mismatch')
    frozen = read_json(base.FREEZE)
    require(sha(AMP) == frozen['engineering']['sha256'], 'AMP receipt changed')
    require(sha(INITIAL_WEIGHTS) == '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5', 'initial weights changed')
    for phase in ('open', 'closed'):
        for i in range(16):
            directory = PREFIX / f'{phase}_{i:02d}'
            receipt = read_json(directory / 'result.json')
            require(receipt['status'] == 'COMPLETED' and receipt['state_restored'], 'incomplete prefix batch')
            require(sha(directory / 'batch.pt') == receipt['raw_batch']['sha256'], 'prefix raw batch changed')
            inputs.extend([directory / 'result.json', directory / 'batch.pt'])
    hashes.update({str(p): sha(p) for p in inputs})
    for p in BINDING_SOURCES:
        dest = output / 'source_snapshot/repo' / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    manifest = dict(parent_pid=os.getpid(), token_sha256=hashlib.sha256(token.encode()).hexdigest(),
                    source_sha256=hashes, cases=fixed_cases(), scale=SCALE, timeout=TIMEOUT,
                    git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write_json(output / 'manifest.json', manifest)
    proc = None
    terminal = dict(status='STARTING', training_started=False)
    try:
        with (output / 'worker.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker', str(output)],
                cwd=ROOT, env={**os.environ, TOKEN: token, 'PYTHONDONTWRITEBYTECODE': '1'}, stdout=log, stderr=subprocess.STDOUT)
            write_json(output / 'process.json', dict(parent_pid=os.getpid(), worker_pid=proc.pid,
                command=proc.args, utc=datetime.now(timezone.utc).isoformat()))
            sequence, changed, last = -1, time.monotonic(), None
            while proc.poll() is None:
                item = latest_heartbeat(output)
                if item is not None:
                    require(item['pid'] == proc.pid, 'wrong worker heartbeat')
                    if item['sequence'] != sequence:
                        sequence, last, changed = item['sequence'], item, time.monotonic()
                if time.monotonic() - changed >= TIMEOUT:
                    proc.kill()
                    proc.wait(timeout=10)
                    terminal = dict(status='TIMEOUT', last_heartbeat=last, training_started=False,
                                    owned_worker_pid=proc.pid, returncode=proc.returncode)
                    break
                try:
                    proc.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
            if terminal['status'] != 'TIMEOUT':
                result = read_json(output / 'result.json') if (output / 'result.json').exists() else {}
                verify_sources(manifest)
                passed = proc.returncode == 0 and result.get('status') == 'PASSED_PRESSURE_ZERO_TRAINING'
                terminal = dict(status='PASSED' if passed else 'WORKER_FAILED', worker_result=result,
                    returncode=proc.returncode, owned_worker_pid=proc.pid, source_verified_after=True, training_started=False)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        terminal = dict(status='PARENT_FAILED', error=exception_info(exc), training_started=False)
    write_json(output / 'terminal.json', terminal)
    print(json.dumps(dict(output=str(output), status=terminal['status'], training_started=False)), flush=True)
    return 0 if terminal['status'] == 'PASSED' else 2


# run_arm is a local, explicitly scaled derivative of the frozen real_03 helper.
# It is defined below before CLI dispatch; the original helper stays unchanged.


def run_arm(arm, model, preds, batch, ordinal, directory, beat, manifest):
    from regionssod.training import bounded_retention_objective_v6 as objective
    from regionssod.training.bounded_retention_continuous_numerics import optimize_point as old
    from regionssod.training.bounded_retention_point_geometry_cache import optimize_point as new
    from regionssod.training.p1 import P1DetectionLoss

    require(arm in ARMS, "only original P1 and new paired arms allowed")
    beat("prepare_loss", arm=arm)
    verify_sources(manifest)
    cleared = clear_numerical_caches()
    criterion = P1DetectionLoss(model) if arm == "p1" else objective.ContinuousNativeELoss(model)
    if arm != "p1":
        criterion.mode = arm.split("_")[1]
        criterion.posterior_call_number = ordinal - 1
    assigned = {}

    def capture_assignment(module, inputs, output):
        # Detached GPU references only; CPU copies and hashing are outside loss timing.
        assigned.update({name: value.detach() for name, value in zip(
            ("labels", "boxes", "scores", "foreground", "gt_index"), output, strict=True)})

    point_stats = {"calls": 0, "seconds": 0., "by_dimension": {}, "cells_sum": 0}
    solver = new if arm.startswith("new_") else old

    def observed_point(domain, score, **kwargs):
        require(kwargs.get("max_cells", 20000) == 20000 and kwargs.get("tolerance", 1e-5) == 1e-5,
                "point resource/precision contract changed")
        start = time.perf_counter()
        try:
            result = solver(domain, score, **kwargs)
            require(result.diagnostics.get("qualified") is True and
                    0 <= result.gap <= Fraction.from_float(1e-5), "unqualified point result")
            require(result.diagnostics.get("cells", 0) <= 20000, "point cell budget exceeded")
            # Independent original point check, including for any proposed mean witness.
            require(domain.point(result.point) is not None, "returned point fails original oracle")
            point_stats["cells_sum"] += result.diagnostics.get("cells", 0)
            return result
        except Exception as exc:
            try:
                # DFLScore defines both logits and logits_U. Guard collection/IO
                # nevertheless: a failed diagnostic must never replace the solver error.
                save_tensor_file(directory / f"{arm}.point_failure.pt", {
                    "record_physical": domain.physical, "nominal_box": domain.box,
                    "canvas": domain.canvas, "active_sides": domain.active_sides,
                    "affine_bounds": domain.affine_bounds, "clip_bounds": domain.clip_bounds,
                    "logits_U": cpu_tree(score.logits), "offsets": score.offsets,
                    "signs": score.signs, "stride": score.stride,
                    "max_cells": 20000, "tolerance": 1e-5,
                    "error": repr(exc), "diagnostics": getattr(exc, "diagnostics", None)})
            except Exception as diagnostic_error:
                exc.add_note(f"point_failure artifact could not be saved: {diagnostic_error!r}; "
                             "the complete pre-loss request.pt remains the replay source")
            raise
        finally:
            elapsed = time.perf_counter() - start
            point_stats["calls"] += 1
            point_stats["seconds"] += elapsed
            group = point_stats["by_dimension"].setdefault(str(domain.dimension), {"calls": 0, "seconds": 0.})
            group["calls"] += 1
            group["seconds"] += elapsed

    arm_report = {"arm": arm, "cleared_numerical_caches": cleared,
                  "posterior_call_number": ordinal, "gradient_scale": SCALE,
                  "gradient_scope": "both heads and every requires_grad model parameter; no .grad accumulation"}
    write_json(directory / f"{arm}.started.json", arm_report)
    with ExitStack() as stack:
        handle = criterion.assigner.register_forward_hook(capture_assignment)
        stack.callback(handle.remove)
        if arm != "p1":
            stack.enter_context(patch.object(objective, "optimize_point", observed_point))
        beat("loss", arm=arm)
        torch.cuda.synchronize(model.device if hasattr(model, "device") else None)
        torch.cuda.reset_peak_memory_stats()
        start, cpu_start = time.perf_counter(), time.process_time()
        with torch.autocast("cuda", dtype=torch.float16):
            full_loss, loss_items = criterion(preds, batch)
        torch.cuda.synchronize()
        arm_report.update(loss_seconds=time.perf_counter() - start,
                          loss_cpu_seconds=time.process_time() - cpu_start,
                          loss_cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                          loss_cuda_peak_reserved=torch.cuda.max_memory_reserved())
    beat("save_loss", arm=arm)
    require(assigned, "assigner hook did not observe the full assignment")
    assignment = cpu_tree(assigned)
    geometry, certificates = geometry_snapshot(criterion)
    loss_cpu = cpu_tree(full_loss)
    arm_report.update(full_loss=loss_cpu.tolist(), loss_items=cpu_tree(loss_items),
                      full_loss_stats=finite_stats(loss_cpu), point_search=point_stats,
                      criterion_diagnostics=cpu_tree(getattr(criterion, "diagnostics", {})),
                      supervision_by_dimension=supervision_profile(criterion))
    arm_report["observation_artifact"] = save_tensor_file(directory / f"{arm}.loss.pt", {
        "full_loss": loss_cpu, "loss_items": cpu_tree(loss_items),
        "assignment": assignment, "geometry": geometry, "certificates": certificates,
        "criterion_diagnostics": arm_report["criterion_diagnostics"]})
    write_json(directory / f"{arm}.json", arm_report)
    require(arm_report["full_loss_stats"]["finite"], f"nonfinite {arm} unscaled full loss")
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    require(named and all(p.grad is None for p in model.parameters()), "unexpected model gradient storage")
    inputs = (preds["boxes"], preds["scores"], *(p for _, p in named))
    beat("backward", arm=arm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, cpu_start = time.perf_counter(), time.process_time()
    gradients = torch.autograd.grad(full_loss.sum() * SCALE, inputs, retain_graph=True, allow_unused=True)
    torch.cuda.synchronize()
    arm_report.update(backward_seconds=time.perf_counter() - start,
                      backward_cpu_seconds=time.process_time() - cpu_start,
                      backward_cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                      backward_cuda_peak_reserved=torch.cuda.max_memory_reserved())
    beat("save_gradients", arm=arm)
    names = ["head/boxes", "head/scores", *(f"parameter/{name}" for name, _ in named)]
    grads = dict(zip(names, [None if g is None else g.detach().cpu().double() / SCALE
                            for g in gradients], strict=True))
    stats = {name: finite_stats(value) for name, value in grads.items()}
    arm_report.update(gradients=stats,
                      unused_parameters=[k for k, v in stats.items() if k.startswith("parameter/") and v["status"] == "UNUSED"],
                      gradient_finite=all(v["finite"] is True for v in stats.values() if v["status"] == "OBSERVED"),
                      model_grad_storage_untouched=all(p.grad is None for p in model.parameters()),
                      status="COMPLETED")
    write_json(directory / f"{arm}.json", arm_report)
    require(grads["head/boxes"] is not None and grads["head/scores"] is not None, "head gradient missing")
    require(any(v is not None for k, v in grads.items() if k.startswith("parameter/")), "no model parameter gradients")
    require(not arm_report['unused_parameters'], 'some trainable model parameters have no gradient')
    require(arm_report["gradient_finite"] and arm_report["model_grad_storage_untouched"],
            f"nonfinite or accumulated {arm} gradients at fixed scale1024; no retry")
    verify_sources(manifest)
    return {"report": arm_report, "loss": loss_cpu, "assignment": assignment,
            "geometry": geometry, "certificates": certificates, "gradients": grads}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(worker(args.worker.resolve()) if args.worker else parent(args.output))
