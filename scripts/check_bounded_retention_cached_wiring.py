"""Zero-training production factory/model.loss replay of certified pressure cases.

No optimizer, training, or validation inference. Only the owning diagnostic child
can be terminated by the unchanged 120-second stage guard. Existing evidence is
read-only; new artifacts must be under the independent cache audit root.
"""
from __future__ import annotations
import argparse
import gc
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import subprocess
import sys
import time
import traceback
from types import FunctionType
from unittest.mock import patch

import check_bounded_retention_point_real as base

ROOT = base.ROOT
OUT_ROOT = ROOT / 'outputs/bounded_retention_cache_20261106'
PRESSURE = OUT_ROOT / 'audit_01'
QUALIFICATION = OUT_ROOT / 'qualification_02/report.json'
QUALIFICATION_SHA = '21dc0f55a66cbf2dadf2d1e078def951ceeeb0477452a9760f427d7142188a88'
PARENT_FREEZE = ROOT / 'outputs/bounded_retention_continuous_freeze_20261105_v2/freeze.json'
PARENT_FREEZE_SHA = 'a04ec4f117472fc2e11f988a61534d3de5898c9c9bff5d7f7a592928a6a57901'
CONTRACT = ROOT / 'docs/paper/candidates/bounded_retention/cached_training_contract.md'
ARMS = ('reference', 'candidate', 'control')
TIMEOUT = 120
TOKEN = 'REGIONSSOD_CACHED_WIRING_DIAGNOSTIC_TOKEN'
SCALE = 1024.
require, sha, write_json = base.require, base.sha, base.write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def merge_hashes(*maps):
    result = {}
    for mapping in maps:
        for name, digest in mapping.items():
            name = str(Path(name).resolve())
            require(name not in result or result[name] == digest, f'conflicting input SHA: {name}')
            result[name] = digest
    return result


def elapsed_ok(seconds):
    require(math.isfinite(seconds) and 0 <= seconds <= TIMEOUT, 'invalid/exceeded 120-second stage')


class StrictHeartbeat(base.Heartbeat):
    """A returned stage cannot evade its deadline by publishing a new event."""
    def __init__(self, output):
        super().__init__(output)
        self.started = None

    def __call__(self, stage, **context):
        now = time.perf_counter()
        if self.started is not None:
            elapsed = now - self.started
            elapsed_ok(elapsed)
            context['previous_stage_seconds'] = elapsed
        self.started = now
        return super().__call__(stage, **context)


def strict_tree_equal(actual, expected):
    """Byte identity for tensors; frozen metadata equality for other leaves."""
    import torch
    from audit_bounded_retention_cached_pressure_v2 import saved_tree_equal
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, torch.Tensor):
        return actual.dtype == expected.dtype and actual.shape == expected.shape and torch.equal(
            actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(strict_tree_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(strict_tree_equal(a, b) for a, b in zip(actual, expected))
    return saved_tree_equal(actual, expected)


def diagnostic_factory(launcher, output):
    """Original factory code; isolate only diagnostic output and saving flags."""
    def args(config):
        return launcher.expected_args(config) | {
            'project': str(output), 'name': config['name'],
            'save_dir': str(output / config['name']), 'save': False}
    namespace = dict(launcher.make_trainer.__globals__)
    namespace['expected_args'] = args
    original = launcher.make_trainer
    function = FunctionType(original.__code__, namespace, original.__name__,
                            original.__defaults__, original.__closure__)
    function.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    return function


def worker(output):
    import numpy as np
    import torch
    import yaml
    sys.path.insert(0, str(ROOT / 'src'))
    import run_bounded_retention_cached as launcher
    from regionssod.training.p1 import P1Trainer, P1DetectionLoss
    from regionssod.training.bounded_retention_training_cached import CachedNativeETrainer
    from regionssod.training.bounded_retention_objective_cached import CachedNativeELoss, CachedNativeEAssigner
    from regionssod.training.bounded_retention_point_geometry_cache import cache_info, clear_geometry_cache
    from regionssod.training.bounded_retention_source import BoundSourceFactory
    from ultralytics.data import dataset as dataset_module
    from ultralytics.utils.torch_utils import ModelEMA
    base.torch, base.np = torch, np
    beat = StrictHeartbeat(output)
    manifest = read(output / 'manifest.json')
    token = os.environ.pop(TOKEN, '')
    require(os.getppid() == manifest['parent_pid'] and token and
            hashlib.sha256(token.encode()).hexdigest() == manifest['token_sha256'], 'invalid worker token')
    report = {'status': 'RUNNING', 'training_started': False, 'optimizer_steps': 0,
              'validation_inference_calls': 0, 'case_count': 0, 'cases': [],
              'input_sha256': manifest['source_sha256'], 'diagnostic_scale': SCALE,
              'factory_override': 'same code object; project/name/save_dir/save only',
              'source_factories': {}, 'ema_checks': {}}
    try:
        beat('verify_inputs')
        base.verify_sources(manifest)
        qualification = read(QUALIFICATION)
        require(qualification['accepted'] and qualification['checks']['checked_arms'] == 102,
                'pressure qualification absent')
        require(torch.cuda.is_available(), 'CUDA required')
        report['runtime'] = {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
        native_factory = diagnostic_factory(launcher, output / 'trainers')
        initial = torch.load(PRESSURE / 'initial_state.pt', map_location='cpu', weights_only=False)['state_dict']
        epoch9 = torch.load(PRESSURE / 'epoch9_state.pt', map_location='cpu', weights_only=False)['state_dict']
        states = {'initial': initial, 'epoch9': epoch9}
        for arm in ARMS:
            beat('trainer_setup', arm=arm, weights=None)
            config = yaml.safe_load((ROOT / f'configs/candidates/bounded_retention_cached/{arm}30.yaml').read_text())
            factory = None if arm == 'reference' else BoundSourceFactory()
            def source(**kwargs):
                result = factory(**kwargs)
                if factory.counts['objects'] % 5000 == 0:
                    beat('source_binding', objects=factory.counts['objects'])
                return result
            trainer = native_factory(arm, config, source=source if factory else None)
            require(type(trainer) is (P1Trainer if arm == 'reference' else CachedNativeETrainer),
                    'production factory returned wrong Trainer')
            trainer.setup_model()
            trainer.model = trainer.model.to(trainer.device)
            trainer.set_model_attributes()
            model = trainer.model
            require(trainer.device.type == 'cuda' and trainer.optimizer is None and trainer.ema is None,
                    'wrong device or optimizer/training EMA created')
            require(base.model_hashes(model) == qualification['checks']['state_hashes']['initial'],
                    'native factory initialization differs from certified model')
            require(not hasattr(model, 'criterion'), 'criterion created before native lazy loss')
            ema = ModelEMA(model)
            require(not hasattr(ema.ema, 'criterion'), 'EMA captured criterion')
            beat('loader_setup')
            original_save = dataset_module.save_dataset_cache_file
            def redirected_cache(prefix, path, data, version):
                destination = output / 'dataset_cache' / (hashlib.sha256(str(path).encode()).hexdigest() + '.cache')
                destination.parent.mkdir(exist_ok=True)
                return original_save(prefix, destination, data, version)
            with patch.object(dataset_module, 'save_dataset_cache_file', redirected_cache):
                loader = trainer.get_dataloader(trainer.data['train'], batch_size=8, rank=-1, mode='train')
                val_loader = trainer.get_dataloader(trainer.data['val'], batch_size=8, rank=-1, mode='val')
            require(len(loader.dataset) == 11310 and len(loader) == 1414 and len(val_loader.dataset) == 548,
                    'production dataloader population changed')
            if factory is not None:
                expected_factory = read(PRESSURE / 'source_factory.json')
                require(dict(factory.counts) == expected_factory['counts'] and
                        factory.reads == expected_factory['reads'] and
                        base.json_value(factory.bindings) == expected_factory['bindings'],
                        'production legal source bindings differ')
                report['source_factories'][arm] = dict(counts=dict(factory.counts), reads=factory.reads,
                                                     bindings=factory.bindings)
                report['input_sha256'] = merge_hashes(report['input_sha256'], factory.reads)
            del loader, val_loader
            old_arm = 'p1' if arm == 'reference' else f'new_{arm}'
            for ordinal, state_name in enumerate(('initial', 'epoch9'), 1):
                beat('case_setup', arm=arm, weights=state_name)
                directory = output / f'{arm}_{state_name}'
                directory.mkdir()
                original = PRESSURE / f'{ordinal:02d}_{state_name}_pressure'
                request = torch.load(original / 'request.pt', map_location='cpu', weights_only=False)
                expected = torch.load(original / f'{old_arm}.loss.pt', map_location='cpu', weights_only=False)
                receipt = read(original / f'{old_arm}.json')
                model.load_state_dict(states[state_name], strict=True)
                model.train()
                model.zero_grad(set_to_none=True)
                if hasattr(model, 'criterion'):
                    del model.criterion
                before = base.model_hashes(model)
                buffers = {n: v.detach().clone() for n, v in model.named_buffers()}
                raw = torch.load(PRESSURE / 'pressure_batch.pt', map_location='cpu', weights_only=False)
                batch = trainer.preprocess_batch(raw)
                require(strict_tree_equal(base.cpu_tree(batch), request['cpu_batch']), 'production preprocessing differs')
                rng = request['rng']
                random.setstate(rng[0]); np.random.set_state(rng[1])
                torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
                try:
                    beat('forward')
                    torch.cuda.synchronize(); start = time.perf_counter()
                    with torch.autocast('cuda', dtype=torch.float16):
                        preds = model(batch['img'])
                    torch.cuda.synchronize(); forward_seconds = time.perf_counter() - start
                    elapsed_ok(forward_seconds)
                    require(strict_tree_equal(base.cpu_tree(preds), request['predictions']), 'production predictions differ')
                    # This is exactly the native lazy criterion factory. Explicit
                    # construction permits a read-only TAL hook before model.loss.
                    model.criterion = model.init_criterion()
                    criterion = model.criterion
                    require(type(criterion) is (P1DetectionLoss if arm == 'reference' else CachedNativeELoss),
                            'production criterion factory differs')
                    if arm != 'reference':
                        require(type(criterion.assigner) is CachedNativeEAssigner and criterion.mode == arm,
                                'cached assigner or mode not connected')
                        criterion.posterior_call_number = ordinal - 1
                    base.clear_numerical_caches()
                    assigned = {}
                    def capture(module, inputs, result):
                        assigned.update({name: value.detach() for name, value in zip(
                            ('labels', 'boxes', 'scores', 'foreground', 'gt_index'), result, strict=True)})
                    hook = criterion.assigner.register_forward_hook(capture)
                    beat('loss')
                    torch.cuda.synchronize(); start = time.perf_counter()
                    try:
                        with torch.autocast('cuda', dtype=torch.float16):
                            loss, items = model.loss(batch, preds)
                        torch.cuda.synchronize(); loss_seconds = time.perf_counter() - start
                    finally:
                        hook.remove()
                    elapsed_ok(loss_seconds)
                    geometry, certificates = base.geometry_snapshot(criterion)
                    assignment = base.cpu_tree(assigned)
                    require(strict_tree_equal(base.cpu_tree(loss), expected['full_loss']) and
                            strict_tree_equal(base.cpu_tree(items), expected['loss_items']), 'production loss differs')
                    require(strict_tree_equal(assignment, expected['assignment']) and
                            strict_tree_equal(geometry, expected['geometry']) and
                            strict_tree_equal(certificates, expected['certificates']), 'production TAL/geometry/certificate differs')
                    cache = [cache_info(s['domain']) for s in getattr(criterion, 'supports', {}).values()]
                    if arm != 'reference':
                        require(sum(c['hits'] for c in cache) > 0 and criterion.posterior_call_number == ordinal,
                                'production cache not used or posterior counter changed')
                    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                    names = ['head/boxes', 'head/scores', *(f'parameter/{n}' for n, _ in named)]
                    beat('backward')
                    torch.cuda.synchronize(); start = time.perf_counter()
                    gradients = torch.autograd.grad(loss.sum() * SCALE,
                        (preds['boxes'], preds['scores'], *(p for _, p in named)), allow_unused=True)
                    torch.cuda.synchronize(); backward_seconds = time.perf_counter() - start
                    elapsed_ok(backward_seconds)
                    stats = {n: base.finite_stats(None if g is None else g.detach().cpu().double() / SCALE)
                             for n, g in zip(names, gradients, strict=True)}
                    require(stats == receipt['gradients'], 'production full-model/head gradients differ')
                    require(all(x['finite'] is True for x in stats.values()) and
                            all(p.grad is None for p in model.parameters()), 'nonfinite/unused/accumulated gradients')
                    row = dict(arm=arm, weights=state_name, forward_seconds=forward_seconds,
                        loss_seconds=loss_seconds, backward_seconds=backward_seconds, gradients=stats,
                        full_loss=base.cpu_tree(loss), cache=cache, predictions_identical=True,
                        loss_identical=True, assignment_identical=True, geometry_identical=True,
                        certificates_identical=True, gradients_identical=True)
                    # Validation loss receives supplied training predictions only:
                    # no validation image forward or label is consumed here.
                    beat('validation_loss_dispatch')
                    model.eval()
                    plain = {k: v for k, v in batch.items() if k not in ('obs', 'retention_observations')}
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                        actual_val = model.loss(plain, preds)
                        expected_val = P1DetectionLoss(model)(preds, plain)
                    require(strict_tree_equal(base.cpu_tree(actual_val), base.cpu_tree(expected_val)),
                            'standard validation loss does not recover original P1')
                    row['validation_loss_equals_p1'] = True
                    beat('ema_serialization')
                    ema.update_attr(model, include=('yaml', 'nc', 'args', 'names', 'stride', 'class_weights'))
                    require(not hasattr(ema.ema, 'criterion'), 'criterion/cache leaked into EMA')
                    stream = io.BytesIO()
                    torch.save(ema.ema, stream)
                    stream.seek(0)
                    restored_ema = torch.load(stream, map_location='cpu', weights_only=False)
                    require(not hasattr(restored_ema, 'criterion'), 'serialized EMA contains criterion')
                    report['ema_checks'][f'{arm}_{state_name}'] = True
                    del restored_ema, stream
                finally:
                    with torch.no_grad():
                        for name, value in model.named_buffers():
                            value.copy_(buffers[name])
                    after = base.model_hashes(model)
                    write_json(directory / 'state.json', dict(before=before, after=after, restored=before == after))
                require(before == after, 'production model/BN state drift')
                row['state_restored'] = True
                write_json(directory / 'result.json', row)
                report['cases'].append(row)
                report['case_count'] += 1
                beat('release_case_resources')
                for support in getattr(criterion, 'supports', {}).values():
                    clear_geometry_cache(support['domain'])
                support = None
                if hasattr(criterion, 'supports'):
                    criterion.supports.clear()
                del model.criterion
                del preds, gradients, batch, raw, request, expected, loss, items, criterion
                del plain, named, assigned, actual_val, expected_val, capture, hook
                gc.collect()
            model.load_state_dict(initial, strict=True)
            require(base.model_hashes(model) == qualification['checks']['state_hashes']['initial'], 'final state drift')
            require(trainer.optimizer is None and trainer.ema is None, 'training optimizer/EMA constructed')
            del trainer, model, ema
        beat('final_verification')
        base.verify_sources({'source_sha256': report['input_sha256']})
        require(report['case_count'] == 6, 'six fixed production cases incomplete')
        report.update(status='PASSED_CACHED_PRODUCTION_WIRING', all_sources_unchanged=True, all_state_restored=True)
        code = 0
    except BaseException as exc:
        report.update(status='STOPPED_NO_RETRY', error=base.exception_info(exc))
        (output / 'worker_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        code = 2
    try:
        beat('save_final_receipt')
        write_json(output / 'result.json', report)
        beat('done')
    except BaseException as exc:
        report.update(status='STOPPED_NO_RETRY', error=base.exception_info(exc))
        write_json(output / 'result.json', report)
        code = 2
    return code


def parent(output):
    output = output.resolve()
    require(output.parent == OUT_ROOT and re.fullmatch(r'wiring_\d{2}', output.name), 'fresh wiring_NN required')
    require(sha(QUALIFICATION) == QUALIFICATION_SHA, 'fixed pressure qualification changed')
    qualification = read(QUALIFICATION)
    require(qualification['accepted'] is True and qualification['checks']['checked_cases'] == 34,
            'pressure qualification not complete')
    require(sha(PARENT_FREEZE) == PARENT_FREEZE_SHA, 'fixed legal-population parent freeze changed')
    import run_bounded_retention_cached as launcher
    import yaml
    original_inputs = launcher.hashes(read(PARENT_FREEZE)['input_sha256'])
    config = yaml.safe_load((ROOT / 'configs/candidates/bounded_retention_cached/reference30.yaml').read_text())
    legal = launcher.legal_population(config, original_inputs)
    require(all(str(p) in original_inputs for p in legal), 'parent does not bind complete legal loader inputs')
    legal_hashes = {str(p): original_inputs[str(p)] for p in legal}
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT / 'scripts/run_bounded_retention_cached.py', CONTRACT,
             ROOT / 'scripts/audit_bounded_retention_cached_pressure_v2.py', QUALIFICATION, PARENT_FREEZE,
             *sorted((ROOT / 'configs/candidates/bounded_retention_cached').glob('*.yaml'))]
    hashes = merge_hashes(base.snapshot_sources(output), qualification['evidence_sha256'], legal_hashes,
                         {str(p.resolve()): sha(p) for p in paths})
    for path in paths:
        if path.is_relative_to(ROOT) and 'outputs' not in path.relative_to(ROOT).parts:
            destination = output / 'source_snapshot/repo' / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    token = secrets.token_hex(32)
    manifest = dict(parent_pid=os.getpid(), token_sha256=hashlib.sha256(token.encode()).hexdigest(),
                    source_sha256=hashes, timeout_seconds=TIMEOUT, training_started=False)
    write_json(output / 'manifest.json', manifest)
    base.verify_sources(manifest)
    proc = None
    terminal = {'status': 'PARENT_FAILED', 'training_started': False}
    try:
        with (output / 'worker.log').open('x', encoding='utf-8') as log:
            command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker', str(output)]
            proc = subprocess.Popen(command, cwd=ROOT, env=os.environ | {TOKEN: token, 'PYTHONDONTWRITEBYTECODE': '1'},
                                    stdout=log, stderr=subprocess.STDOUT)
            write_json(output / 'process.json', dict(parent_pid=os.getpid(), worker_pid=proc.pid, command=command,
                                                     utc=datetime.now(timezone.utc).isoformat()))
            changed, sequence = time.monotonic(), -1
            while proc.poll() is None:
                event = base.latest_heartbeat(output)
                if event is not None:
                    require(event['pid'] == proc.pid, 'heartbeat PID mismatch')
                    if event['sequence'] != sequence:
                        sequence, changed = event['sequence'], time.monotonic()
                if time.monotonic() - changed >= TIMEOUT:
                    proc.kill(); proc.wait(timeout=10)
                    terminal = dict(status='TIMEOUT', last_heartbeat=event, training_started=False)
                    break
                try:
                    proc.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
            if terminal['status'] != 'TIMEOUT':
                report = read(output / 'result.json') if (output / 'result.json').exists() else {}
                base.verify_sources(manifest)
                final_event = base.latest_heartbeat(output)
                passed = (proc.returncode == 0 and report.get('status') == 'PASSED_CACHED_PRODUCTION_WIRING'
                          and final_event is not None and final_event['stage'] == 'done')
                terminal = dict(status='PASSED' if passed else 'WORKER_FAILED', returncode=proc.returncode,
                    worker_result_sha256=sha(output / 'result.json') if report else None,
                    source_verified_after=True, training_started=False)
    except BaseException as exc:
        if proc is not None and proc.poll() is None:
            proc.kill(); proc.wait(timeout=10)
        terminal = dict(status='PARENT_FAILED', error=base.exception_info(exc), training_started=False)
    write_json(output / 'terminal.json', terminal)
    print(json.dumps(dict(output=str(output), status=terminal['status'], training_started=False)))
    return 0 if terminal['status'] == 'PASSED' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(worker(args.worker.resolve()) if args.worker else parent(args.output))
