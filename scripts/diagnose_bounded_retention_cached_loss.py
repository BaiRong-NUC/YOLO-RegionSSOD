"""Exact geometry cache saved-logit diagnosis, not a training qualification.

Independent derivative of the protected timeout diagnostic; the shared solver
changes only by bounded exact contraction memoization. See the frozen protocol.

Three serial subprocesses: P1 exact calibration, candidate, strongest control.
Each loss stage has one absolute 120s deadline; nested trace never renews it.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import check_bounded_retention_point_real as base

ROOT = base.ROOT
OLD = ROOT / 'outputs/bounded_retention_pressure_20261106/audit_02'
REVIEW = ROOT / 'outputs/bounded_retention_pressure_20261106/terminal_review_01/report.json'
OUT = ROOT / 'outputs/bounded_retention_cache_20261106'
PROTOCOL = ROOT / 'docs/paper/candidates/bounded_retention/point_geometry_cache_protocol.md'
TOKEN = 'REGIONSSOD_CACHE_DIAG_TOKEN'
ARMS = ('p1', 'candidate', 'control')
LIMIT = 120
require, sha, write = base.require, base.sha, base.write_json
REFERENCE = ROOT / 'outputs/bounded_retention_timeout_20261106/trace_04'
REFERENCE_MANIFEST_SHA = '7fa49c8edf3be2116303c5b9b31a51715a0c31d7887b6d67b649610261d76025'
ADDENDUM = ROOT / 'docs/paper/candidates/bounded_retention/point_geometry_cache_audit_addendum.md'
TRACE_PINS = {
    'candidate': 'df0866fdbddc3de9a98e1f5a0e0f9c33387d41ba8f8a9644c494fef8f86b79c1',
    'control': '8158be3ad787649b20798b7f984d8faa9629f24d028e52b99989ea54dc901266',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def point_records(path):
    active, records, seen = {}, [], set()
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'enter':
                require(row['id'] not in seen, 'duplicate event ID')
                seen.add(row['id'])
                if row['name'] == 'point':
                    parent = active.get(row.get('parent'))
                    require(parent and parent['name'] == 'support', 'point has no active support parent')
                    row['support_identity'] = parent['metadata']['identity']
                active[row['id']] = row
            else:
                require(row['id'] in active, 'exit without enter')
                entry = active.pop(row['id'])
                require(entry['name'] == row['name'], 'event name mismatch')
                if row['name'] == 'point':
                    require(row['kind'] == 'exit', 'failed point')
                    records.append((entry['support_identity'], entry['metadata'],
                                    {k: row['result'][k] for k in ('point', 'gap', 'cells')}))
    require(not any(row['name'] == 'point' for row in active.values()), 'unclosed point calls')
    return records


def compare_points(original, current):
    a, b = point_records(original), point_records(current)
    require(len(a) == len(b) == 2590, 'fixed point population changed')
    require(a == b, 'point input or recorded certificate differs from uncached v2')
    return dict(count=2590, equal=True, scope='all metadata and recorded point/gap/cells')


def validate_loss_window(directory):
    start = read(Path(directory) / 'loss_boundary.json')['started']
    end = read(Path(directory) / 'loss_done.json')['finished']
    require(all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
                for t in (start, end)) and 0 <= end-start <= LIMIT,
            'saved loss window missing, invalid or beyond frozen 120s deadline')
    return end-start


def tensor_bytes_equal(a, b):
    import torch
    return (a.dtype == b.dtype and a.shape == b.shape and
            torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                        b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def deadline(started, loss_started, now):
    # Inner trace activity cannot change either start time.
    return now - (started if loss_started is None else loss_started) >= LIMIT


def domain_data(domain):
    # No cached methods or executable objects; Fraction values remain exact strings.
    row = {name: getattr(domain, name) for name in
           ('box','canvas','mask','active_sides','affine_bounds','clip_bounds','bounds')}
    row['source_key'] = domain._key
    return json.dumps(row, sort_keys=True, default=str).encode('utf-8')


def python_stack(thread_id):
    frame = sys._current_frames().get(thread_id)
    if frame is None:
        return []
    return [dict(file=f.filename, function=f.name, line=f.lineno)
            for f in traceback.extract_stack(frame)]


class StackSampler:
    """Best-effort Python snapshots; no native signal or deadline changes."""
    def __init__(self, directory):
        self.directory, self.target = Path(directory), threading.get_ident()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        index = 0
        while not self.stop.wait(20):
            index += 1
            write(self.directory / f'stack_{index:03d}.json',
                  dict(wall=time.monotonic(), cpu=time.process_time(), stack=python_stack(self.target)))

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=1)


class Trace:
    def __init__(self, path):
        self.file = Path(path).open('x', encoding='utf-8', buffering=1)
        self.serial, self.stack, self.io_seconds = 0, [], 0.

    def event(self, **row):
        tick = time.perf_counter()
        row.update(wall=time.monotonic(), cpu=time.process_time(), pid=os.getpid())
        self.file.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        self.file.flush()
        self.io_seconds += time.perf_counter() - tick

    def call(self, name, function, *args, metadata=None, summary=None, **kwargs):
        self.serial += 1
        ident = self.serial
        self.event(kind='enter', id=ident, parent=self.stack[-1] if self.stack else None,
                   name=name, metadata=metadata or {})
        self.stack.append(ident)
        wall, cpu = time.monotonic(), time.process_time()
        try:
            result = function(*args, **kwargs)
            self.event(kind='exit', id=ident, name=name, seconds=time.monotonic()-wall,
                       cpu_seconds=time.process_time()-cpu,
                       result={} if summary is None else summary(result))
            return result
        except BaseException as exc:
            self.event(kind='error', id=ident, name=name, error=repr(exc))
            raise
        finally:
            self.stack.pop()


def metadata_view(torch, args, device):
    # Only the interface read by the unchanged criterion. No detector is built.
    return SimpleNamespace(parameters=lambda: iter([torch.empty(0, device=device)]),
        model=[SimpleNamespace(stride=torch.tensor([8., 16., 32.], device=device), nc=10, reg_max=16)],
        args=SimpleNamespace(**args), p1_options={'soft_weights': True, 'uncertain_ignore': True},
        bounded_retention_mode='candidate', class_weights=None)


def move_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.to('cuda:0')
    if isinstance(value, dict):
        return {k: move_tree(torch, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_tree(torch, v) for v in value)
    return value


def worker(directory, arm):
    directory = Path(directory)
    manifest = read(directory.parent / 'manifest.json')
    require(hashlib.sha256(os.environ.get(TOKEN, '').encode()).hexdigest() == manifest['token_sha256'],
            'unauthenticated diagnostic worker')
    base.verify_sources(manifest)
    trace = Trace(directory / 'trace.jsonl')
    result = dict(arm=arm, training_started=False, detector_forwards=0, backwards=0, optimizer_steps=0)
    code = 2
    try:
        sys.path.insert(0, str(ROOT / 'src'))
        import numpy as np
        import torch
        import yaml
        from regionssod.training import bounded_retention_objective_v6 as objective
        from regionssod.training import bounded_retention_normalizer_v6 as norm
        from regionssod.training import bounded_retention_continuous_sampling as sampling
        from regionssod.training.bounded_retention_point_geometry_cache import optimize_point, cache_info
        from regionssod.training.p1 import P1DetectionLoss
        base.torch, base.np = torch, np
        request = torch.load(OLD / '01_initial_pressure/request.pt', map_location='cpu', weights_only=False)
        args = yaml.safe_load((OLD / 'trainer/args.yaml').read_text(encoding='utf-8'))
        require((args['box'], args['cls'], args['dfl'], args['seed'], args['deterministic']) ==
                (7.5, .5, 1.5, 0, True), 'historical loss configuration differs')
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        view = metadata_view(torch, args, 'cuda:0')
        batch = move_tree(torch, request['cpu_batch'])
        preds = move_tree(torch, request['predictions'])
        require(tuple(preds['boxes'].shape) == (8, 64, 8400) and
                tuple(preds['scores'].shape) == (8, 10, 8400) and
                [tuple(v.shape[-2:]) for v in preds['feats']] == [(80,80),(40,40),(20,20)],
                'unexpected saved prediction schema')
        for name in ('boxes', 'scores'):
            preds[name].requires_grad_(True)
        criterion = P1DetectionLoss(view) if arm == 'p1' else objective.ContinuousNativeELoss(view)
        if arm != 'p1':
            criterion.mode, criterion.posterior_call_number = arm, 0
        assigned = {}
        def capture(module, inputs, output):
            assigned.update(zip(('labels','boxes','scores','foreground','gt_index'),
                                (v.detach() for v in output), strict=True))
        criterion.assigner.register_forward_hook(capture)
        rng = request['rng']
        random.setstate(rng[0]); np.random.set_state(rng[1])
        torch.set_rng_state(rng[2]); torch.cuda.set_rng_state_all(rng[3])
        result['runtime'] = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            cuda=torch.version.cuda, threads=torch.get_num_threads(), interop=torch.get_num_interop_threads())
        result['cleared_caches'] = base.clear_numerical_caches()
        write(directory / 'setup.json', result)
        original_support = objective.continuous_support
        def support(*a, **kw):
            return trace.call('support', original_support, *a, **kw,
                metadata={'identity': [a[0]['source_object_id'], a[0]['occurrence_id']]})
        domains = {}
        def point(domain, score, **kw):
            require(kw.get('max_cells', 20000) == 20000 and kw.get('tolerance', 1e-5) == 1e-5,
                    'point numerical limit changed')
            if id(domain) not in domains:
                data = domain_data(domain)
                digest = hashlib.sha256(data).hexdigest()
                (directory / f'domain_{digest}.json').write_bytes(data)
                domains[id(domain)] = (domain, digest)
            digest = domains[id(domain)][1]
            meta = dict(dimension=domain.dimension, domain_sha256=digest,
                        logits=score.exact_logits, offsets=score.offsets,
                        signs=score.signs, stride=score.stride)
            def solve():
                value = optimize_point(domain, score, **kw)
                require(value.diagnostics.get('qualified') and 0 <= value.gap <= 1e-5,
                        'uncertified point')
                require(domain.point(value.point) is not None, 'point outside original domain')
                return value
            return trace.call('point', solve, metadata=meta, summary=lambda v:
                dict(cells=v.diagnostics.get('cells'), gap=str(v.gap), point=list(map(str,v.point)), cache=cache_info(domain)))
        def wrap(name, function):
            def called(*a, **kw):
                dimension = getattr(a[0], 'dimension', None) if a else None
                return trace.call(name, function, *a, **kw,
                    metadata={'dimension': dimension}, summary=(lambda v: v.diagnostics)
                    if name == 'posterior' else None)
            return called
        with ExitStack() as stack:
            stack.enter_context(patch.object(objective, 'continuous_support', support))
            stack.enter_context(patch.object(objective, 'optimize_point', point))
            for module, function, label in ((norm,'normalizer','normalizer'),
                    (norm,'integrate_interval_union','interval_integral'),
                    (sampling,'sample_posterior','posterior'),
                    (sampling,'_partition','posterior_partition')):
                stack.enter_context(patch.object(module, function, wrap(label, getattr(module,function))))
            stack.enter_context(StackSampler(directory))
            write(directory / 'loss_boundary.json', {'started': time.monotonic(), 'limit': LIMIT})
            trace.call('cuda_pre', torch.cuda.synchronize)
            with torch.autocast('cuda', dtype=torch.float16):
                loss, items = trace.call('criterion', criterion, preds, batch)
            trace.call('cuda_post', torch.cuda.synchronize)
            write(directory / 'loss_done.json', {'finished': time.monotonic()})
            validate_loss_window(directory)
        payload = dict(full_loss=base.cpu_tree(loss), assignment=base.cpu_tree(assigned),
                       geometry=base.geometry_snapshot(criterion)[0],
                       diagnostics=base.cpu_tree(getattr(criterion, 'diagnostics', {})))
        base.save_tensor_file(directory / 'loss.pt', payload)
        if arm == 'p1':
            old = torch.load(OLD / '01_initial_pressure/p1.loss.pt', map_location='cpu', weights_only=False)
            require(tensor_bytes_equal(payload['full_loss'], old['full_loss']) and
                    payload['assignment'].keys() == old['assignment'].keys() and
                    all(tensor_bytes_equal(v, old['assignment'][k]) for k,v in payload['assignment'].items()),
                    'P1 saved-logit calibration differs from original full-model loss/assignment')
            result['exact_historical_p1'] = True
        require(bool(torch.isfinite(loss).all()), 'nonfinite loss')
        result.update(status='COMPLETED_DIAGNOSTIC', full_loss=loss.detach().cpu().tolist(),
                      trace_io_seconds=trace.io_seconds)
        base.verify_sources(manifest)
        code = 0
    except BaseException as exc:
        result.update(status='ERROR', error=repr(exc))
        (directory / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    write(directory / 'result.json', result)
    trace.file.close()
    return code


def parent(output):
    output = output.resolve()
    require(output.parent == OUT and re.fullmatch(r'trace_\d{2}', output.name), 'fresh trace_NN required')
    require(sha(OLD / 'terminal.json') == 'bff7ae8ffb968a682c57e077bc1e4dfb074acfcaebe0437cc2243fb1217012aa',
            'old terminal changed')
    require(sha(REFERENCE / 'manifest.json') == REFERENCE_MANIFEST_SHA, 'reference manifest changed')
    base.verify_sources(read(REFERENCE / 'manifest.json'))
    historical = read(OLD / 'manifest.json')
    base.verify_sources(historical)
    review = read(REVIEW)
    for rel, digest in review['artifacts'].items():
        require(sha(OLD / rel) == digest, f'old artifact changed: {rel}')
    output.mkdir(parents=True, exist_ok=False)
    sources = dict(historical['source_sha256'])
    sources.update({str(OLD / rel): digest for rel,digest in review['artifacts'].items()})
    for p in (Path(__file__).resolve(), PROTOCOL, ADDENDUM, REVIEW, REFERENCE / 'manifest.json', OLD / 'trainer/args.yaml',
              ROOT / 'scripts/diagnose_bounded_retention_loss_timeout.py',
              ROOT / 'src/regionssod/training/bounded_retention_point_geometry_cache.py'):
        sources[str(p)] = sha(p)
    for arm, digest in TRACE_PINS.items():
        path = REFERENCE / arm / 'trace.jsonl'
        require(sha(path) == digest, 'reference trace changed')
        sources[str(path)] = digest
    token = secrets.token_hex(32)
    manifest = dict(source_sha256=sources, token_sha256=hashlib.sha256(token.encode()).hexdigest(),
        arms=ARMS, loss_seconds=LIMIT, git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(output / 'manifest.json', manifest)
    terminals = {}
    for arm in ARMS:
        directory = output / arm
        directory.mkdir()
        with (directory / 'worker.log').open('x', encoding='utf-8') as log:
            command = [sys.executable,'-B','-u',str(Path(__file__).resolve()),'--worker',str(directory),'--arm',arm]
            proc = subprocess.Popen(command, cwd=ROOT, env={**os.environ,TOKEN:token}, stdout=log, stderr=subprocess.STDOUT)
            write(directory / 'process.json', dict(parent_pid=os.getpid(), worker_pid=proc.pid,
                  command=command, start_time=time.time()))
            start, loss_start, killed = time.monotonic(), None, False
            while proc.poll() is None:
                boundary = directory / 'loss_boundary.json'
                if loss_start is None and boundary.exists():
                    loss_start = read(boundary)['started']
                if not (directory / 'loss_done.json').exists() and deadline(start, loss_start, time.monotonic()):
                    print(f'{arm}: frozen deadline reached; terminating owned diagnostic PID {proc.pid}', flush=True)
                    proc.kill(); proc.wait(timeout=10); killed = True
                    break
                # Saving is bounded separately; never extend numerical work.
                if (directory / 'loss_done.json').exists() and time.monotonic()-read(directory/'loss_done.json')['finished'] > LIMIT:
                    proc.kill(); proc.wait(timeout=10); killed = True
                    break
                try:
                    proc.wait(timeout=.2)
                except subprocess.TimeoutExpired:
                    pass
            terminal = dict(status='TIMEOUT' if killed else 'COMPLETED' if proc.returncode==0 else 'ERROR',
                returncode=proc.returncode, worker_pid=proc.pid, training_started=False,
                elapsed_seconds=time.monotonic()-start, loss_started=loss_start)
            if terminal['status'] == 'COMPLETED':
                try:
                    terminal['verified_loss_seconds'] = validate_loss_window(directory)
                except Exception as exc:
                    terminal.update(status='ERROR', error=repr(exc))
            write(directory / 'terminal.json', terminal)
            terminals[arm] = terminal
            if terminal['status']=='ERROR' or (arm=='p1' and terminal['status']!='COMPLETED'):
                break
    base.verify_sources(manifest)
    comparisons, comparison_error = {}, None
    complete = len(terminals) == 3 and all(v['status'] == 'COMPLETED' for v in terminals.values())
    if complete:
        try:
            for arm in TRACE_PINS:
                comparisons[arm] = compare_points(REFERENCE / arm / 'trace.jsonl', output / arm / 'trace.jsonl')
        except Exception as exc:
            comparison_error = repr(exc)
    equivalent = complete and len(comparisons) == 2 and comparison_error is None
    write(output / 'point_equivalence.json', dict(accepted=equivalent, comparisons=comparisons, error=comparison_error))
    write(output / 'terminal.json', dict(status='DIAGNOSTIC_ONLY', arms=terminals,
        all_bound_sources_unchanged=True, training_started=False, pressure_qualified=False,
        point_equivalence_passed=equivalent))
    print(json.dumps({a:v['status'] for a,v in terminals.items()}))
    return 0 if equivalent else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--worker', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    raise SystemExit(worker(args.worker, args.arm) if args.worker else parent(args.output))
