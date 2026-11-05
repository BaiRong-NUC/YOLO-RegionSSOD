"""Frozen pure-synthetic whole-3U capacity receipt; never a training permit."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction as Q
import hashlib
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='curved_3u_capacity_01')
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {'.', '..'}:
        raise ValueError('fresh single directory name required')
    out = ROOT / 'outputs/bounded_retention_continuous_20261105' / args.name
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / 'snapshot'
    paths = [Path(__file__).resolve(), *sorted((ROOT / 'src/regionssod').rglob('*.py')),
             *sorted((ROOT / 'docs/paper/candidates/bounded_retention').glob('continuous*.md'))]
    hashes = {}
    for p in paths:
        content = p.read_bytes()
        rel = p.relative_to(ROOT)
        target = snapshot / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[str(rel)] = hashlib.sha256(content).hexdigest()
    sys.path.insert(0, str(snapshot / 'src'))
    import numpy as np
    import torch
    from regionssod.training.bounded_retention_observation import build_ideal_observation
    from regionssod.training.bounded_retention_continuous_domain import ContinuousDomain
    from regionssod.training.bounded_retention_continuous_numerics import (
        DFLScore, NumericalCertificationError, integrate_domain, optimize_point)
    started = time.perf_counter()
    report = {'status': 'RUNNING', 'pid': os.getpid(), 'command': [sys.executable, *sys.argv],
        'started_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': hashes,
        'torch': torch.__version__, 'numpy': np.__version__, 'optimizer_steps': 0,
        'model_initialized': False, 'validation_inference': False, 'training_started': False,
        'max_cells': 20000, 'relative_tolerance': 1e-4, 'point_tolerance': 1e-5}
    (out / 'started.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    code = 0
    try:
        physical = build_ideal_observation([Q(1, 2), Q(9, 10), Q(1), Q(1, 5)])
        affine = np.array([64., 64., 64., 64.])
        record = {'physical': physical, 'original_unknown': physical.unknown,
            'affine': affine, 'affine_bounds': np.repeat(affine[:, None], 2, axis=1),
            'augmentation_clip_window_bounds': np.repeat(np.array([0., 0., 256., 256.])[:, None], 2, axis=1)}
        domain = ContinuousDomain(record, [64., 115.2, 128., 128.], (256, 256))
        assert domain.active_sides == (0, 2, 3)
        z = torch.zeros(3, 16, dtype=torch.float64, requires_grad=True)
        score = DFLScore(z, (96, 96, 120), (-1, 1, 1), 8)
        report['bounds'] = domain.bounds
        tau = domain._key[-1]
        with localcontext() as ctx:
            ctx.prec = 80
            a = Decimal(4096) / 5 * Decimal(tau.denominator) / Decimal(tau.numerator)
            h0, h1, h2 = Decimal(64) / 5, a / 128, a / 64
            f = lambda h: 64*a*h.ln() - 6144*h
            g = lambda h: (-a*a/h - 128*a*h.ln() + 4096*h)/2
            report['analytic_volume_reference'] = str(f(h1)-f(h0)+g(h2)-g(h1))
        point_started = time.perf_counter()
        point = optimize_point(domain, score)
        report['point_seconds'] = time.perf_counter()-point_started
        report['point'] = point.point
        report['point_diagnostics'] = point.diagnostics
        assert point.diagnostics['qualified'] and point.gap <= Q(1, 100000)
        report['stage'] = 'whole_C_integral'
        integral_started = time.perf_counter()
        integral = integrate_domain(domain, score, max_cells=20000)
        report['integral_seconds'] = time.perf_counter()-integral_started
        report['integral_diagnostics'] = integral.diagnostics
        report['log_integral'] = float(integral.log_integral.detach())
        report['gradient'] = torch.autograd.grad(integral.log_integral, z)[0].tolist()
        assert integral.diagnostics['qualified'] and integral.relative_residual_bound <= 1e-4
        report['status'] = 'CAPACITY_COMPONENT_PASSED_NOT_TRAINING_PERMIT'
    except Exception as exc:
        code = 2
        report['status'] = 'CAPACITY_NOT_QUALIFIED_NO_TRAINING'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc),
                           'diagnostics': getattr(exc, 'diagnostics', None)}
        if 'integral_started' in locals():
            report['integral_seconds'] = time.perf_counter()-integral_started
        (out / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (out / 'result.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        print(json.dumps({k: report.get(k) for k in ['status', 'point_seconds', 'integral_seconds', 'error']}, default=str))
        print(out / 'result.json')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
