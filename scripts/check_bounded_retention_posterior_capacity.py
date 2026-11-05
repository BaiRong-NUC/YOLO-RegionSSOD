"""Fresh frozen-snapshot V5 numerical receipt; no model, dataset or training."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
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
    parser.add_argument('--name', default='posterior_capacity_01')
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {'.', '..'}:
        raise ValueError('fresh single directory name required')
    out = ROOT/'outputs/bounded_retention_continuous_20261105'/args.name
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out/'snapshot'
    hashes = {}
    for p in [Path(__file__).resolve(), *sorted((ROOT/'src/regionssod').rglob('*.py')),
              *sorted((ROOT/'docs/paper/candidates/bounded_retention').glob('continuous*.md'))]:
        raw, rel = p.read_bytes(), p.relative_to(ROOT)
        target = snapshot/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        hashes[str(rel)] = hashlib.sha256(raw).hexdigest()
    sys.path.insert(0, str(snapshot/'src'))
    import torch
    from regionssod.training.bounded_retention_observation import build_ideal_observation
    from regionssod.training.bounded_retention_continuous_domain import ContinuousDomain
    from regionssod.training.bounded_retention_continuous_numerics import DFLScore, optimize_point
    from regionssod.training.bounded_retention_continuous_sampling import sample_posterior
    from regionssod.training.bounded_retention_continuous_normalizer import normalizer
    start = time.perf_counter()
    report = {'status': 'RUNNING', 'pid': os.getpid(), 'command': [sys.executable, *sys.argv],
        'source_sha256': hashes, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'torch': torch.__version__, 'optimizer_steps': 0, 'model_initialized': False,
        'validation_inference': False, 'training_started': False, 'cases': []}
    (out/'started.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    code = 0
    try:
        for tau in [Q(35, 100), Q.from_float(.35)-Q.from_float(1e-12)]:
            for profile in ['flat', 'nonzero_ramp']:
                case = {'tau': str(tau), 'profile': profile, 'seed': 0, 'K': 256}
                report['cases'].append(case)
                case_start = time.perf_counter()
                physical = build_ideal_observation([Q(1, 2), Q(9, 10), Q(1), Q(1, 5)], retention=tau)
                record = {'physical': physical, 'original_unknown': physical.unknown,
                    'affine_bounds': tuple((Q(x), Q(x)) for x in (64, 64, 64, 64)),
                    'augmentation_clip_window_bounds': tuple((Q(x), Q(x)) for x in (0, 0, 256, 256))}
                domain = ContinuousDomain(record, [64., 115.2, 128., 128.], (256, 256))
                z = torch.zeros(3, 16, dtype=torch.float64) if profile == 'flat' else (
                    torch.tensor([-.7, .5, .8], dtype=torch.float64)[:, None]*torch.arange(16).double()[None])
                z.requires_grad_()
                score = DFLScore(z, (96, 96, 120), (-1, 1, 1), 8)
                case['stage'] = 'global_point'
                point = optimize_point(domain, score)
                case['point'], case['point_diagnostics'] = point.point, point.diagnostics
                assert point.diagnostics['qualified'] and point.gap <= Q(1, 100000)
                case['stage'] = 'posterior'
                result = sample_posterior(domain, score, seed=0, samples=256)
                case['posterior_diagnostics'] = result.diagnostics
                assert result.diagnostics['qualified'] and result.diagnostics['accepted'] == 256
                assert result.diagnostics['acceptance_lower_bound'] >= .125
                assert result.diagnostics['q_mean_error_bound'] <= 1e-10
                z4 = torch.stack((z[0], torch.zeros_like(z[0]), z[1], z[2]))
                logd, diag = normalizer(z4, (64., 115.2, 128., 128.), (96., 120.), 8,
                    (True, False, True, True), (256., 256.), audit=True)
                assert diag['qualified']
                surrogate = logd-(result.q_mean*z).sum()
                grad = torch.autograd.grad(surrogate, z)[0]
                assert torch.isfinite(surrogate) and torch.isfinite(grad).all()
                case.update(stage='passed', normalizer_diagnostics=diag,
                    log_mass_bounds=[result.log_mass_lower, result.log_mass_upper],
                    q_mean=result.q_mean.tolist(), first_gradient=grad.tolist(),
                    seconds=time.perf_counter()-case_start)
                print(json.dumps({k: case[k] for k in ['tau', 'profile', 'stage', 'seconds']}), flush=True)
        report['status'] = 'NUMERICAL_COMPONENT_PASSED_NOT_TRAINING_PERMIT'
    except Exception as exc:
        code = 2
        report['status'] = 'NUMERICAL_COMPONENT_FAILED_NO_TRAINING'
        report['error'] = {'type': type(exc).__name__, 'message': str(exc),
                           'diagnostics': getattr(exc, 'diagnostics', None)}
        (out/'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (out/'result.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        print(json.dumps({k: report.get(k) for k in ['status', 'error', 'elapsed_seconds']}, default=str))
        print(out/'result.json')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
