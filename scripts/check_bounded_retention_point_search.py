"""Serial zero-training point benchmark with durable requests and hard budgets."""
from __future__ import annotations
import argparse
from fractions import Fraction as Q
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from regionssod.training.bounded_retention_observation import build_ideal_observation
from regionssod.training.bounded_retention_domain_v6 import NativeEDomain
from regionssod.training.bounded_retention_continuous_numerics import DFLScore, optimize_point as reference
from regionssod.training.bounded_retention_point_search_v2 import optimize_point as revision


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')


def requests():
    generator=torch.Generator().manual_seed(20261106)
    for n in (2,3):
        p=build_ideal_observation([Q(9,10),Q(9,10),Q(1,5),Q(1,5)] if n==2 else
                                  [Q(1,2),Q(3,4),Q(1),Q(1,2)])
        record={'physical':p,'original_unknown':p.unknown,
                'affine_bounds':tuple((Q(v),Q(v)) for v in (10,10,20,20)),
                'augmentation_clip_window_bounds':tuple((Q(v),Q(v)) for v in (0,0,64,64))}
        for kind in ('flat','linear_ridge','random','sharp'):
            logits=(torch.zeros(n,16,dtype=torch.float64) if kind=='flat' else
                    torch.arange(16,dtype=torch.float64).repeat(n,1) if kind=='linear_ridge' else
                    torch.randn(n,16,generator=generator,dtype=torch.float64)*(64 if kind=='sharp' else 1))
            yield f'{n}u_{kind}',dict(record=record,box=[28,28,30,30] if n==2 else [20,25,30,30],
                canvas=(64,64),logits=logits,offsets=(28,28) if n==2 else (25,25,25),
                signs=(1,1) if n==2 else (-1,1,1),stride=1,
                tolerance=1e-5,max_cells=20000,synthetic=True,seed=20261106)


def worker(path,arm):
    started=time.perf_counter()
    try:
        expected=json.loads((path.parent/'environment.json').read_text(encoding='utf-8'))['code_sha256']
        def verify_code():
            if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in expected.items()):
                raise ValueError('benchmark source changed')
        verify_code()
        # Only locally generated benchmark files; not arbitrary external pickles.
        with path.open('rb') as f:
            req=pickle.load(f)
        torch.set_num_threads(1)
        domain=NativeEDomain(req['record'],req['box'],req['canvas'])
        score=DFLScore(req['logits'],req['offsets'],req['signs'],req['stride'])
        solver_started=time.perf_counter()
        result=(reference if arm=='reference' else revision)(domain,score,
            tolerance=req['tolerance'],max_cells=req['max_cells'])
        solver_seconds=time.perf_counter()-solver_started
        assert domain.point(result.point) is not None
        assert result.diagnostics['qualified'] and result.gap<=Q.from_float(req['tolerance'])
        verify_code()
        out=dict(status='CERTIFIED',point=result.point,gap=result.gap,
                 lower=score.value_at(result.point),upper=score.value_at(result.point)+result.gap,
                 diagnostics=result.diagnostics,solver_seconds=solver_seconds,source_verified=True)
    except Exception as exc:
        out=dict(status='ERROR',error=repr(exc),diagnostics=getattr(exc,'diagnostics',{}))
    out.update(worker_body_seconds=time.perf_counter()-started,pid=os.getpid(),threads=1)
    save(path.with_name(f'{path.stem}.{arm}.json'),out)


def main(output):
    output.mkdir(parents=True,exist_ok=False)
    code_paths=[Path(__file__),*sorted((ROOT/'src/regionssod/training').glob('bounded_retention*.py'))]
    for p in code_paths:
        target=output/'source_snapshot'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    save(output/'environment.json',dict(python=sys.version,platform=platform.platform(),
        torch=torch.__version__,cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        solver_device='CPU exact Fraction',threads=1,cache='fresh worker per arm/request',
        budget_seconds=120,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}))
    rows=[]
    for name,request in requests():
        path=output/f'{name}.pkl'
        with path.open('xb') as f:
            pickle.dump(request,f,protocol=5)
        row=dict(name=name,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for arm in ('reference','revision'):
            started=time.perf_counter()
            with path.with_name(f'{name}.{arm}.stdout.log').open('w') as log:
                proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(path),'--arm',arm],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                try:
                    code=proc.wait(timeout=120)
                    dest=path.with_name(f'{name}.{arm}.json')
                    result=json.loads(dest.read_text(encoding='utf-8')) if dest.exists() else dict(status='WORKER_ERROR')
                    result['returncode']=code
                except subprocess.TimeoutExpired:
                    # This PID was created above for THIS bounded diagnostic only.
                    proc.kill()
                    proc.wait()
                    result=dict(status='TIMEOUT',pid=proc.pid,budget_seconds=120)
                    save(path.with_name(f'{name}.{arm}.json'),result)
            result['worker_wall_seconds']=time.perf_counter()-started
            row[arm]=result
            print(name,arm,result['status'],round(result['worker_wall_seconds'],3),flush=True)
        if all(row[a]['status']=='CERTIFIED' for a in ('reference','revision')):
            a,b=row['reference'],row['revision']
            row['score_intervals_overlap']=Q(a['lower'])<=Q(b['upper']) and Q(b['lower'])<=Q(a['upper'])
            row['point_identical']=a['point']==b['point']
        rows.append(row)
        save(output/'results.json',rows)
    stats={}
    for arm in ('reference','revision'):
        times=[r[arm]['worker_wall_seconds'] for r in rows]
        stats[arm]=dict(requests=len(rows),certified=sum(r[arm]['status']=='CERTIFIED' for r in rows),
            errors=sum(r[arm]['status']=='ERROR' for r in rows),timeouts=sum(r[arm]['status']=='TIMEOUT' for r in rows),
            worker_errors=sum(r[arm]['status']=='WORKER_ERROR' or r[arm].get('returncode',0)!=0 for r in rows),
            total_wall_seconds=sum(times),median=float(np.median(times)),p90=float(np.percentile(times,90)),max=max(times))
    stats['certificate_conflicts']=sum(r.get('score_intervals_overlap') is False for r in rows)
    stats['revision_requests_qualified']=all(r['revision']['status']=='CERTIFIED' and
        r['revision'].get('returncode')==0 for r in rows) and not stats['certificate_conflicts']
    stats['source_verified_after']=all(hashlib.sha256(p.read_bytes()).hexdigest()==
        hashlib.sha256((output/'source_snapshot'/p.relative_to(ROOT)).read_bytes()).hexdigest() for p in code_paths)
    stats['whole_loss_speedup_established']=False
    stats['training_started']=False
    save(output/'summary.json',stats)
    return 0 if stats['revision_requests_qualified'] and stats['source_verified_after'] else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--arm',choices=('reference','revision'))
    args=parser.parse_args()
    if args.worker:
        worker(args.worker,args.arm)
    elif args.output:
        raise SystemExit(main(args.output))
    else:
        parser.error('provide --output or --worker')
