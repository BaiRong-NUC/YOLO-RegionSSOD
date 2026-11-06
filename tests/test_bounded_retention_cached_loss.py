"""No pressure gate can proceed on missing or changed point evidence."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('cached_loss', SCRIPTS / 'diagnose_bounded_retention_cached_loss.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def write_trace(path, count=2590, changed=False, incomplete=False):
    with path.open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(name='support', kind='enter', id=0,
            metadata={'identity':[11,22]})) + '\n')
        for i in range(count):
            stream.write(json.dumps(dict(name='point', kind='enter', id=i+1, parent=0,
                metadata={'domain_sha256':'domain', 'logits':[1, 2]})) + '\n')
            if not incomplete or i != count-1:
                stream.write(json.dumps(dict(name='point', kind='exit', id=i+1,
                    result={'point':['1', '2'], 'gap':'0', 'cells':3 if changed else 1,
                            'cache':{'hits':i}})) + '\n')
        stream.write(json.dumps(dict(name='support', kind='exit', id=0, result={})) + '\n')


def test_full_population_and_exact_records_pass(tmp_path):
    a,b = tmp_path/'a',tmp_path/'b'
    write_trace(a); write_trace(b)
    assert diag.compare_points(a,b)['equal']


@pytest.mark.parametrize('change', ['count', 'certificate', 'unfinished'])
def test_changed_or_missing_evidence_fails(tmp_path, change):
    a,b = tmp_path/'a',tmp_path/'b'
    write_trace(a)
    write_trace(b, count=2589 if change == 'count' else 2590,
                changed=change == 'certificate', incomplete=change == 'unfinished')
    with pytest.raises(RuntimeError):
        diag.compare_points(a,b)


def test_deadline_remains_absolute_and_numerics_are_not_extended():
    assert diag.LIMIT == 120
    assert diag.deadline(0, 10, 130)
    assert not diag.deadline(0, 10, 129.9)


@pytest.mark.parametrize('end', [130.05, float('nan'), float('inf'), 9])
def test_late_or_invalid_finished_worker_is_rejected(tmp_path, end):
    (tmp_path/'loss_boundary.json').write_text(json.dumps({'started':10}))
    (tmp_path/'loss_done.json').write_text(json.dumps({'finished':end}))
    with pytest.raises(RuntimeError):
        diag.validate_loss_window(tmp_path)


def test_completed_window_boundary_and_missing_receipt(tmp_path):
    (tmp_path/'loss_boundary.json').write_text(json.dumps({'started':10}))
    with pytest.raises(FileNotFoundError):
        diag.validate_loss_window(tmp_path)
    (tmp_path/'loss_done.json').write_text(json.dumps({'finished':130}))
    assert diag.validate_loss_window(tmp_path) == 120


def test_dtype_and_signed_zero_are_part_of_strict_calibration():
    import torch
    a=torch.tensor([0.], dtype=torch.float32)
    assert diag.tensor_bytes_equal(a,a.clone())
    assert not diag.tensor_bytes_equal(a,a.double())
    assert not diag.tensor_bytes_equal(a,-a)


@pytest.mark.parametrize('change', ['identity', 'reused_id'])
def test_identity_change_or_reused_event_id_is_rejected(tmp_path, change):
    a,b=tmp_path/'a',tmp_path/'b'
    write_trace(a); write_trace(b)
    lines=b.read_text().splitlines()
    if change == 'identity':
        row=json.loads(lines[0]); row['metadata']['identity']=[11,23]
        lines[0]=json.dumps(row)
    else:
        lines.insert(-1, lines[1]); lines.insert(-1, lines[2])
    b.write_text('\n'.join(lines)+'\n')
    with pytest.raises(RuntimeError):
        diag.compare_points(a,b)
