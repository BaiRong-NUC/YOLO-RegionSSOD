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
        for i in range(count):
            stream.write(json.dumps(dict(name='point', kind='enter', id=i,
                metadata={'domain_sha256':'domain', 'logits':[1, 2]})) + '\n')
            if not incomplete or i != count-1:
                stream.write(json.dumps(dict(name='point', kind='exit', id=i,
                    result={'point':['1', '2'], 'gap':'0', 'cells':3 if changed else 1,
                            'cache':{'hits':i}})) + '\n')


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
