"""Independent point capacity revision: exact geometry, no training data."""
from fractions import Fraction as Q
from itertools import product

import pytest
import torch

from regionssod.training.bounded_retention_observation import build_ideal_observation
from regionssod.training.bounded_retention_domain_v6 import NativeEDomain
from regionssod.training.bounded_retention_continuous_numerics import DFLScore, NumericalCertificationError, optimize_point as old
from regionssod.training.bounded_retention_point_search_v2 import contract_bounds, optimize_point
from regionssod.training.bounded_retention_point_bound import joint_score_upper
from test_bounded_retention_continuous_domain import record


def fixture(n=2, flip=False, uncertain=False):
    p = build_ideal_observation([Q(9,10),Q(9,10),Q(1,5),Q(1,5)] if n==2 else
                              [Q(1,2),Q(3,4),Q(1),Q(1,2)])
    r = record(p, affine=(-10,10,40,20) if flip else (10,10,20,20))
    if uncertain:
        r['affine_bounds'] = tuple((a-Q(1,10000),b+Q(1,10000)) for a,b in r['affine_bounds'])
    box = ([30,28,32,30] if flip else [28,28,30,30]) if n==2 else [20,25,30,30]
    return NativeEDomain(r,box,(64,64))


@pytest.mark.parametrize('n', [2,3])
@pytest.mark.parametrize('flip', [False,True])
@pytest.mark.parametrize('uncertain', [False,True])
def test_conditional_contraction_preserves_all_checked_feasible_points(n,flip,uncertain):
    d=fixture(n,flip,uncertain)
    root=d.bounds
    hits=0
    for fractions in product((Q(0),Q(1,2)),repeat=n):
        cell=tuple((lo+f*(hi-lo),lo+(f+Q(1,2))*(hi-lo)) for (lo,hi),f in zip(root,fractions))
        contracted=contract_bounds(d,cell)
        for p in product(*(tuple(lo+(hi-lo)*i/4 for i in range(5)) for lo,hi in cell)):
            if d.point(p) is not None:
                hits+=1
                assert contracted is not None
                assert all(lo<=x<=hi for x,(lo,hi) in zip(p,contracted))
        if contracted is not None:
            assert all(a<=c<=e<=b for (a,b),(c,e) in zip(cell,contracted))
    assert hits>0


def test_area_coupled_two_u_global_score_known_closed_form():
    d=fixture()
    s=DFLScore(torch.arange(16,dtype=torch.float64).repeat(2,1),(28,28),(1,1),1)
    r=optimize_point(d,s)
    assert d.point(r.point) is not None
    # x,y are box side lengths: x,y>=2 and xy<=4/.35.
    supremum=Q(2)+Q(40,7)
    assert s.value_at(r.point)<=supremum<=s.value_at(r.point)+r.gap
    assert r.gap<=Q.from_float(1e-5)


@pytest.mark.parametrize('seed', range(4))
def test_reference_and_new_global_scores_overlap_on_nonflat_two_u(seed):
    torch.manual_seed(seed)
    d=fixture()
    s=DFLScore(torch.randn(2,16,dtype=torch.float64),(28,28),(1,1),1)
    a,b=old(d,s),optimize_point(d,s)
    va,vb=s.value_at(a.point),s.value_at(b.point)
    assert va<=vb+b.gap and vb<=va+a.gap
    assert d.point(b.point) is not None and b.diagnostics['qualified']


def test_single_unknown_uses_exact_original_solver_and_open_endpoint():
    p=build_ideal_observation([Q(3,4),Q(1,2),Q(1,2),Q(1,5)])
    d=NativeEDomain(record(p),[32,24,33,26],(64,64))
    s=DFLScore(-torch.arange(16,dtype=torch.float64)[None],(30,),(1,),1)
    a,b=old(d,s),optimize_point(d,s)
    assert a.point==b.point and a.gap==b.gap
    assert d.point(b.point) is not None


@pytest.mark.parametrize('flip',[False,True])
@pytest.mark.parametrize('uncertain',[False,True])
def test_same_axis_upper_bound_dominates_exact_feasible_grid(flip,uncertain):
    d=fixture(3,flip,uncertain)
    generator=torch.Generator().manual_seed(20261106)
    s=DFLScore(torch.randn(3,16,generator=generator,dtype=torch.float64),
               (25,25,25),(-1,1,1),1)
    for fractions in product((Q(0),Q(1,2)),repeat=3):
        cell=tuple((lo+f*(hi-lo),lo+(f+Q(1,2))*(hi-lo)) for (lo,hi),f in zip(d.bounds,fractions))
        upper,_=joint_score_upper(d,s,cell)
        assert upper<=s.cell_score_bounds(cell)[1]
        for p in product(*(tuple(lo+(hi-lo)*i/4 for i in range(5)) for lo,hi in cell)):
            if d.point(p) is not None:
                assert s.value_at(p)<=upper


def test_three_unknown_translation_ridge_certifies_without_capacity_increase():
    d=fixture(3)
    s=DFLScore(torch.arange(16,dtype=torch.float64).repeat(3,1),(25,25,25),(-1,1,1),1)
    r=optimize_point(d,s)
    # Sum clipped distances <= x-width + y-extension, with w>=10,h>=5,
    # wh<=50/.35; maximum w+h is attained at (w,h)=(200/7,5).
    assert s.value_at(r.point)<=Q(235,7)<=s.value_at(r.point)+r.gap
    assert r.diagnostics['cells']<=20000


def test_empty_domain_is_not_misreported_as_resource_limit():
    d=fixture()
    r=record(d.physical)
    empty=NativeEDomain(r,[100,100,101,101],(64,64))
    s=DFLScore(torch.zeros(2,16),(28,28),(1,1),1)
    with pytest.raises(NumericalCertificationError) as error:
        optimize_point(empty,s)
    assert error.value.diagnostics['status']=='NO_WITNESS'


def test_cell_budget_failure_preserves_uncertified_diagnostics():
    d=fixture()
    s=DFLScore(torch.arange(16,dtype=torch.float64).repeat(2,1),(28,28),(1,1),1)
    with pytest.raises(NumericalCertificationError) as error:
        optimize_point(d,s,max_cells=1)
    diag=error.value.diagnostics
    assert diag['status']=='RESOURCE_LIMIT' and not diag['qualified'] and diag['cells']==1
    assert diag['gap']>Q.from_float(1e-5)


@pytest.mark.parametrize('flip',[False,True])
def test_uncertain_clip_and_open_output_face_upper_and_contraction(flip):
    base=fixture(3,flip,True)
    rec=record(base.physical,affine=(-10,10,40,20) if flip else (10,10,20,20))
    rec['affine_bounds']=base.affine_bounds
    rec['augmentation_clip_window_bounds']=((Q(0),Q(1,10000)),(Q(0),Q(0)),
        (Q(42),Q(43)),(Q(33),Q(35)))
    d=NativeEDomain(rec,[20,31,30,32],(64,64))
    s=DFLScore(torch.arange(16,dtype=torch.float64).repeat(3,1),(25,25,25),(-1,1,1),1)
    cell=d.bounds
    c=contract_bounds(d,cell)
    upper,_=joint_score_upper(d,s,cell)
    hits=0
    for p in product(*(tuple(lo+(hi-lo)*i/8 for i in range(9)) for lo,hi in cell)):
        if d.point(p) is not None:
            hits+=1
            assert c is not None and all(lo<=x<=hi for x,(lo,hi) in zip(p,c))
            assert s.value_at(p)<=upper
    assert hits>0


def test_tied_saturated_point_selection_is_deterministic():
    d=fixture(3)
    s=DFLScore(torch.arange(16,dtype=torch.float64).repeat(3,1),(100,100,100),(1,1,1),1)
    a,b=optimize_point(d,s),optimize_point(d,s)
    assert a.point==b.point and a.gap==b.gap==0
    assert d.point(a.point) is not None
