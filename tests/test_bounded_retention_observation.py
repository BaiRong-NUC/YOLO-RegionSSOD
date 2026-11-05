"""Synthetic source / exact rational checks; no real labels or model execution."""
from fractions import Fraction as Q
from itertools import product
import math

import numpy as np
import pytest

from regionssod.data.annotations import Tile, YoloBox, clip_box_to_tile
from regionssod.training.bounded_retention_observation import (
    Interval, UnsupportedObservation, _arithmetic_budget, _visible_vertices,
    build_ideal_observation, build_observation, serialized_interval,
)


def query(*, left=None, top=None, right=None, bottom=None):
    out = np.tile([-np.inf,np.inf],(1,4,1))
    for i,value in enumerate((left,top,right,bottom)):
        if value is not None:
            out[0,i] = value if isinstance(value,tuple) else (value,value)
    return out


def check_witness(obs, report, index=0):
    witness = report.witnesses[index]
    assert witness is not None
    for axis,ci,wi in ((witness.x,0,2),(witness.y,1,3)):
        assert axis.visible_left == max(axis.input_left,Q(0))
        assert axis.visible_right == min(axis.input_right,Q(1))
        assert obs.intervals[ci].contains((axis.visible_left+axis.visible_right)/2)
        assert obs.intervals[wi].contains(axis.visible_width)
        assert 0 < axis.input_width <= obs.source_ratio_upper
    assert witness.ratio == witness.visible_area / witness.input_area
    assert witness.area_cap == witness.visible_area / obs.effective_retention
    if report.mask[index]:
        assert witness.input_area <= witness.area_cap


@pytest.mark.parametrize("value", [0.,1.,.1,.99999999996,.100000000049,
                                     .0123456789123,1e-300,math.nextafter(0.,1.)])
def test_10g_interval_covers_actual_binary64_formatter(value):
    assert serialized_interval(format(value,'.10g')).contains(value)


@pytest.mark.parametrize("token", ['nan','inf','-0.1','1.1','abc','.12345678901','1e-999999'])
def test_malformed_tokens_fail_explicitly(token):
    with pytest.raises(UnsupportedObservation):
        build_observation([token,'.5','.2','.2'])


def test_float32_known_edge_counterexample_preserves_physical_witness():
    source = YoloBox(0,.2421875,.3125,.109375,.0625)
    box = clip_box_to_tile(source,Tile(160,160,260,260),640,640)
    tokens = [format(v,'.10g') for v in (box.x,box.y,box.w,box.h)]
    obs = build_observation(tokens)
    assert tokens == ['0.15','0.4','0.3','0.4']
    assert obs.unknown.tolist() == [True,False,False,False]
    nominal = np.asarray(tokens,dtype=np.float32)
    nominal_right = float(nominal[0]+nominal[2]/2)
    assert Q.from_float(nominal_right) != Q(3,10)
    # The nominal float32 value need not even lie inside the physical interval.
    assert not obs.edge_intervals[2].contains(nominal_right)
    report = obs.certify_ranges(query(left=-.4))
    assert report.mask.tolist() == [True]
    assert report.certificate['kind'] == 'SOURCE_OUTER'
    assert report.certificate['exact_source_inverse'] is False
    check_witness(obs,report)


def test_zero_envelope_threshold_equality_and_adjacent_float():
    obs = build_ideal_observation([Q(7,32),Q(1,2),Q(7,16),Q(1,4)])
    rows = np.concatenate([query(left=-.8125),query(left=math.nextafter(-.8125,-math.inf))])
    report = obs.certify_ranges(rows)
    assert report.mask.tolist() == [True,False]
    assert report.witnesses[0].ratio == Q(35,100)
    assert report.statuses[1] == 'RETENTION_EXCLUDED'
    assert report.certificate['kind'] == 'IDEAL_ALGEBRA_ONLY'
    check_witness(obs,report)


def test_binary64_addition_rounding_and_tolerance_are_in_certificate():
    obs = build_observation(['.21875','.5','.4375','.25'])
    cert = obs.certificate
    b = cert['arithmetic']
    u,eta = b['unit_roundoff'],b['underflow_half_ulp']
    tau,c = cert['source_tau'],cert['source_additive_tolerance']
    exact_k = (((tau-eta)/(1+u)-c)-eta)/(1+u)
    assert b['comparison_k'] <= exact_k < tau-c
    assert obs.effective_retention < b['comparison_k']
    assert obs.compatible_ranges(query(left=-.8125))[0]
    assert not obs.compatible_ranges(query(left=-.8125001))[0]


def test_unknown_contains_touch_outward_and_inward_rounding_ambiguity():
    obs = build_observation(['.2','.5','.4','.25'])
    rows = np.concatenate([query(left=0.),query(left=-.1),query(left=1e-11),query(left=1e-5)])
    report = obs.certify_ranges(rows)
    assert report.mask.tolist() == [True,True,True,False]
    assert obs.visible_edge_bounds[0,0] == 0
    assert obs.visible_edge_bounds[0,1] > 0
    for i in range(3):
        check_witness(obs,report,i)


def test_one_joint_witness_rejects_independent_center_width_splice():
    # width says R=.4 at L=0, center says R=.6: cannot choose one per constraint.
    obs = build_ideal_observation([Q(3,10),Q(1,2),Q(2,5),Q(1,4)])
    assert not obs.unknown.any()  # actual left=.1, not boundary
    report = obs.certify_ranges(query(left=0))
    assert report.statuses == ('UNSUPPORTED_KNOWN_EDGE_QUERY',)
    # Both-unknown records retain the same center/width joint constraints.
    obs2 = build_observation(['.5','.5','1','.25'])
    assert obs2.unknown.tolist() == [True,False,True,False]
    assert not obs2.compatible_ranges(query(left=.1,right=1.))[0]
    ok = obs2.certify_ranges(query(left=-.25,right=1.25))
    assert ok.mask[0]
    check_witness(obs2,ok)


def test_two_axis_area_coupling_uses_same_visible_witness():
    obs = build_ideal_observation([Q(1,8),Q(1,8),Q(1,4),Q(1,4)])
    rows = np.concatenate([query(left=-.25,top=0),query(left=0,top=-.25),
                           query(left=-.25,top=-.25)])
    report = obs.certify_ranges(rows)
    assert report.mask.tolist() == [True,True,False]
    assert report.witnesses[2].ratio == Q(1,4)
    for i in range(3):
        check_witness(obs,report,i)


def test_noneunknown_requires_joint_visible_feasibility():
    obs = build_ideal_observation([Q(1,2),Q(1,2),Q(1,4),Q(1,4)])
    assert not obs.unknown.any()
    report = obs.certify_ranges(query())
    assert report.mask[0] and report.witnesses[0].ratio == 1
    check_witness(obs,report)
    with pytest.raises(UnsupportedObservation,match='EMPTY_VISIBLE_JOINT'):
        build_observation(['.1','.5','.8','.2'])


def test_halfline_queries_keep_original_unknown_existential():
    obs = build_ideal_observation([Q(1,8),Q(1,2),Q(1,4),Q(1,4)])
    rows = np.concatenate([query(left=(-np.inf,-.25)),query(left=(-np.inf,-1)),
                           query(left=(0,.1))])
    report = obs.certify_ranges(rows)
    assert report.mask.tolist() == [True,False,True]
    assert report.witnesses[0].x.input_left == Q(-1,4)
    assert report.witnesses[2].x.input_left == 0
    for i in range(3):
        check_witness(obs,report,i)


def test_right_halfline_and_mirrored_inward_ambiguity():
    obs = build_observation(['.8','.5','.4','.25'])
    assert obs.unknown.tolist() == [False,False,True,False]
    rows = np.concatenate([query(right=(1.25,np.inf)),query(right=(3,np.inf)),
                           query(right=1-1e-11)])
    report = obs.certify_ranges(rows)
    assert report.mask.tolist() == [True,False,True]
    for i in range(3):
        check_witness(obs,report,i)


def test_invalid_and_negative_width_queries_do_not_clamp():
    obs = build_observation(['.5','.5','1','.25'])
    rows = np.concatenate([query(left=(1.,0.),right=1.),
                           query(left=np.nan,right=1.),query(left=np.inf,right=1.),
                           query(left=.8,right=.2),query(left=-.5,right=1.)])
    report = obs.certify_ranges(rows)
    assert report.statuses[:3] == ('INVALID_RANGE',)*3
    assert report.statuses[3] == 'EMPTY_JOINT_GEOMETRY'
    assert report.mask.tolist() == [False,False,False,False,True]
    check_witness(obs,report,4)


def test_input_width_cap_cannot_be_bypassed_by_retention_tolerance():
    obs = build_ideal_observation([Q(1,2),Q(1,2),1,Q(1,4)],source_ratio_upper=2,retention=Q(1,10))
    assert obs.compatible_ranges(query(left=-.5,right=1.5))[0]
    assert not obs.compatible_ranges(query(left=-1,right=2))[0]


def test_points_ignore_known_nan_and_empty_batch():
    obs = build_observation(['.2','.5','.4','.25'])
    assert obs.compatible(np.array([[-.1,np.nan,np.nan,np.nan]])).tolist() == [True]
    assert obs.compatible_ranges(np.empty((0,4,2))).shape == (0,)
    with pytest.raises(ValueError,match='ranges'):
        obs.compatible_ranges(np.zeros((4,2)))


@pytest.mark.parametrize('kwargs', [dict(source_ratio_upper=.9),dict(source_ratio_upper=np.inf),
                                   dict(tile_size_lower=0),dict(tile_size_upper=2**53),
                                   dict(min_visibility=0),dict(retention_tolerance=.35)])
def test_unproven_numeric_domains_fail_explicitly(kwargs):
    with pytest.raises(UnsupportedObservation):
        build_observation(['.5','.5','.2','.2'],**kwargs)


def test_tiny_area_cannot_be_reported_as_certified_acceptance():
    with pytest.raises(UnsupportedObservation,match='UNPROVEN_POSITIVE|UNINFORMATIVE'):
        build_observation(['.5','.5','1e-20','1e-20'])
    with pytest.raises(UnsupportedObservation,match='UNINFORMATIVE'):
        build_observation(['.5','.5','1e-8','1e-8'])


def test_public_bounds_and_fraction_error_budgets_are_cached():
    before = _arithmetic_budget.cache_info().hits
    obs = build_observation(['.15','.4','.3','.4'])
    after = _visible_vertices.cache_info().hits
    same = build_observation(['.15','.4','.3','.4'])
    assert _arithmetic_budget.cache_info().hits > before
    assert _visible_vertices.cache_info().hits > after
    assert obs.certificate['source_dimension_upper'] < 2**53
    # Per-observation certificate mutation cannot poison the shared cache.
    obs.certificate['arithmetic']['center'] = Q(100)
    again = build_observation(['.15','.4','.3','.4'])
    assert again.intervals == same.intervals


def test_fraction_grid_oracle_for_fixed_visible_boxes_and_all_clip_patterns():
    # Independent direct clip/ratio evaluation, not polygon-vertex enumeration.
    for coords in ([Q(1,8),Q(1,2),Q(1,4),Q(1,4)],
                   [Q(7,8),Q(1,2),Q(1,4),Q(1,4)],
                   [Q(1,2),Q(1,2),Q(1),Q(1,4)],
                   [Q(1,8),Q(1,8),Q(1,4),Q(1,4)]):
        obs = build_ideal_observation(coords,source_ratio_upper=4)
        visible = [coords[0]-coords[2]/2,coords[1]-coords[3]/2,
                   coords[0]+coords[2]/2,coords[1]+coords[3]/2]
        unknown = np.flatnonzero(obs.unknown)
        for atoms in product([Q(-1),Q(-1,4),Q(0),Q(1,4),Q(1),Q(5,4),Q(2)],repeat=len(unknown)):
            g = list(visible)
            ranges = query()
            for index,value in zip(unknown,atoms):
                g[index] = value
                ranges[0,index] = float(value)
            vl,vt,vr,vb = max(g[0],0),max(g[1],0),min(g[2],1),min(g[3],1)
            expected = False
            if g[2]>g[0] and g[3]>g[1] and [vl,vt,vr,vb]==visible:
                ratio = (vr-vl)*(vb-vt)/((g[2]-g[0])*(g[3]-g[1]))
                expected = ratio>=Q(35,100) and g[2]-g[0]<=4 and g[3]-g[1]<=4
            assert bool(obs.compatible_ranges(ranges)[0]) == expected


def test_actual_exporter_synthetic_retained_boxes_are_covered():
    # No image decoder, source labels, files, or dataset constructor. Cover
    # interior/touch/cut, image-exterior input corners, and source sizes.
    count = 0
    for width,height,tile in [(640,640,Tile(160,160,260,260)),
                              (1024,768,Tile(0,0,256,256)),
                              (1920,1080,Tile(800,300,1120,620))]:
        for cx,cy,bw,bh in product([.125,.25,.5,.75],[.125,.375,.5],
                                   [.03125,.125,.5,1.],[.0625,.25,.5]):
            source = YoloBox(0,cx,cy,bw,bh)
            result = clip_box_to_tile(source,tile,width,height)
            if result is None:
                continue
            tokens = [format(v,'.10g') for v in (result.x,result.y,result.w,result.h)]
            obs = build_observation(tokens,source_ratio_upper=12.)
            ideal = [((Q(cx)-Q(bw)/2)*width-tile.x1)/tile.width,
                     ((Q(cy)-Q(bh)/2)*height-tile.y1)/tile.height,
                     ((Q(cx)+Q(bw)/2)*width-tile.x1)/tile.width,
                     ((Q(cy)+Q(bh)/2)*height-tile.y1)/tile.height]
            # The actual rational edge may not be a double; query its outward
            # one-ulp enclosure instead of silently replacing it by a point.
            ranges = query()
            for i in np.flatnonzero(obs.unknown):
                f = float(ideal[i])
                ranges[0,i] = (math.nextafter(f,-math.inf),math.nextafter(f,math.inf))
            report = obs.certify_ranges(ranges)
            assert report.mask[0], (tokens,report.statuses,ideal)
            check_witness(obs,report)
            count += 1
    assert count >= 30


def test_fraction_oracle_checks_distinct_ieee_area_paths_and_output_envelopes():
    rng = np.random.default_rng(431)
    width,height,tile = 1920,1080,Tile(800,300,1120,620)
    observed = 0
    for _ in range(120):
        cx,cy = rng.uniform(-.1,1.1,2)
        bw,bh = rng.uniform(.01,1.6,2)
        src = YoloBox(0,(tile.x1+tile.width*cx)/width,
                      (tile.y1+tile.height*cy)/height,
                      tile.width*bw/width,tile.height*bh/height)
        result = clip_box_to_tile(src,tile,width,height)
        if result is None:
            continue
        tokens = [format(v,'.10g') for v in (result.x,result.y,result.w,result.h)]
        obs = build_observation(tokens)
        exact = [((Q(src.x)-Q(src.w)/2)*width-tile.x1)/tile.width,
                 ((Q(src.y)-Q(src.h)/2)*height-tile.y1)/tile.height,
                 ((Q(src.x)+Q(src.w)/2)*width-tile.x1)/tile.width,
                 ((Q(src.y)+Q(src.h)/2)*height-tile.y1)/tile.height]
        l,t,r,b = max(exact[0],0),max(exact[1],0),min(exact[2],1),min(exact[3],1)
        ideal_xywh = ((l+r)/2,(t+b)/2,r-l,b-t)
        for interval,value in zip(obs.intervals,ideal_xywh):
            assert interval.contains(value)
        budget = obs.certificate['arithmetic']
        # Distinct source arithmetic paths, evaluated by binary64, compared
        # against an independent exact-rational ideal geometry.
        ax1=max((src.x-src.w/2)*width,float(tile.x1))
        ax2=min((src.x+src.w/2)*width,float(tile.x2))
        ay1=max((src.y-src.h/2)*height,float(tile.y1))
        ay2=min((src.y+src.h/2)*height,float(tile.y2))
        n_hat=Q((ax2-ax1)*(ay2-ay1))/(tile.width*tile.height)
        d_hat=Q((src.w*width)*(src.h*height))/(tile.width*tile.height)
        ideal_n=(r-l)*(b-t)
        ideal_d=(exact[2]-exact[0])*(exact[3]-exact[1])
        assert abs(n_hat-ideal_n)<=budget['numerator']
        assert abs(d_hat-ideal_d)<=budget['denominator']
        assert n_hat>=budget['comparison_k']*d_hat
        assert ideal_n/ideal_d>=obs.effective_retention
        assert ideal_n+obs.certificate['additive_area_slack']>=budget['comparison_k']*ideal_d
        ranges=query()
        for i in np.flatnonzero(obs.unknown):
            value=float(exact[i])
            ranges[0,i]=(math.nextafter(value,-math.inf),math.nextafter(value,math.inf))
        report=obs.certify_ranges(ranges)
        assert report.mask[0]
        check_witness(obs,report)
        observed+=1
    assert observed>=30
