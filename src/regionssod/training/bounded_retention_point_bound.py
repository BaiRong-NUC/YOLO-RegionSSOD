"""Exact same-axis score upper bounds over a necessary source polygon.

This is an upper bound, NOT an alternative observation set or point control.
Every DFL piecewise-affine axis score has an affine majorant on its output
cell. For a fixed source endpoint the majorant is maximized at the appropriate
reachable interval endpoint. Interval affine transport and ordered clipping
make that endpoint piecewise affine. Cutting the source polygon at all its
breakpoints reduces the maximum to vertices of a finite planar subdivision.
Cross-axis retention is relaxed only for the upper bound. No witness is
accepted here. Strict positive-output faces are relaxed to their closure.
"""
from fractions import Fraction as Q
from itertools import combinations
import math

from .bounded_retention_continuous_domain import FREE, _axis_options, _branches, _planes, _forward
from .bounded_retention_observation import _vertices


def conditional_polygons(domain, cell, axis):
    ranges=domain._ranges(cell)
    if ranges is None:
        return ()
    intervals,unknown,span_cap,tau=domain._key
    other=_axis_options(domain._key,1-axis,ranges)
    if not other:
        return ()
    needed=tau/max(w.ratio for _,w in other)
    u=(unknown[axis],unknown[axis+2])
    result=[]
    for branch in _branches(u):
        base,lv,rv=_planes(intervals[axis],intervals[axis+2],u,span_cap,branch)
        planes=list(base)
        for k,(lo,hi) in enumerate((ranges[axis],ranges[axis+2])):
            if lo!=FREE[0]:
                planes.append((-Q(1) if k==0 else Q(0),-Q(1) if k==1 else Q(0),-lo))
            if hi!=FREE[1]:
                planes.append((Q(1) if k==0 else Q(0),Q(1) if k==1 else Q(0),hi))
        na,nb,nc=(rv[k]-lv[k] for k in range(3))
        planes.append((-needed-na,needed-nb,nc))
        vertices=_vertices(planes)
        if vertices:
            result.append((tuple(dict.fromkeys(planes)),vertices))
    return tuple(result)


def _majorant(score,j,cell):
    lo,hi=cell
    knots=score.breakpoints(j,lo,hi)
    if lo==hi:
        return Q(0),score.axis_value(j,lo)
    middle=(lo+hi)/2
    a,b=next((a,b) for a,b in zip(knots,knots[1:]) if a<=middle<=b)
    slope=(score.axis_value(j,b)-score.axis_value(j,a))/(b-a)
    intercept=max(score.axis_value(j,x)-slope*x for x in knots)
    return slope,intercept


def joint_score_upper(domain,score,cell):
    """Return a valid tighter bound plus unverified candidate point hints."""
    axis_upper=[max(score.axis_value(j,x) for x in score.breakpoints(j,*cell[j]))
                for j in range(domain.dimension)]
    upper=sum(axis_upper,Q(0))
    hints=[]
    for axis in (0,1):
        indices=[j for j,s in enumerate(domain.active_sides) if s%2==axis]
        if len(indices)!=2:
            continue
        majorants={j:_majorant(score,j,cell[j]) for j in indices}
        bound=None
        best_outputs=None
        tied_outputs=[]
        for planes,vertices in conditional_polygons(domain,cell,axis):
            cuts=[set(),set()]
            for j in indices:
                side=domain.active_sides[j]
                source_side=domain._mapping[side]
                coord=0 if source_side<2 else 1
                scale,offset,lower,higher=domain._transport(side)
                slope,_=majorants[j]
                if slope==0:
                    continue
                cuts[coord].add(Q(0))
                # All switches of scale endpoint at g=0; all clipping and
                # intersection switches when s*g+t crosses a finite face.
                faces=(cell[j][1],higher[1],lower[1]) if slope>0 else (cell[j][0],lower[0],higher[0])
                t=offset[1] if slope>0 else offset[0]
                for v in faces:
                    if not math.isfinite(v):
                        continue
                    for s in scale:
                        cuts[coord].add((v-t)/s)
            lines=list(planes)
            for k in (0,1):
                lo,hi=min(p[k] for p in vertices),max(p[k] for p in vertices)
                lines.extend((Q(1) if k==0 else Q(0),Q(1) if k==1 else Q(0),v)
                             for v in sorted(cuts[k]) if lo<v<hi)
            points=set(vertices)
            for (a,b,c),(d,e,f) in combinations(tuple(dict.fromkeys(lines)),2):
                det=a*e-b*d
                if not det:
                    continue
                x,y=(c*e-b*f)/det,(a*f-c*d)/det
                if all(p*x+q*y<=r for p,q,r in planes):
                    points.add((x,y))
            for point in sorted(points):
                value=Q(0)
                outputs={}
                for j in indices:
                    side=domain.active_sides[j]
                    g=point[0 if domain._mapping[side]<2 else 1]
                    reached=_forward((g,g),*domain._transport(side))
                    if reached is None:
                        raise ValueError('source polygon has invalid transport')
                    lo,hi=max(cell[j][0],reached[0]),min(cell[j][1],reached[1])
                    if lo>hi:
                        raise ValueError('inverse range disagrees with forward transport')
                    slope,intercept=majorants[j]
                    x=hi if slope>0 else lo
                    outputs[j]=x
                    value+=slope*x+intercept
                if bound is None or value>bound:
                    bound,best_outputs=value,outputs
                    tied_outputs=[outputs]
                elif value==bound:
                    tied_outputs.append(outputs)
        if bound is not None:
            baseline=sum(axis_upper[j] for j in indices)
            upper-=max(Q(0),baseline-bound)
            hint=list(score.cell_maximizer(cell))
            for j,x in best_outputs.items():
                hint[j]=x
            hints.append(tuple(hint))
            mean=list(hint)
            for j in indices:
                mean[j]=sum((p[j] for p in tied_outputs),Q(0))/len(tied_outputs)
            hints.append(tuple(mean))
    return upper,hints
