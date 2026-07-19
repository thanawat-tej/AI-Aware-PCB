#!/usr/bin/env python3
"""verify_collision_boundary.py -- Monte Carlo verification of the three-region
ordering d_coll <= d_req for ALL THREE scenario families, run directly against
the shipped classifier (compare_boundaries.py) so the verification certifies the
exact code that produces the thesis results, at the final parameter bindings.

For each random state the gap axis is scanned. The region must be monotone in
the gap (collision, then unsafe, then safe, each possibly empty) and the margin
d_req - d_coll must be strictly positive. Reports the minimum and median margin
per family.  Usage: python3 verify_collision_boundary.py [n_samples]
"""
import sys, numpy as np
from compare_boundaries import PCB, cls_decel, cls_cutin, cls_cutout, W_LAT

RANK = {'collision': 0, 'unsafe': 1, 'safe': 2}
GAP_MAX, TOL = 220.0, 1e-3

def _bisect(region_of_gap, want_rank, lo, hi):
    """Smallest gap in [lo,hi] whose rank >= want_rank (rank monotone in gap)."""
    if RANK[region_of_gap(hi)] < want_rank: return None
    if RANK[region_of_gap(lo)] >= want_rank: return lo
    while hi - lo > TOL:
        mid = 0.5*(lo+hi)
        if RANK[region_of_gap(mid)] >= want_rank: hi = mid
        else: lo = mid
    return hi

def scan(region_of_gap):
    """(d_coll, d_req, monotone). d_coll = sup of the collision region, d_req =
    inf of the safe region, found by bisection. Monotonicity of the region rank
    in the gap is checked on a coarse grid, which is what licenses bisection."""
    ranks=[RANK[region_of_gap(g)] for g in np.linspace(0.0,GAP_MAX,45)]
    mono=all(b>=a for a,b in zip(ranks,ranks[1:]))
    d_unsafe_start=_bisect(region_of_gap,1,0.0,GAP_MAX)   # end of collision region
    d_req=_bisect(region_of_gap,2,0.0,GAP_MAX)
    d_coll=0.0 if d_unsafe_start is None else d_unsafe_start
    if d_req is None: d_req=GAP_MAX
    return d_coll,d_req,mono

def run(n):
    """Per family: (min margin, median margin, monotone, n_with_boundary, n_trivially_safe).
    A state that is safe at zero gap has no boundary at all (every gap is safe),
    so its margin is undefined and it is counted separately, not as zero."""
    rng = np.random.default_rng(42)
    results = {}
    # deceleration
    m=[]; ok=True; trivial=0
    for _ in range(n):
        u=rng.uniform(3,20); w=rng.uniform(0,u+3)
        f=lambda gp: cls_decel(gp,u,w,PCB)
        if f(0.0)=='safe': trivial+=1; continue
        dc,dr,mono=scan(f); ok&=mono; m.append(dr-dc)
    results['deceleration']=(min(m),float(np.median(m)),ok,len(m),trivial)
    # cut-in (center-to-center d_y >= W_LAT, overlap-onset convention)
    m=[]; ok=True; trivial=0
    for _ in range(n):
        u=rng.uniform(6,20); w=rng.uniform(1,u); vy=rng.uniform(0.05,2.0); dy=rng.uniform(W_LAT+0.05,5.0)
        f=lambda gp: cls_cutin(gp,vy,u,w,dy,PCB)
        if f(0.0)=='safe': trivial+=1; continue
        dc,dr,mono=scan(f); ok&=mono; m.append(dr-dc)
    results['cut-in']=(min(m),float(np.median(m)),ok,len(m),trivial)
    # cut-out
    m=[]; ok=True; trivial=0
    for _ in range(n):
        u=rng.uniform(3,20); vr=rng.uniform(0,15); vyo=rng.uniform(0.2,3.0)
        tb=rng.uniform(0,1.0); dyo=rng.uniform(0.3,3.0)
        f=lambda gp: cls_cutout(gp,vyo,u,vr,tb,dyo,PCB)
        if f(0.0)=='safe': trivial+=1; continue
        dc,dr,mono=scan(f); ok&=mono; m.append(dr-dc)
    results['cut-out']=(min(m),float(np.median(m)),ok,len(m),trivial)
    return results

if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 2000
    res=run(n)
    print(f"Monte Carlo region-ordering verification | {n} samples per family | final bindings")
    print(f"{'family':>13} | {'min margin (m)':>15} | {'median (m)':>11} | {'boundary/trivial':>17} | monotone")
    allok=True
    for fam,(mn,md,mono,nb,tr) in res.items():
        print(f"{fam:>13} | {mn:15.3f} | {md:11.2f} | {nb:>8}/{tr:<8} | {'yes' if mono else 'NO'}")
        allok &= mono and mn>0
    print("(trivial = safe at zero gap, no boundary exists, margin undefined by construction)")
    print("RESULT:", "PASS (d_coll < d_req strictly wherever a boundary exists, regions monotone)" if allok else "FAIL")
