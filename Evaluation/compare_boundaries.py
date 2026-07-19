"""
Three-way safety-envelope comparison: PCB vs RSS vs JRC.
Parameterized classifiers (one param set per envelope), matching the
monotonicity-fixed logic of pcb_analysis_new.py:
  - deceleration: lead-faster short-circuit DISABLED (it broke cut-in reuse)
  - cut-in: worst region over the FIXED entry-time grid across the budget
    interval, so raising EPS_Y/GAMMA_Y can only ADD detections (FN never rises)
Confusion matrix per model, read as:
  FN = predicted SAFE but (unacceptable) collision -> SOUNDNESS violation
  FP = predicted DANGER but no unacceptable collision -> CONSERVATIVENESS
"""
import pandas as pd, numpy as np
from dataclasses import dataclass

# Exact JRC cut-in mode: when True (or with the --jrc-sim CLI flag), JRC cut-in rows
# are classified by replaying the vendored closed-loop R157 CC driver (jrc_sim.py),
# whose crash outcome includes the late trigger (lateral overlap + TTC<=2s) that the
# three-phase stopping-distance approximation cannot represent. Costs ~0.2 ms/row.
# Note: exact mode returns only safe/unsafe for JRC cut-in (no collision region).
JRC_CUTIN_SIM = False

# --- entry-time grid (set to match pcb_analysis_new.py if yours differ) ------
T_ENTRY_CAP  = 8.0
T_ENTRY_STEP = 0.05
_T_GRID = np.arange(0.0, T_ENTRY_CAP + 1e-9, T_ENTRY_STEP)
# Lateral overlap (collision threat) begins at EDGE contact, i.e. when the
# center-to-center lateral gap d_y falls to the sum of half-widths. d_y is
# reported center-to-center, so the intruder is a threat from d_y = W_LAT, not
# d_y = 0 (full alignment). Set to (W_ego + W_obs)/2 for your vehicles.
W_LAT = 1.9
_REGION_RANK = {'safe': 0, 'unsafe': 1, 'collision': 2}
def _worse(a, b): return a if _REGION_RANK[a] >= _REGION_RANK[b] else b

@dataclass
class P:
    name: str
    DELTA_SYS: float; RHO_ACT: float
    EPS_MAX: float; GAMMA_MAX: float; EPS_Y_MAX: float; GAMMA_Y_MAX: float
    ALPHA_MAX: float = 3.0; BETA_MIN: float = 4.0; BETA_MAX: float = 8.0
    V_FLOOR: float = 0.1
    ego_model: str = 'three_phase'   # 'three_phase' (PCB), 'jrc_cc' (JRC), or 'rss'
    # RSS-native constants (Shalev-Shwartz et al., Lemma 2): single response time rho,
    # ego accelerates at RSS_A_ACCEL during rho then brakes at RSS_A_MIN_BRAKE; lead
    # brakes at RSS_A_MAX_BRAKE. Only read when ego_model=='rss'.
    RSS_RHO: float = 0.6; RSS_A_ACCEL: float = 3.0
    RSS_A_MIN_BRAKE: float = 4.0; RSS_A_MAX_BRAKE: float = 8.0
    RSS_A_ACCEL_LAT: float = 1.0; RSS_A_MIN_BRAKE_LAT: float = 1.0; RSS_MU: float = 0.3

# --- the three envelopes -----------------------------------------------------
PCB = P("PCB", 0.15, 0.15, 0.97, 0.70, 0.32, 0.18)                       # uncertainty-aware
RSS = P("RSS", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ego_model='rss',           # canonical RSS (Lemmas 2 & 4): own rho, no ramp, no budgets
        RSS_RHO=0.3, RSS_A_ACCEL=3.0, RSS_A_MIN_BRAKE=4.0, RSS_A_MAX_BRAKE=8.0,
        RSS_A_ACCEL_LAT=1.0, RSS_A_MIN_BRAKE_LAT=1.0, RSS_MU=0.3)
JRC = P("JRC", 0.15, 0.6, 0.0,  0.0,  0.0,  0.0, ego_model='jrc_cc')   # CC-driver braking, perfect perception

# --- ego stopping-distance models -------------------------------------------
def jrc_cc_stop(v, tau=0.75, a_rel=0.4, j=12.65, a_max=7.59):
    d1 = v*tau - 0.5*a_rel*tau**2; v1 = max(0.0, v - a_rel*tau)
    tj = (a_max - a_rel)/j
    v2 = max(0.0, v1 - (a_rel*tj + 0.5*j*tj**2))
    d2 = max(0.0, v1*tj - 0.5*a_rel*tj**2 - j*tj**3/6)
    return d1 + d2 + v2**2/(2*a_max)

def rss_safe_distance(v_r, v_f, p):
    """Canonical RSS longitudinal safe distance (Shalev-Shwartz et al., Lemma 2):
    d = [ v_r*rho + 0.5*a*rho^2 + (v_r + rho*a)^2/(2*a_min_brake)
          - v_f^2/(2*a_max_brake) ]_+ . Ego accelerates during a single response
    time rho then brakes at its MINIMUM braking; lead brakes at its MAXIMUM. No
    perception margin: RSS assumes the perceived state equals the true state."""
    rho, a = p.RSS_RHO, p.RSS_A_ACCEL
    d = (v_r*rho + 0.5*a*rho**2 + (v_r + rho*a)**2/(2*p.RSS_A_MIN_BRAKE)
         - v_f**2/(2*p.RSS_A_MAX_BRAKE))
    return max(0.0, d)

def rss_safe_distance_lat(v1, v2, p):
    """Canonical RSS lateral safe distance (Shalev-Shwartz et al., Lemma 4 / Def. 33).
    Both cars accelerate laterally toward each other at a_accel_lat during rho, then
    brake at a_min_brake_lat to zero lateral velocity; safe if final separation >= mu.
    c1 is the cut-in vehicle closing at v1>=0 toward the ego (v2=0, ego holds lane)."""
    rho, a = p.RSS_RHO, p.RSS_A_ACCEL_LAT
    v1r = v1 + rho*a
    v2r = 0.0 - rho*a
    term1 = (v1 + v1r)/2.0*rho + v1r**2/(2*p.RSS_A_MIN_BRAKE_LAT)
    term2 = (0.0 + v2r)/2.0*rho - v2r**2/(2*p.RSS_A_MIN_BRAKE_LAT)
    return p.RSS_MU + max(0.0, term1 - term2)

def cls_cutin_rss(p_long, v_y_rel, u_t, w_cut, d_y, p):
    """RSS cut-in: a state is DANGEROUS only when BOTH the longitudinal and the
    lateral distances are unsafe (RSS Def. 33: a collision requires proximity in
    both axes). Longitudinal via Lemma 2, lateral via Lemma 4. No perception
    margin. Returns safe / unsafe (RSS has no separate unavoidable-collision zone
    in its cut-in rule, so we mark the physically-unavoidable subset collision
    using best-case longitudinal braking, consistent with the other envelopes)."""
    d_long_safe = rss_safe_distance(u_t, w_cut, p)         # Lemma 2 (long.)
    d_lat_safe  = rss_safe_distance_lat(abs(v_y_rel), 0.0, p)  # Lemma 4 (lat.)
    long_unsafe = p_long < d_long_safe
    lat_unsafe  = (d_y - W_LAT) < d_lat_safe               # edge clearance vs lateral safe dist
    if not (long_unsafe and lat_unsafe):
        return 'safe'
    collision_thresh = max(0.0, (u_t**2 - max(0.0, w_cut)**2)/(2*p.RSS_A_MAX_BRAKE))
    return 'collision' if p_long < collision_thresh else 'unsafe'

def ego_stop(u, p):
    if p.ego_model == 'jrc_cc':
        return jrc_cc_stop(u)
    uA = u + p.ALPHA_MAX*p.DELTA_SYS
    uB = max(0.0, uA + (p.ALPHA_MAX-p.BETA_MIN)*p.RHO_ACT/2.0)
    dxA = u*p.DELTA_SYS + 0.5*p.ALPHA_MAX*p.DELTA_SYS**2
    dxB = uA*p.RHO_ACT + (2*p.ALPHA_MAX-p.BETA_MIN)*p.RHO_ACT**2/6.0
    dxC = uB**2/(2*p.BETA_MIN) if uB > 0 else 0.0
    return dxA+dxB+dxC

def proj_ego(u, T, p):
    if T <= 0: return u, 0.0
    if T <= p.DELTA_SYS: return u+p.ALPHA_MAX*T, u*T+0.5*p.ALPHA_MAX*T**2
    uA = u+p.ALPHA_MAX*p.DELTA_SYS; dxA = u*p.DELTA_SYS+0.5*p.ALPHA_MAX*p.DELTA_SYS**2
    if T <= p.DELTA_SYS+p.RHO_ACT:
        tau=T-p.DELTA_SYS
        up=uA+p.ALPHA_MAX*tau-(p.ALPHA_MAX+p.BETA_MIN)*tau**2/(2*p.RHO_ACT)
        dxB=uA*tau+p.ALPHA_MAX*tau**2/2-(p.ALPHA_MAX+p.BETA_MIN)*tau**3/(6*p.RHO_ACT)
        return up, dxA+dxB
    uB=max(0.0,uA+(p.ALPHA_MAX-p.BETA_MIN)*p.RHO_ACT/2)
    dxBf=uA*p.RHO_ACT+(2*p.ALPHA_MAX-p.BETA_MIN)*p.RHO_ACT**2/6
    dtC=T-p.DELTA_SYS-p.RHO_ACT
    if uB>0 and dtC<=uB/p.BETA_MIN:
        up=uB-p.BETA_MIN*dtC; dxC=uB*dtC-0.5*p.BETA_MIN*dtC**2
    else:
        up=0.0; dxC=uB**2/(2*p.BETA_MIN) if uB>0 else 0.0
    return up, dxA+dxBf+dxC

# --- classifiers (monotonicity-fixed) ---------------------------------------
def cls_decel(p_t, u_t, w_t, p, lead_faster_safe=True):
    if p.ego_model == 'rss':
        required = rss_safe_distance(u_t, w_t, p)
        collision_thresh = (u_t**2 - w_t**2)/(2*p.RSS_A_MAX_BRAKE)
        if p_t < collision_thresh: return 'collision'
        if p_t < required:         return 'unsafe'
        return 'safe'
    v_e=max(0.0, w_t-p.GAMMA_MAX-p.BETA_MAX*p.DELTA_SYS)
    required=max(0.0, p.EPS_MAX+ego_stop(u_t,p)-v_e**2/(2*p.BETA_MAX))
    w_best=max(0.0, w_t-p.GAMMA_MAX)
    collision_thresh=u_t**2/(2*p.BETA_MAX)-w_best**2/(2*p.BETA_MAX)
    # lead-faster short-circuit DISABLED (was: if lead_faster_safe and u_t<w_t: return 'safe')
    if p_t<collision_thresh: return 'collision'
    if p_t<required: return 'unsafe'
    return 'safe'

def _cutin_at_entry_time(p_long, u_t, w_cut, T_entry, p):
    if T_entry <= 0:
        return cls_decel(p_long, u_t, w_cut, p, lead_faster_safe=False)   # Case 2
    u_at, dx_ego = proj_ego(u_t, T_entry, p)                              # Case 3
    w_cons=max(0.0, w_cut-p.GAMMA_MAX)
    p_long_proj=(p_long-p.EPS_MAX)+w_cons*T_entry-dx_ego
    s_ego_best=u_t**2/(2*p.BETA_MAX); s_cut_best=w_cut*T_entry
    if p_long+s_cut_best-s_ego_best < 0: return 'collision'
    v_e_proj=max(0.0, w_cons-p.GAMMA_MAX-p.BETA_MAX*p.DELTA_SYS)
    required=max(0.0, p.EPS_MAX+ego_stop(u_at,p)-v_e_proj**2/(2*p.BETA_MAX))
    if p_long_proj<required: return 'unsafe'
    return 'safe'

def cls_cutin(p_long, v_y_rel, u_t, w_cut, d_y, p):
    if p.ego_model == 'rss':
        return cls_cutin_rss(p_long, v_y_rel, u_t, w_cut, d_y, p)
    if v_y_rel+p.GAMMA_Y_MAX <= 0: return 'safe'                          # Case 1
    d_ov = d_y - W_LAT                        # lateral EDGE clearance -> distance to overlap onset
    if d_ov <= 0:                             # already overlapping -> in-lane lead right now
        return cls_decel(p_long, u_t, w_cut, p, lead_faster_safe=False)
    T_nom=d_ov/max(p.V_FLOOR, v_y_rel)
    T_min=max(0.0, (d_ov-p.EPS_Y_MAX)/(v_y_rel+p.GAMMA_Y_MAX))
    T_max=min(T_ENTRY_CAP, (d_ov+p.EPS_Y_MAX)/max(p.V_FLOOR, v_y_rel-p.GAMMA_Y_MAX))
    region=_cutin_at_entry_time(p_long, u_t, w_cut, T_nom, p)
    if region != 'collision':
        for T in _T_GRID[(_T_GRID>=T_min)&(_T_GRID<=T_max)]:
            region=_worse(region, _cutin_at_entry_time(p_long, u_t, w_cut, T, p))
            if region=='collision': break
    return region

def cls_cutout(p_rev, v_y_occ, u_t, v_rev, t_bl, d_y_occ, p):
    t_rev=(d_y_occ+p.EPS_Y_MAX)/max(p.V_FLOOR, v_y_occ-p.GAMMA_Y_MAX)
    v_eff=max(0.0, v_rev-p.GAMMA_MAX-p.BETA_MAX*(t_bl+t_rev))
    required=max(0.0, p.EPS_MAX+ego_stop(u_t,p)-v_eff**2/(2*p.BETA_MAX))
    v_best=max(0.0, v_rev-p.GAMMA_MAX)
    collision_thresh=u_t**2/(2*p.BETA_MAX)-v_best**2/(2*p.BETA_MAX)
    if p_rev<collision_thresh: return 'collision'
    if p_rev<required: return 'unsafe'
    return 'safe'

def g(r,k):
    v=r.get(k)
    return None if v is None or (isinstance(v,float) and np.isnan(v)) else float(v)

def classify(r,p):
    s=r['scenario']
    if s=='deceleration': return cls_decel(g(r,'p_t'),g(r,'u_t'),g(r,'w_t'),p)
    if s=='cutin':
        if JRC_CUTIN_SIM and p.ego_model=='jrc_cc':
            vals=[g(r,'u_t'),g(r,'w_cut'),g(r,'v_y_rel'),g(r,'p_long_t'),g(r,'d_y')]
            if all(v is not None for v in vals):
                import jrc_sim
                return jrc_sim.classify_cutin_row(*vals)
        return cls_cutin(g(r,'p_long_t'),g(r,'v_y_rel'),g(r,'u_t'),g(r,'w_cut'),g(r,'d_y'),p)
    if s=='cutout':       return cls_cutout(g(r,'p_rev'),g(r,'v_y_occ'),g(r,'u_t'),g(r,'v_rev_hat'),g(r,'t_blackout'),g(r,'d_y_occ'),p)
    return 'safe'

VACC = 10/3.6   # 2.78 m/s -- impacts at or below this are minor (see v_acc)

def confusion(df, p, vacc=VACC, label_col='collision_occurred'):
    """FN = predicted SAFE but an UNACCEPTABLE collision occurred -> soundness violation (target 0).
       FP = predicted DANGER but no unacceptable collision -> conservativeness.
       map_to_corrected.py snapshots at CLOSEST APPROACH, so the impact speed is the
       closing speed u_t - ground_truth_vel there. Pass vacc=None to score against ANY
       collision instead of only unacceptable-severity ones."""
    TP=FP=TN=FN=0; miss=[]
    for _,r in df.iterrows():
        d=r.to_dict()
        if pd.isna(d.get(label_col)): continue
        actual = int(d[label_col])==1
        if actual and vacc is not None:
            u=g(d,'u_t'); gv=g(d,'ground_truth_vel')
            if u is not None and gv is not None:
                actual = (u-gv) > vacc          # keep only unacceptable-severity collisions
        pred = classify(d,p)!='safe'
        if pred and actual: TP+=1
        elif pred and not actual: FP+=1
        elif not pred and not actual: TN+=1
        else: FN+=1; miss.append(d.get('filename_param'))
    return dict(TP=TP,FP=FP,TN=TN,FN=FN,miss=miss)

def sweep(files, vacc=VACC):
    """Per-sigma breakdown: one CSV per sigma level. FN=soundness, FP=conservativeness.
    Rows sorted by the first number in each filename (name files by sigma, e.g. sig0.45.csv)."""
    import os, re
    rows=[]
    for f in files:
        df=pd.read_csv(f)
        lab=os.path.basename(f).replace('.csv','')
        imp=df['u_t']-df['ground_truth_vel']
        nun=int(((df['collision_occurred']==1)&(imp>vacc)).sum())
        res=dict(label=lab, n=len(df), unacc=nun)
        for p in (RSS,JRC,PCB):
            c=confusion(df,p,vacc); res[p.name]=(c['FN'],c['FP'])
        m=re.search(r'\d+\.?\d*', lab); res['_k']=float(m.group()) if m else 0.0
        rows.append(res)
    rows.sort(key=lambda r:r['_k'])
    print(f"\n{'sigma / file':22}{'n':>4}{'unacc':>7}   {'RSS FN/FP':>12}{'JRC FN/FP':>12}{'PCB FN/FP':>12}")
    print('-'*74)
    for r in rows:
        cells=''.join(f"{f'{r[m][0]}/{r[m][1]}':>12}" for m in ('RSS','JRC','PCB'))
        print(f"{r['label']:22}{r['n']:>4}{r['unacc']:>7}   {cells}")
    print("\nFN = unacceptable collisions beyond the boundary, i.e. in the safe zone (soundness, want 0)   FP = safe states flagged (conservativeness)")

def region_breakdown(df, p, vacc=VACC):
    """Per-region counts for one envelope. Rows land in the model's three regions
    (safe / unsafe / collision-zone); columns split by outcome. 'unacc' in the SAFE
    row = unacceptable collisions beyond the boundary (identical to FN by construction)."""
    cnt={r:{'total':0,'unacc':0,'minor':0,'none':0} for r in ('safe','unsafe','collision')}
    for _,r in df.iterrows():
        d=r.to_dict()
        if pd.isna(d.get('collision_occurred')): continue
        reg=classify(d,p)
        coll=int(d['collision_occurred'])==1
        u,gv=g(d,'u_t'),g(d,'ground_truth_vel')
        unacc = coll and (u is not None and gv is not None and (u-gv)>vacc)
        cnt[reg]['total']+=1
        cnt[reg]['unacc' if unacc else ('minor' if coll else 'none')]+=1
    return cnt

def print_breakdown(df, models=None, vacc=VACC):
    models=models or (RSS,JRC,PCB)
    print("\n=== region breakdown (rows per zone; 'unacc' in SAFE row = collisions beyond the boundary) ===")
    for p in models:
        c=region_breakdown(df,p,vacc)
        print(f"\n{p.name}")
        print(f"  {'region':>10}{'total':>9}{'unacc-coll':>12}{'minor-coll':>12}{'no-coll':>9}")
        for reg in ('safe','unsafe','collision'):
            r=c[reg]; print(f"  {reg:>10}{r['total']:>9}{r['unacc']:>12}{r['minor']:>12}{r['none']:>9}")
        print(f"  -> unacceptable collisions beyond the {p.name} boundary: {c['safe']['unacc']}")

if __name__=='__main__':
    import sys
    args=sys.argv[1:]
    if '--jrc-sim' in args:
        args.remove('--jrc-sim'); JRC_CUTIN_SIM=True
        print("[exact JRC cut-in mode: closed-loop R157 CC driver]")
    if len(args)>1:
        sweep(args)
    elif args:
        df=pd.read_csv(args[0])
        imp=df['u_t']-df['ground_truth_vel']
        n_un=int(((df['collision_occurred']==1)&(imp>VACC)).sum())
        print(f"\n{len(df)} rows | collisions {int(df['collision_occurred'].sum())} | "
              f"unacceptable (>{VACC*3.6:.0f} km/h) {n_un}\n")
        print(f"{'model':6}{'TP':>9}{'FP':>9}{'TN':>9}{'FN':>9}   {'soundness (FN)':>16}{'conserv. (FP)':>15}")
        for p in (RSS,JRC,PCB):
            c=confusion(df,p)
            s='OK (0)' if c['FN']==0 else f"{c['FN']} MISS"
            print(f"{p.name:6}{c['TP']:>9}{c['FP']:>9}{c['TN']:>9}{c['FN']:>9}   {s:>16}{str(c['FP'])+' flg':>15}")
        print_breakdown(df)
    else:
        print("usage: compare_boundaries.py FILE.csv        (single table)")
        print("       compare_boundaries.py sig*.csv        (per-sigma sweep, one file per sigma)")