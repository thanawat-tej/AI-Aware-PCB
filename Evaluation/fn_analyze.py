#!/usr/bin/env python3
"""Diagnose PCB false negatives -- unacceptable collisions (impact > v_acc) that
PCB predicted SAFE. Measures the perception error on every channel present in
corrected.csv against PCB's budget to separate NOISE misses (a bound exceeded)
from SYSTEMATIC misses (all errors within budget -> response/trigger timing).
Checks the LATERAL position channel when ground_truth_d_y is present (add it in
map_to_corrected from true_lat_gap). Usage: python3 fn_analyze.py sig0.15.csv ..."""
import sys, pandas as pd, numpy as np
from compare_boundaries import PCB, RSS, JRC, classify, VACC
ENVS = {'PCB': PCB, 'RSS': RSS, 'JRC': JRC}
# NOTE: the budgets below are PCB's. For RSS/JRC (which carry NO perception margin)
# the noise-vs-budget split is not meaningful -- they have no budget to exceed -- so
# for those models we report the SCENARIO breakdown and the measured errors, and label
# the cause column 'no-margin' to make that explicit rather than pretending otherwise.
EPS, GAM, EPSY, GAMY = PCB.EPS_MAX, PCB.GAMMA_MAX, PCB.EPS_Y_MAX, PCB.GAMMA_Y_MAX
BUD = {'long_pos': EPS, 'long_spd': GAM, 'lat_pos': EPSY, 'lat_spd': GAMY}
CHAN = ['long_pos','long_spd','lat_pos','lat_spd']

def _f(v): return None if v is None or (isinstance(v,float) and pd.isna(v)) else float(v)
def _sub(d,a,b):
    x,y=_f(d.get(a)),_f(d.get(b)); return None if (x is None or y is None) else (x-y)
def _errs(d):
    s=d.get('scenario'); e={}
    if s=='deceleration':
        e['long_pos']=_sub(d,'p_t','ground_truth_dist');     e['long_spd']=_sub(d,'w_t','ground_truth_vel')
    elif s=='cutin':
        e['long_pos']=_sub(d,'p_long_t','ground_truth_dist'); e['long_spd']=_sub(d,'w_cut','ground_truth_vel')
        e['lat_pos']=_sub(d,'d_y','ground_truth_d_y');        e['lat_spd']=_sub(d,'v_y_rel','ground_truth_v_y_rel')
    elif s=='cutout':
        e['long_pos']=_sub(d,'p_rev','ground_truth_dist');   e['long_spd']=_sub(d,'v_rev_hat','ground_truth_vel')
        e['lat_spd']=_sub(d,'v_y_occ','ground_truth_v_y_occ')
    return e

def analyze(path, env=PCB):
    df=pd.read_csv(path); fn=[]; has_latpos=False
    for _,r in df.iterrows():
        d=r.to_dict()
        if int(d.get('collision_occurred',0) or 0)!=1: continue
        u,gv=_f(d.get('u_t')),_f(d.get('ground_truth_vel'))
        if u is None or gv is None or (u-gv)<=VACC: continue
        if classify(d,env)!='safe': continue
        e=_errs(d)
        if e.get('lat_pos') is not None: has_latpos=True
        over=[k for k,v in e.items() if v is not None and v>BUD[k]]
        rec=dict(scenario=d.get('scenario'), impact_kmh=round((u-gv)*3.6,1))
        for k in CHAN: rec[k]=None if e.get(k) is None else round(e[k],2)
        rec['cause']=('noise>budget('+','.join(over)+')') if over else 'within-budget(systematic)'
        fn.append(rec)
    f=pd.DataFrame(fn)
    print(f"\n=== {path}: {len(f)} {env.name} false negatives ===")
    if len(f)==0: print("(none)"); return
    print("\nFN by scenario:\n"+f.groupby('scenario').size().to_string())
    sysn=int(f['cause'].str.startswith('within-budget').sum())
    print(f"\ncause: {len(f)-sysn} noise>budget | {sysn} within-budget ({sysn/len(f)*100:.0f}% systematic)")
    print(f"\nsigned error (dangerous>0; eps={EPS} gam={GAM} eps_y={EPSY} gam_y={GAMY}):")
    print(f[CHAN].describe().round(2).to_string())
    if not has_latpos and (f['scenario'].isin(['cutin'])).any():
        print("\n[!] lat_pos is empty -- add ground_truth_d_y to map_to_corrected (from true_lat_gap) so the "
              "LATERAL-position channel can be checked. Cut-in danger is chiefly lateral, so this is the one to add.")
    else:
        lp=f['lat_pos'].dropna()
        if len(lp):
            share=(lp>EPSY).mean()*100
            print(f"\n>> lateral-position error exceeds eps_y in {share:.0f}% of misses "
                  f"({'LATERAL PERCEPTION is the driver' if share>50 else 'lateral perception is NOT the main driver -> likely a cut-in classifier/timing gap'})")
    print("\nsample:\n"+f.head(10).to_string(index=False))

if __name__=='__main__':
    args=[a for a in sys.argv[1:]]
    model='PCB'
    if '--model' in args:
        i=args.index('--model'); model=args[i+1].upper(); del args[i:i+2]
    env=ENVS[model]
    for p in (args or ['/tmp/sig1.00.csv']): analyze(p, env)