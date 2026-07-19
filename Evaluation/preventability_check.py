#!/usr/bin/env python3
"""Preventability check for the cut-in FN. Your comparison runs four ego models
on the SAME cut-in grid: CC_human_driver reacts LATE (overlap + TTC<=2s), while
RSS / FSM / Reg157 react EARLY (safe-distance / anticipation) -- i.e. the prompt
ego PCB assumes. For every cut-in cell where the CC driver had an UNACCEPTABLE
crash (impact > v_acc), this reports whether the prompt models avoided it:
  avoided by the prompt models  -> PREVENTABLE by a prompt ego -> the FN are the
                                   CC driver's late trigger, PCB is sound   (case a)
  the prompt models also crashed -> UNPREVENTABLE even prompt -> PCB's cut-in
                                   projection is optimistic there            (case b)
Usage: python3 preventability_check.py <comparison_results_dir_for_cut_in>
"""
import sys, os, glob
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
VACC = 10/3.6
CC = 'CC_human_driver'
PROMPT = ['RSS','FSM','Reg157']

def load(dirpath):
    rows=[]
    for f in glob.glob(os.path.join(dirpath,'**','*.csv'), recursive=True):
        stem=os.path.basename(f)[:-4]
        parts=stem.rsplit('_',2)                      # CC_human_driver_70_40 -> (model,ego,cutin)
        if len(parts)!=3: continue
        model,ego,cutin=parts
        try: ego,cutin=float(ego),float(cutin)
        except ValueError: continue
        try: d=pd.read_csv(f)
        except Exception: continue
        if 'Crash' not in d.columns: continue
        for _,r in d.iterrows():
            u=r.get('ego_speed');  u=float(u) if pd.notna(u) else (float(r['velocity'])/3.6 if pd.notna(r.get('velocity')) else np.nan)
            ts=r.get('true_speed'); ts=float(ts) if pd.notna(ts) else np.nan
            rows.append(dict(model=model, ego=ego, cutin=cutin,
                             long_dist=r.get('long_dist'), lat_vel=r.get('lat_vel'),
                             crash=int(bool(r.get('Crash'))),
                             impact=(u-ts) if not (np.isnan(u) or np.isnan(ts)) else np.nan))
    return pd.DataFrame(rows)

def main(dirpath):
    df=load(dirpath)
    if df.empty: sys.exit("No cut-in comparison CSVs (with a 'Crash' column) found under "+dirpath)
    cells={}
    for _,r in df.iterrows():
        c=cells.setdefault((r['ego'],r['cutin'],r['long_dist'],r['lat_vel']), {})
        c[r['model']]=r['crash']
        if r['model']==CC: c['imp']=r['impact']
    bad=[k for k,c in cells.items() if c.get(CC)==1 and (c.get('imp') or 0)>VACC]
    print(f"\nmodels present: {sorted({m for c in cells.values() for m in c if m not in ('imp',)})}")
    print(f"CC-driver UNACCEPTABLE-crash cut-in cells (impact > {VACC*3.6:.0f} km/h): {len(bad)}\n")
    if not bad: print("(none found -- check the directory / column names)"); return
    for m in PROMPT:
        present=[k for k in bad if m in cells[k]]
        if not present: print(f"  vs {m:7}: model not in output"); continue
        avoided=sum(1 for k in present if cells[k][m]==0)
        also   =sum(1 for k in present if cells[k][m]==1)
        verdict=('mostly PREVENTABLE -> (a) PCB sound, CC late trigger'
                 if avoided>=also else 'many UNPREVENTABLE -> (b) PCB cut-in projection optimistic')
        print(f"  vs {m:7}: avoided {avoided:4d} | also crashed {also:4d}   -> {verdict}")
    print("\nIf the prompt models avoid most of these, the 123 FN are the CC driver reacting late in cut-ins,")
    print("not a PCB flaw -- exactly your 'an FN is the SUT falling short' framing, now backed by the data.")

if __name__=='__main__':
    main(sys.argv[1] if len(sys.argv)>1 else 'results')