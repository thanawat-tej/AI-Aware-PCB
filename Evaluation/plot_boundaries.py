#!/usr/bin/env python3
"""Compare RSS / JRC / PCB in a 2-D plane, one panel per model, shared axes.
The boundaries depend on EGO and LEAD speed, so filter to a speed band first
(--umin/--umax on ego, --wmin/--wmax on lead); the dashed boundary is then computed
at the band's center and is valid for the points shown. Points: RED = unacceptable
collision (impact>v_acc), GREEN = otherwise. BLACK RING = that model's false negatives.
Run --gap truth vs --gap perceived to show perception explaining the safe-zone collisions.

  # cut-in, ego 15-16 m/s, lead 9-11 m/s:
  python3 plot_boundaries.py corrected.csv --scenario cutin --umin 15 --umax 16 --wmin 9 --wmax 11
  # deceleration, ego 15-17 m/s:
  python3 plot_boundaries.py corrected.csv --scenario deceleration --umin 15 --umax 17
"""
import sys, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from compare_boundaries import RSS, JRC, PCB, classify, cls_cutin, cls_decel, VACC

def cutin_boundary(p, u, w, dy, yvals, xgrid):
    out=[]
    for vy in yvals:
        b=np.nan
        for gp in xgrid:
            if cls_cutin(gp, vy, u, w, dy, p)=='safe': b=gp; break
        out.append(b)
    return out

def decel_boundary(p, u, yvals, xgrid):   # yvals = closing speed; lead speed w = u - closing
    out=[]
    for clo in yvals:
        w=max(0.0, u-clo); b=np.nan
        for gp in xgrid:
            if cls_decel(gp, u, w, p)=='safe': b=gp; break
        out.append(b)
    return out
# note: for deceleration the y-axis (closing speed) already fixes w given u, so the
# envelope corners only vary u; cutin varies both u and w independently.

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('csv'); ap.add_argument('--scenario', default='cutin')
    ap.add_argument('--gap', choices=['perceived','truth'], default='perceived')
    ap.add_argument('--umin', type=float, default=None); ap.add_argument('--umax', type=float, default=None)
    ap.add_argument('--wmin', type=float, default=None); ap.add_argument('--wmax', type=float, default=None)
    ap.add_argument('--dy', type=float, default=None, help='lateral gap for the cut-in boundary (default 3.5)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--jrc-frontier', action='store_true',
                    help='cutin only: overlay the exact closed-loop CC-driver crash frontier (jrc_sim) on the JRC panel')
    ap.add_argument('--all-pairs', action='store_true',
                    help='auto-bin ego/lead speeds and render one PNG per speed pair')
    ap.add_argument('--bin', type=float, default=2.0, help='speed bin width in m/s (default 2)')
    ap.add_argument('--min-rows', type=int, default=30, help='skip pairs with fewer rows (default 30)')
    ap.add_argument('--outdir', default='pair_plots')
    a=ap.parse_args()
    full=pd.read_csv(a.csv); full=full[full['scenario']==a.scenario].copy()
    wcol='w_cut' if a.scenario=='cutin' else 'w_t' if a.scenario=='deceleration' else 'v_rev_hat'
    if a.all_pairs:
        import os
        os.makedirs(a.outdir, exist_ok=True)
        full=full.dropna(subset=['u_t',wcol])
        ub=(full['u_t']//a.bin)*a.bin; wb=(full[wcol]//a.bin)*a.bin
        pairs=sorted(set(zip(ub,wb)))
        made=0
        for u0,w0 in pairs:
            sub=full[(ub==u0)&(wb==w0)]
            if len(sub)<a.min_rows: continue
            name=f"{a.outdir}/{a.scenario}_{a.gap}_u{u0:g}-{u0+a.bin:g}_w{w0:g}-{w0+a.bin:g}.png"
            render(sub.copy(), a, wcol, name); made+=1
        print(f"{made} plots -> {a.outdir}/ (bins of {a.bin} m/s, min {a.min_rows} rows)"); return
    df=full
    if a.umin is not None: df=df[df['u_t']>=a.umin]
    if a.umax is not None: df=df[df['u_t']<=a.umax]
    if a.wmin is not None: df=df[df[wcol]>=a.wmin]
    if a.wmax is not None: df=df[df[wcol]<=a.wmax]
    if df.empty: sys.exit("no rows after filtering")
    render(df, a, wcol, a.out or f"boundaries_{a.scenario}_{a.gap}.png")

def render(df, a, wcol, out):
    u_c=float(df['u_t'].median()); w_c=float(df[wcol].median())
    print(f"{len(df)} rows in band | ego~{u_c:.1f} m/s, lead~{w_c:.1f} m/s")
    gapcol={'cutin':'p_long_t','deceleration':'p_t','cutout':'p_rev'}[a.scenario]
    df['X']=df[gapcol] if a.gap=='perceived' else df['ground_truth_dist']
    if a.scenario=='cutin': df['Y']=df['v_y_rel']; ylab='cut-in lateral velocity (m/s)'
    elif a.scenario=='deceleration': df['Y']=df['u_t']-df['ground_truth_vel']; ylab='closing speed (m/s)'
    else: df['Y']=df['v_y_occ']; ylab='occluder lateral velocity (m/s)'
    imp=df['u_t']-df['ground_truth_vel']; df['unacc']=(df['collision_occurred']==1)&(imp>VACC)
    xlab=f"{a.gap} longitudinal gap (m)"
    # scan grid extends 60% past the data so worst-case boundaries still resolve
    xgrid=np.linspace(max(0,df['X'].min()), df['X'].max()*1.6+5, 320); yv=np.linspace(df['Y'].min(),df['Y'].max(),60)
    dy_b=a.dy if a.dy is not None else 3.5
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),sharex=True,sharey=True)
    for ax,p in zip(axes,(RSS,JRC,PCB)):
        ax.scatter(df['X'][~df['unacc']], df['Y'][~df['unacc']], s=8, c='#2ca02c', alpha=.30, edgecolors='none', label='no unacc. collision')
        ax.scatter(df['X'][df['unacc']],  df['Y'][df['unacc']],  s=11, c='#d62728', alpha=.6, edgecolors='none', label='unacc. collision')
        def bnd(u,w):
            if a.scenario=='cutin':        return cutin_boundary(p,u,w,dy_b,yv,xgrid)
            if a.scenario=='deceleration': return decel_boundary(p,u,yv,xgrid)
            return None
        if a.scenario=='cutin' and p.name=='JRC':
            # closed-form CC crash band (valid at any speed pair), drawn instead of
            # the stopping-distance approximation. x-axis is center-to-center, the
            # band is derived in edge gap, so add the summed half lengths (4.3 m).
            import jrc_sim
            los,his=[],[]
            for vy_ in yv:
                lo,hi=jrc_sim.cc_cutin_band(u_c,w_c,max(vy_,1e-3),dy_b)
                los.append(lo+4.3); his.append((hi+4.3) if np.isfinite(hi) else np.nan)
            ax.plot(los,yv,'k--',lw=1.4)
            ax.plot(his,yv,'k--',lw=1.4)
            ax.plot([],[],'k--',lw=1.4,label='CC crash band (closed form)')
            xb_med=None; n_safe_coll=None
            b_lo=np.interp(df['Y'],yv,np.array(los)); b_hi=np.interp(df['Y'],yv,np.array(his))
            # for this panel the safe-zone count is collisions beyond the FAR band
            # edge (the merged-behind near side is also CC-safe but rarely populated)
            n_safe_coll=int((df['unacc']&(df['X']>=b_hi)).sum()) if np.isfinite(b_hi).any() else 0
        else:
            xb_med = bnd(u_c,w_c)
        n_safe_coll = None
        if xb_med is not None:
            b_at = np.interp(df['Y'], yv, np.array(xb_med,dtype=float))
            n_safe_coll = int((df['unacc'] & (df['X'] >= b_at)).sum())
            # boundary is monotone in the speeds: worst case = fastest ego + slowest lead,
            # best case = slowest ego + fastest lead -> envelope brackets the whole band
            u_lo,u_hi = float(df['u_t'].min()), float(df['u_t'].max())
            w_lo,w_hi = float(df[wcol].min()), float(df[wcol].max())
            xb_worst, xb_best = bnd(u_hi,w_lo), bnd(u_lo,w_hi)
            ax.fill_betweenx(yv, xb_best, xb_worst, color='k', alpha=.06, zorder=0)
            ax.plot(xb_best,  yv, 'k:',  lw=0.9, alpha=.55)
            ax.plot(xb_med,   yv, 'k--', lw=1.3, alpha=.75)
            ax.plot(xb_worst, yv, 'k-',  lw=0.9, alpha=.55)
            # if the worst-case line exits the visible range, say where it is
            x_right=df['X'].max()+2
            wc=np.array(xb_worst,dtype=float)
            off=np.isfinite(wc)&(wc>x_right)
            if off.any():
                y_at=float(np.array(yv)[off].mean())
                ax.annotate(f'worst-case ≈ {np.nanmax(wc):.0f} m →', xy=(x_right, y_at),
                            xytext=(-4,0), textcoords='offset points',
                            ha='right', va='center', fontsize=7, alpha=.75)
        if a.scenario=='cutin' and getattr(a,'jrc_frontier',False) and p.name=='JRC':
            import jrc_sim
            gx=np.linspace(max(0.5,df['X'].min()), df['X'].max()+2, 40)
            gy=np.linspace(max(0.05,df['Y'].min()), df['Y'].max(), 36)
            CR=np.zeros((len(gy),len(gx)))
            for ai,vy_ in enumerate(gy):
                for bi,gp in enumerate(gx):
                    CR[ai,bi]=jrc_sim.sim_cut_in('CC_human_driver', u_c, w_c, vy_, max(0.0,gp-4.3), init_lat_c2c=dy_b)
            if CR.any() and not CR.all():
                ax.contour(gx, gy, CR, levels=[0.5], colors='k', linewidths=2.0)
                ax.plot([], [], color='k', lw=2.0, label='CC crash frontier (exact sim)')
        title = p.name if n_safe_coll is None else f"{p.name}  (collisions in safe zone {n_safe_coll})"
        ax.set_title(title); ax.set_xlabel(xlab); ax.legend(fontsize=7, markerscale=1.4, loc='upper right')
    axes[0].set_ylabel(ylab)
    axes[0].set_xlim(max(0,df['X'].min())-1, df['X'].max()+2)
    band=f"ego {u_c:.1f}, lead {w_c:.1f} m/s" + (f", d_y={dy_b:.1f}" if a.scenario=='cutin' else "")
    fig.suptitle(f"{a.scenario} — {a.gap} gap — boundary at [{band}] — red=unacc. collision; dotted/dashed/solid = best/median/worst-case boundary in band")
    fig.tight_layout(); fig.savefig(out,dpi=130); plt.close(fig); print("wrote",out)

if __name__=='__main__': main()