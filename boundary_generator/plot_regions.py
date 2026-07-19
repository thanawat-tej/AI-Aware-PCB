#!/usr/bin/env python3
"""plot_regions.py -- methodology figures: the three-region PCB boundary with the
RSS and JRC boundaries overlaid, for all three scenario families.

Imports the SHIPPED classifiers from compare_boundaries.py (and jrc_sim.py for the
exact CC cut-in band), so every figure is guaranteed to match the numbers in the
results chapter. No boundary math is duplicated here.

Regions (PCB): green = safe, yellow = unsafe but recoverable, red = unavoidable.
Overlays: the RSS safe/unsafe boundary (blue) and the JRC one (purple), each from
its own model. Both baselines are perfect-perception, so they carry no margins.

AXIS CONVENTION. This tool plots the CLASSIFIERS, not data -- it sweeps a gap value
and asks each envelope for its verdict. The quantities the classifiers consume are
PERCEIVED ones: PCB compares the gap against d_req = eps_max + s_ego(u) - ..., and
that eps_max margin exists precisely because the gap it is given may be wrong. So
the x-axis is the PERCEIVED longitudinal gap, i.e. the boundary as the ego applies
it to the state it believes it is in. This is the deployed reading, since no ego
ever has ground truth.

The paired truth/perceived comparison lives in plot_boundaries.py, which plots real
DATA rows and can put either the true or the perceived gap on the axis. Here there
are no rows, so there is nothing to displace and only the perceived reading exists.

  python3 plot_regions.py                       # all three families, default speeds
  python3 plot_regions.py --scenario cutin --u 16 --w 10
  python3 plot_regions.py --no-baselines        # PCB regions only
"""
import argparse, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from compare_boundaries import (PCB, RSS, JRC, cls_decel, cls_cutin, cls_cutout,
                                rss_safe_distance, rss_safe_distance_lat, W_LAT)

RANK = {'safe': 2, 'unsafe': 1, 'collision': 0}
CMAP = ListedColormap(['#e8534f', '#f2c14e', '#7fbf7b'])   # collision, unsafe, safe
NX, NY = 260, 200

def _boundary_x(cls, yvals, xgrid, want=2):
    """First gap on the x-axis at which the model's region reaches `want` (2=safe)."""
    out = []
    for y in yvals:
        b = np.nan
        for gp in xgrid:
            if RANK[cls(float(gp), float(y))] >= want:
                b = gp; break
        out.append(b)
    return out

def panel(ax, scen, u, w, dy, show_base, yaxis='lead'):
    if scen == 'deceleration':
        if yaxis == 'ego':                       # y = EGO speed, lead fixed at --w
            xg = np.linspace(0, 120, NX); yg = np.linspace(2, 25, NY)
            pcb = lambda gp, y: cls_decel(gp, y, w, PCB)
            rss = lambda gp, y: cls_decel(gp, y, w, RSS)
            jrc = lambda gp, y: cls_decel(gp, y, w, JRC)
            ylab = 'ego speed (km/h)'; y_kmh = True
        else:                                    # y = LEAD speed, ego fixed at --u (default)
            xg = np.linspace(0, 90, NX); yg = np.linspace(0, 20, NY)
            pcb = lambda gp, y: cls_decel(gp, u, y, PCB)
            rss = lambda gp, y: cls_decel(gp, u, y, RSS)
            jrc = lambda gp, y: cls_decel(gp, u, y, JRC)
            ylab = 'lead speed (km/h)'; y_kmh = True
    elif scen == 'cutin':
        xg = np.linspace(0, 90, NX); yg = np.linspace(0.05, 2.0, NY)  # y = lateral speed
        pcb = lambda gp, y: cls_cutin(gp, y, u, w, dy, PCB)
        rss = lambda gp, y: cls_cutin(gp, y, u, w, dy, RSS)
        jrc = lambda gp, y: cls_cutin(gp, y, u, w, dy, JRC)
        ylab = 'cut-in lateral speed (m/s)'; y_kmh = False   # lateral: m/s is the meaningful unit
    else:
        xg = np.linspace(0, 90, NX); yg = np.linspace(0.2, 3.0, NY)   # y = occluder lat speed
        pcb = lambda gp, y: cls_cutout(gp, y, u, w, 0.0, 1.9, PCB)
        rss = lambda gp, y: cls_cutout(gp, y, u, w, 0.0, 1.9, RSS)
        jrc = lambda gp, y: cls_cutout(gp, y, u, w, 0.0, 1.9, JRC)
        ylab = 'occluder lateral speed (m/s)'; y_kmh = False

    Z = np.array([[RANK[pcb(x, y)] for x in xg] for y in yg])
    yplot = yg*3.6 if y_kmh else yg          # display only; classifiers always see SI
    ax.pcolormesh(xg, yplot, Z, cmap=CMAP, vmin=0, vmax=2, shading='auto', alpha=.85)

    # PCB's own two boundaries, drawn explicitly
    ax.plot(_boundary_x(pcb, yg, xg, want=2), yplot, 'k-',  lw=2.0, label='PCB  $d_{safe}$')
    ax.plot(_boundary_x(pcb, yg, xg, want=1), yplot, 'k--', lw=1.4, label='PCB  $d_{coll}$')

    if show_base:
        ax.plot(_boundary_x(rss, yg, xg, want=2), yplot, color='#1f77b4', lw=2.0, ls='-.',
                label='RSS (own formula)')
        # ax.plot(_boundary_x(jrc, yg, xg, want=2), yplot, color='#7B3FA0', lw=2.0, ls=':',
        #         label='JRC (CC driver)')

    ax.set_xlabel('perceived longitudinal gap (m)'); ax.set_ylabel(ylab)
    uk, wk = u*3.6, w*3.6
    decel_ttl = (f'Deceleration  (lead {wk:.0f} km/h fixed)' if yaxis=='ego'
                 else f'Deceleration  (ego {uk:.0f} km/h fixed)')
    ttl = {'deceleration': decel_ttl,
           'cutin': (f'Cut-in  (ego speed {uk:.0f} km/h, intruder speed {wk:.0f} km/h, '
                     f'$d_y$={dy:g} m centre-to-centre, edge clearance {dy-W_LAT:.1f} m)'),
           'cutout': f'Cut-out  (ego speed {uk:.0f} km/h, revealed speed {wk:.0f} km/h)'}[scen]
    ax.set_title(ttl, fontsize=11)
    ax.legend(fontsize=8, loc='lower right', framealpha=.92)
    ax.set_xlim(xg[0], xg[-1]); ax.set_ylim(yplot[0], yplot[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=['deceleration','cutin','cutout','all'], default='all')
    ap.add_argument('--u', type=float, default=60.0, help='ego speed (km/h)')
    ap.add_argument('--w', type=float, default=36.0, help='other-vehicle speed (km/h)')
    ap.add_argument('--ms', action='store_true',
                    help='interpret --u/--w as m/s instead of km/h (internals are always SI)')
    ap.add_argument('--dy', type=float, default=3.5, help='cut-in lateral gap, center-to-center (m)')
    ap.add_argument('--yaxis', choices=['lead','ego'], default='lead',
                    help="deceleration only: y-axis is the lead speed (default, recommended) "
                         "or the ego speed (with --w fixing the lead). Lead deceleration is NOT "
                         "an option: the boundary charges the lead a worst-case beta_max by "
                         "construction and never reads its actual rate, so that axis would be flat.")
    ap.add_argument('--no-baselines', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    # Units. Kinematics are SI throughout (m, m/s, m/s^2) and every equation in the
    # thesis is SI. Only the human-facing speed inputs and the LONGITUDINAL speed
    # axes are km/h, since road speeds are read in km/h and the severity policy is
    # already stated as 10 km/h. Lateral speeds stay in m/s: 0.4 m/s of lane drift
    # is meaningful, 1.4 km/h of it is not.
    if not a.ms:
        a.u /= 3.6; a.w /= 3.6
    scens = ['deceleration','cutin','cutout'] if a.scenario=='all' else [a.scenario]
    for sc in scens:
        fig, ax = plt.subplots(figsize=(9, 6))
        panel(ax, sc, a.u, a.w, a.dy, not a.no_baselines, a.yaxis)
        suffix = '_egoaxis' if (sc == 'deceleration' and a.yaxis == 'ego') else ''
        out = a.out or f'pcb_three_region_{sc}{suffix}.png'
        if len(scens) > 1: out = f'pcb_three_region_{sc}{suffix}.png'
        fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
        print('wrote', out)

if __name__ == '__main__':
    main()