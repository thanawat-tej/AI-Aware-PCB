#!/usr/bin/env python3
"""Map JRC-FSM comparison-run CSVs into the `corrected.csv` schema.

The comparison runs (safety_check_runner.py comparison ...) write per-cell CSVs
under results/<scenario>[ _noise_pX_sY ]/ with perception columns
(perceived_*/true_*, ego_speed, reacted, ...). pcb_analysis.py instead expects a
single CSV in the `corrected.csv` layout, one row per scenario point. This
script reads the JRC result folders and emits that layout so the simulated data
can be fed straight into pcb_analysis.py.

Usage:
    python3 map_to_corrected.py [results_dir] [-o OUT.csv]

    results_dir   folder holding the run sub-folders (default: results)
    -o            output path (default: mapped_corrected.csv)

Folder -> scenario:
    car_following* -> deceleration     cut_in*  -> cutin     cut_out* -> cutout
A '_noise_' in the folder name marks a noisy run (inputs_were_groundtruth=0);
otherwise the run is perfect-perception (inputs_were_groundtruth=1).

Notes / conventions:
  * Snapshots are taken at CLOSEST APPROACH (smallest true longitudinal gap),
    so every row is populated.
  * JRC stores gaps as EDGE-TO-EDGE clearances; corrected.csv documents the
    longitudinal/lateral gaps as CENTER-TO-CENTER, so we add back the vehicle
    extent (LON_C2C / LAT_C2C). Set CONVERT_TO_CENTER=False to keep edge gaps.
  * Columns pcb_analysis.py does not consume (ego_max_accel, ego_accel_prebrake)
    are left blank. Perception-quality columns reflect the additive-noise model:
    the object is always seen, so coverage=1, gaps=0, staleness=0.
"""

import os
import sys
import argparse
import math
import pandas as pd

# Cut-in danger is a moving target across the episode: the closest-approach
# (min-gap) frame captures the longitudinal threat once the ego has caught up,
# while the lane-entry frame captures the active lateral cut-in. Neither frame
# alone is sufficient, so for cut-in we classify BOTH candidate frames and emit
# the more dangerous one (worst-case). Needs pcb_analysis.classify_cutin; if it
# can't be imported we fall back to the lane-entry frame.
WORST_CASE_CUTIN = True
try:
    sys.path.insert(0, '/home/nimda/visual')
    import pcb_analysis as _pcb
    _SEVERITY = {'safe': 0, 'unsafe': 1, 'collision': 2}
except Exception:
    _pcb = None

# --- geometry / config (mirrors utility/global_parameters.py) ----------------
LENGTH = 4.3            # vehicle length [m]
WIDTH = 1.9             # vehicle width [m]
G = 9.81

CONVERT_TO_CENTER = True
LON_C2C = LENGTH if CONVERT_TO_CENTER else 0.0   # half_len(ego)+half_len(npc)
LAT_C2C = WIDTH if CONVERT_TO_CENTER else 0.0    # half_wid(ego)+half_wid(npc)

# cut-out occlusion fields the JRC sim does not model -> documented defaults
# (corrected.csv uses a fixed lateral-clearance constant and a capped blackout).
D_Y_OCC_DEFAULT = 1.9
T_BLACKOUT_DEFAULT = 0.0

# exact corrected.csv column order
COLUMNS = [
    'scenario', 'p_t', 'u_t', 'w_t', 'p_long_t', 'v_y_rel', 'w_cut', 'd_y',
    'p_rev', 'v_y_occ', 'v_rev_hat', 't_blackout', 'd_y_occ',
    'collision_occurred', 'ground_truth_dist', 'ground_truth_vel',
    'ground_truth_v_y_rel', 'ground_truth_d_y','ground_truth_v_y_occ', 'v_y_occ_peak',
    'npc_actual_decel', 'ego_max_accel', 'ego_accel_prebrake', 'filename_param',
    'perception_detected', 'inputs_were_groundtruth', 'perception_coverage',
    'perception_max_gap_s', 'staleness_at_decision_s', 'label',
]


def _num(v):
    """None for NaN/missing, else float."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def scenario_of(folder):
    name = os.path.basename(folder.rstrip('/'))
    if name.startswith('car_following'):
        return 'deceleration'
    if name.startswith('cut_in'):
        return 'cutin'
    if name.startswith('cut_out'):
        return 'cutout'
    return None


def base_row(scenario, gt_flag):
    """A blank corrected.csv row with shared metadata pre-filled."""
    r = {c: '' for c in COLUMNS}
    r['scenario'] = scenario
    # additive-noise model: object always seen, no dropouts/staleness
    r['perception_detected'] = 1
    r['inputs_were_groundtruth'] = gt_flag
    r['perception_coverage'] = 1.0
    r['perception_max_gap_s'] = 0.0
    r['staleness_at_decision_s'] = 0.0
    return r


def map_deceleration(df, model, gt_flag):
    out = []
    for _, x in df.iterrows():
        u = _num(x.get('ego_speed'))
        if u is None:                       # fallback: initial ego speed (km/h)
            u = _num(x.get('velocity'))
            u = u / 3.6 if u is not None else None
        r = base_row('deceleration', gt_flag)
        r['u_t'] = u
        r['p_t'] = _num(x['perceived_gap']) + LON_C2C
        r['w_t'] = _num(x['perceived_speed'])
        r['collision_occurred'] = int(bool(x['Crash']))
        r['ground_truth_dist'] = _num(x['true_gap']) + LON_C2C
        r['ground_truth_vel'] = _num(x['true_speed'])
        decel = _num(x.get('max_deceleration'))
        r['npc_actual_decel'] = G * decel if decel is not None else ''
        r['filename_param'] = r['npc_actual_decel']
        r['label'] = "decel_%s_v%s_d%s" % (model, x.get('velocity'),
                                           x.get('max_deceleration'))
        out.append(r)
    return out


def _cutin_frame(x, prefix, ego_kmh):
    """Build a cut-in feature dict from one snapshot frame ('' = closest
    approach, 'entry_' = lane entry). Returns None if the frame is absent."""
    g = _num(x.get(prefix + 'perceived_gap'))
    if g is None:
        return None
    u = _num(x.get(prefix + 'ego_speed'))
    if u is None and ego_kmh is not None:
        u = ego_kmh / 3.6
    return {
        'u_t': u,
        'p_long_t': g + LON_C2C,
        'v_y_rel': abs(_num(x.get(prefix + 'perceived_lat_speed'))),
        'w_cut': _num(x.get(prefix + 'perceived_speed')),
        'd_y': _num(x.get(prefix + 'perceived_lat_gap')) + LAT_C2C,
        'ground_truth_d_y': (_num(x.get(prefix + 'true_lat_gap')) + LAT_C2C
                             if _num(x.get(prefix + 'true_lat_gap')) is not None else ''),
        'ground_truth_dist': _num(x.get(prefix + 'true_gap')) + LON_C2C,
        'ground_truth_vel': _num(x.get(prefix + 'true_speed')),
        'ground_truth_v_y_rel': abs(_num(x.get(prefix + 'true_lat_speed'))),
    }


def _cutin_severity(f):
    return _SEVERITY[_pcb.classify_cutin(
        f['p_long_t'], f['v_y_rel'], f['u_t'], f['w_cut'], f['d_y'])]


def map_cutin(df, model, ego_kmh, cutin_kmh, gt_flag):
    out = []
    for _, x in df.iterrows():
        closest = _cutin_frame(x, '', ego_kmh)
        entry = _cutin_frame(x, 'entry_', ego_kmh)
        # Worst-case: keep whichever candidate frame the classifier rates more
        # dangerous (entry preferred on ties as the cut-in-defining instant).
        if entry is None:
            chosen = closest
        elif WORST_CASE_CUTIN and _pcb is not None:
            chosen = entry if _cutin_severity(entry) >= _cutin_severity(closest) \
                else closest
        else:
            chosen = entry
        r = base_row('cutin', gt_flag)
        r.update(chosen)
        r['collision_occurred'] = int(bool(x['Crash']))
        r['filename_param'] = _num(x.get('lat_vel'))
        r['label'] = "cutin_%s_e%s_c%s_d%s_l%s" % (
            model, ego_kmh, cutin_kmh, x.get('long_dist'), x.get('lat_vel'))
        out.append(r)
    return out


def map_cutout(df, model, gt_flag):
    out = []
    for _, x in df.iterrows():
        u = _num(x.get('ego_speed'))
        if u is None:
            u = _num(x.get('velocity'))
            u = u / 3.6 if u is not None else None
        # occluder lateral exit speed comes from the scenario parameter (the
        # closest-approach snapshot tracks the revealed/stopped vehicle, not the
        # occluder), so v_y_occ is read from the cell's lateral velocity.
        v_y_occ = abs(_num(x['lateral_velocities']))
        r = base_row('cutout', gt_flag)
        r['u_t'] = u
        r['p_rev'] = _num(x['perceived_gap']) + LON_C2C
        r['v_y_occ'] = v_y_occ
        r['v_rev_hat'] = _num(x['perceived_speed'])
        r['t_blackout'] = T_BLACKOUT_DEFAULT
        r['d_y_occ'] = D_Y_OCC_DEFAULT
        r['collision_occurred'] = int(bool(x['Crash']))
        r['ground_truth_dist'] = _num(x['true_gap']) + LON_C2C
        r['ground_truth_vel'] = _num(x['true_speed'])
        r['ground_truth_v_y_occ'] = v_y_occ
        r['v_y_occ_peak'] = v_y_occ
        r['filename_param'] = v_y_occ
        r['label'] = "cutout_%s_v%s_f%s_l%s" % (
            model, x.get('velocity'), x.get('front_distance'),
            x.get('lateral_velocities'))
        out.append(r)
    return out


def parse_cutin_name(fname):
    """'CC_human_driver_70_40.csv' -> (model, ego_kmh, cutin_kmh)."""
    stem = fname[:-4] if fname.endswith('.csv') else fname
    parts = stem.rsplit('_', 2)
    if len(parts) == 3:
        model, ego, cutin = parts
        return model, _num(ego), _num(cutin)
    return stem, None, None


# source columns each scenario mapper needs; files missing any are skipped
# (e.g. legacy result folders written before the perception columns existed).
REQUIRED_SRC = {
    'deceleration': ['perceived_gap', 'true_gap', 'perceived_speed',
                     'true_speed', 'Crash'],
    'cutin': ['perceived_gap', 'true_gap', 'perceived_lat_gap',
              'perceived_lat_speed', 'true_lat_speed', 'perceived_speed',
              'true_speed', 'Crash', 'entry_perceived_gap'],
    'cutout': ['perceived_gap', 'true_gap', 'perceived_speed', 'true_speed',
               'lateral_velocities', 'Crash'],
}


def run_tag(folder):
    """Short id for the run's noise config, taken from the folder name, e.g.
    'p0.5_s1' for results/cut_in_high_speed_noise_p0.5_s1, or 'clean'. Used to
    namespace labels so different noise configs (or clean vs noisy) never share
    a label -- otherwise the same scenario cell collides across configs."""
    name = os.path.basename(folder.rstrip('/'))
    return name.split('_noise_', 1)[1] if '_noise_' in name else 'clean'


def process_folder(folder):
    scen = scenario_of(folder)
    if scen is None:
        return []
    gt_flag = 0 if '_noise_' in os.path.basename(folder) else 1
    tag = run_tag(folder)
    rows = []
    skipped = 0
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(folder, fname))
        if any(c not in df.columns for c in REQUIRED_SRC[scen]):
            skipped += 1   # legacy file without perception columns
            continue
        if scen == 'deceleration':
            model = fname[:-len('_car_following.csv')]
            new = map_deceleration(df, model, gt_flag)
        elif scen == 'cutout':
            model = fname[:-len('_cut_out.csv')]
            new = map_cutout(df, model, gt_flag)
        elif scen == 'cutin':
            model, ego, cutin = parse_cutin_name(fname)
            new = map_cutin(df, model, ego, cutin, gt_flag)
        for r in new:                       # namespace by run config
            r['label'] = tag + '|' + r['label']
        rows += new
    if skipped:
        print("    (skipped %d legacy file(s) without perception columns)"
              % skipped)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results_dir', nargs='?', default='results')
    ap.add_argument('-o', '--out', default='mapped_corrected.csv')
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        sys.exit("No such directory: %s" % args.results_dir)

    all_rows = []
    folders = []
    for name in sorted(os.listdir(args.results_dir)):
        path = os.path.join(args.results_dir, name)
        if os.path.isdir(path) and scenario_of(path):
            folders.append(path)
            n = process_folder(path)
            all_rows += n
            print("  %-45s %s  (%d rows)" % (name, scenario_of(path), len(n)))

    if not all_rows:
        sys.exit("No mappable scenario folders found under %s" % args.results_dir)

    out = pd.DataFrame(all_rows, columns=COLUMNS)
    out.to_csv(args.out, index=False)
    print("\nWrote %d rows from %d folders -> %s" %
          (len(out), len(folders), args.out))
    print("scenario counts:\n", out['scenario'].value_counts().to_string())


if __name__ == '__main__':
    main()