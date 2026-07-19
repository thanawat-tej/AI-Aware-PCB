"""
False-Negative Diagnostic Tool
================================
For each scenario where the PCB framework predicted SAFE but a collision
actually occurred, this tool produces a detailed per-scenario report
showing exactly what happened and why the miss happened.

Usage:
    python diagnose_false_negatives.py  <pcb_data_csv>  <json_directory>
                                          [--output  report.txt]

Inputs:
    pcb_data_csv     : CSV produced by pcb_analysis.py (or json_to_csv.py)
                       Rows with collision_occurred=1 and PCB region=safe
                       are diagnosed as false negatives.
    json_directory   : directory containing the original *_data.json files
                       referenced by the 'label' column in the CSV.

The diagnostic for each false negative reports:
  1. SNAPSHOT FRAME    -- what the AV "saw" at the moment classified safe
  2. GROUND TRUTH      -- what was actually happening at the snapshot
  3. COLLISION MOMENT  -- the actual physical contact event
  4. PERCEPTION TIMELINE -- did perception cover the run-up to collision?
  5. ATTRIBUTION       -- which of three hypotheses best explains the miss
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path

# Re-use core PCB classifiers and box utilities from existing tools
from pcb_analysis import (
    classify_deceleration, classify_cutin, classify_cutout,
    EPS_MAX, GAMMA_MAX, BETA_MAX, DELTA_SYS,
)
from json_to_csv import (
    _vel_mag, _xy, _vxy, _box_to_2d_corners,
    project_onto_heading, _vel_relative_to_ego, _pick_perceived_match,
    detect_collision_in_trajectory, compute_perception_stats,
)


# -----------------------------------------------------------------
# Re-use the SAT polygon overlap test
# -----------------------------------------------------------------
def _project_polygon(corners, axis):
    dots = [c[0]*axis[0] + c[1]*axis[1] for c in corners]
    return min(dots), max(dots)


def _polygons_overlap(p1, p2, min_overlap=0.01):
    if len(p1) < 3 or len(p2) < 3: return False, 0.0
    min_depth = float('inf')
    for poly in (p1, p2):
        n = len(poly)
        for i in range(n):
            x1,y1 = poly[i]; x2,y2 = poly[(i+1)%n]
            nx, ny = -(y2-y1), (x2-x1)
            mag = math.hypot(nx, ny)
            if mag < 1e-12: continue
            axis = (nx/mag, ny/mag)
            mn1, mx1 = _project_polygon(p1, axis)
            mn2, mx2 = _project_polygon(p2, axis)
            d = min(mx1, mx2) - max(mn1, mn2)
            if d < min_depth: min_depth = d
            if d < min_overlap:
                return False, d
    return True, min_depth


# -----------------------------------------------------------------
# Per-frame inspection helpers
# -----------------------------------------------------------------
def frame_state(frame, plane):
    """Return a dict summarising the kinematic state of one frame."""
    ego = frame.get('ego') or {}
    npc = frame.get('npc1') or {}
    out = {'t': frame.get('timestamp'),
           'ev': _vel_mag(ego.get('velocity')),
           'nv': _vel_mag(npc.get('velocity')),
           'perc_count': len(frame.get('perception_objects') or []),
           'ego_pos': ego.get('position'),
           'npc_pos': npc.get('position'),
           'has_npc1': npc.get('position') is not None}
    if out['ego_pos'] and out['npc_pos']:
        out['gap_gt'] = math.hypot(*[a - b for a, b in zip(_xy(out['ego_pos'], plane),
                                                             _xy(out['npc_pos'], plane))])
        # Longitudinal gap relative to ego heading
        if ego.get('velocity') and (out['ev'] or 0) > 0.2:
            long_gap, _, _ = project_onto_heading(
                out['ego_pos'], out['npc_pos'], ego['velocity'], plane)
            out['long_gap_gt'] = abs(long_gap)
        else:
            out['long_gap_gt'] = out['gap_gt']
    else:
        out['gap_gt'] = None
        out['long_gap_gt'] = None
    # Perceived equivalents
    perc = _pick_perceived_match(frame, out['npc_pos'], plane) if out['npc_pos'] else None
    if perc and perc.get('position'):
        if ego.get('velocity') and (out['ev'] or 0) > 0.2:
            long_perc, _, _ = project_onto_heading(
                out['ego_pos'], perc['position'], ego['velocity'], plane)
            out['long_gap_perc'] = abs(long_perc)
        else:
            out['long_gap_perc'] = math.hypot(*[a - b for a, b in zip(
                _xy(out['ego_pos'], plane), _xy(perc['position'], plane))])
        out['perc_vel'] = _vel_mag(perc.get('velocity'))
    else:
        out['long_gap_perc'] = None
        out['perc_vel'] = None
    return out


def find_frame_at_time(frames, target_t):
    """Find frame whose timestamp is closest to target_t."""
    best, bd = None, float('inf')
    for i, fr in enumerate(frames):
        d = abs((fr.get('timestamp') or 0) - target_t)
        if d < bd: best, bd = i, d
    return best


def perception_gap_around(frames, center_idx, window_before_s=3.0):
    """
    Look at the `window_before_s` seconds leading up to center_idx; report
    how many of those frames had any perception_objects.

    Returns (n_frames, n_with_perception, total_window_s, longest_gap_s).
    """
    center_t = frames[center_idx].get('timestamp') or 0
    window = []
    for i in range(center_idx, -1, -1):
        t = frames[i].get('timestamp') or 0
        if center_t - t > window_before_s: break
        window.append((t, bool(frames[i].get('perception_objects'))))
    if not window: return 0, 0, 0.0, 0.0
    window = list(reversed(window))
    n = len(window)
    n_with = sum(1 for _, h in window if h)
    total_s = window[-1][0] - window[0][0]
    # Longest gap
    longest_gap = 0.0
    gap_start = None
    for t, h in window:
        if not h:
            if gap_start is None: gap_start = t
            longest_gap = max(longest_gap, t - gap_start)
        else:
            gap_start = None
    return n, n_with, total_s, longest_gap


# -----------------------------------------------------------------
# Cause attribution for one false negative
# -----------------------------------------------------------------
def attribute_cause(snap, ground_truth, coll, window_perception, plane,
                     csv_w_t=None):
    """
    Determine the most likely cause of the miss, with explanation.
    Returns (primary_cause, explanation_lines).

    Parameters
    ----------
    csv_w_t : float or None
        The perceived NPC velocity value AS RECORDED IN THE CSV ROW.
        This is what the classifier actually saw, and is what determines
        whether the safe-pull-away shortcut fires. It can differ from
        the body-frame magnitude that appears in snap['perc_vel'].
    """
    explanations = []
    primary = None

    # Hypothesis A: perception failure either at the snapshot frame
    # or across the 3-second window before the collision
    n, n_with, total_s, longest_gap = window_perception
    coverage_3s = n_with / n if n else 0.0

    snap_undetected = (snap is not None
                       and snap.get('perc_count', 0) == 0)

    if snap_undetected:
        explanations.append(
            f"[A] PERCEPTION FAILURE AT SNAPSHOT: The framework reasoned at "
            f"t={snap['t']:.2f}s but perception reported ZERO objects at that")
        explanations.append(
            f"    frame. The CSV fell back to ground-truth values, so the")
        explanations.append(
            f"    classifier saw 'safe' from perfect data while the real AV")
        explanations.append(
            f"    had no perception input at all.")
        if total_s >= 1.0 and (longest_gap >= 1.5 or coverage_3s < 0.5):
            explanations.append(
                f"    Pre-collision window also degraded: "
                f"{n_with}/{n} frames covered ({100*coverage_3s:.0f}%); "
                f"longest blackout = {longest_gap:.2f} s.")
        primary = primary or 'A'
    elif total_s >= 1.0 and (longest_gap >= 1.5 or coverage_3s < 0.5):
        explanations.append(
            f"[A] PERCEPTION FAILURE IN WINDOW: In the {total_s:.1f} s "
            f"before collision, perception detected the NPC in only "
            f"{n_with}/{n} frames ({100*coverage_3s:.0f}%);")
        explanations.append(
            f"    longest blackout = {longest_gap:.2f} s. The snapshot frame "
            f"itself had perception data, but the surrounding window did not.")
        primary = primary or 'A'

    # Hypothesis B: snapshot frame appears benign even with perception
    if snap and snap['ev'] is not None and snap['nv'] is not None:
        closing = snap['ev'] - (snap['nv'] or 0)
        if abs(closing) < 1.0 and snap['ev'] < 3.0:
            explanations.append(
                f"[B] SLOW-SPEED ACCUMULATION: At the snapshot frame "
                f"(ego_v={snap['ev']:.2f}, npc_v={snap['nv']:.2f}), the dynamics "
                f"look benign (small closing rate {closing:+.2f} m/s,")
            explanations.append(
                "    low speeds). The collision likely accumulated across multiple")
            explanations.append(
                "    near-stationary frames -- any single snapshot would miss it.")
            primary = primary or 'B'

    # Hypothesis C: perception distance over-estimation
    if (snap and snap.get('long_gap_perc') is not None
            and snap.get('long_gap_gt') is not None):
        err = snap['long_gap_perc'] - snap['long_gap_gt']
        if abs(err) > EPS_MAX:
            explanations.append(
                f"[C] PERCEPTION DISTANCE OVER-ESTIMATION: At the snapshot, "
                f"perception reported the gap as {snap['long_gap_perc']:.2f} m "
                f"but the true gap was {snap['long_gap_gt']:.2f} m")
            explanations.append(
                f"    (error = {err:+.2f} m, exceeding the framework's "
                f"EPS_MAX = {EPS_MAX} m budget).")
            primary = primary or 'C'

    # Hypothesis E: perception VELOCITY over-estimation triggering safe shortcut
    # (the framework's `if u_t < w_t: return 'safe'` rule fires when
    # perceived NPC velocity exceeds ego velocity).
    #
    # Use the CSV's w_t value when available -- that is exactly what the
    # classifier saw and is the correct signal for whether the pull-away
    # shortcut fired. Fall back to the body-frame magnitude if csv_w_t is
    # not supplied.
    eff_perc_vel = csv_w_t if csv_w_t is not None else snap.get('perc_vel') if snap else None
    if (eff_perc_vel is not None
            and snap and snap.get('nv') is not None
            and snap.get('ev') is not None):
        v_err = eff_perc_vel - snap['nv']
        if v_err > GAMMA_MAX and eff_perc_vel > snap['ev']:
            src = "the CSV value fed to the classifier" if csv_w_t is not None else "perception"
            explanations.append(
                f"[E] PERCEPTION VELOCITY OVER-ESTIMATION: At the snapshot, "
                f"{src} reported the NPC velocity as "
                f"{eff_perc_vel:.2f} m/s but the true velocity was "
                f"{snap['nv']:.3f} m/s (error = +{v_err:.2f} m/s).")
            explanations.append(
                f"    Because perceived w_t ({eff_perc_vel:.2f}) > "
                f"ego u_t ({snap['ev']:.2f}), the framework's safe-pull-away")
            explanations.append(
                f"    shortcut fires (u_t < w_t -> safe). This is a velocity-")
            explanations.append(
                f"    perception calibration failure, distinct from distance error.")
            primary = primary or 'E'

    # Hypothesis D: framework simply wrong at this state
    if not explanations:
        explanations.append(
            "[D] UNEXPLAINED: No clear perception or snapshot-selection issue. "
            "The framework classified a genuinely dangerous state as safe.")
        explanations.append(
            "    This suggests either a framework-assumption violation (e.g., NPC")
        explanations.append(
            "    decelerating faster than BETA_MAX) or a corner case worth deeper")
        explanations.append(
            "    inspection.")
        primary = 'D'

    return primary, explanations


# -----------------------------------------------------------------
# Main diagnostic per file
# -----------------------------------------------------------------
def diagnose_one(csv_row, frames, plane):
    """
    Build a diagnostic report for one false-negative scenario.
    Returns a list of strings forming the report.
    """
    lines = []
    label = csv_row.get('label', '(unlabeled)')
    lines.append("=" * 72)
    lines.append(f"FALSE NEGATIVE: {label}")
    lines.append("=" * 72)

    # 1. Snapshot frame: re-derive what the row was extracted from
    # The label format is "<filename>@<timestamp>"
    try:
        snap_t = float(label.split('@')[-1])
    except Exception:
        snap_t = None

    if snap_t is None:
        lines.append("  Cannot parse snapshot timestamp from label.")
        return lines

    snap_i = find_frame_at_time(frames, snap_t)
    if snap_i is None:
        lines.append("  Cannot locate snapshot frame in JSON.")
        return lines

    snap = frame_state(frames[snap_i], plane)

    # 2. Find the actual collision frame using SAT on all frames
    coll_info = detect_collision_in_trajectory(frames, plane)
    coll_i = coll_info.get('collision_frame')
    coll_t = coll_info.get('collision_time')
    coll_state = frame_state(frames[coll_i], plane) if coll_i is not None else None

    # 3. Perception coverage in the run-up
    pwin = None
    if coll_i is not None:
        pwin = perception_gap_around(frames, coll_i, window_before_s=3.0)

    # 4. Trajectory-level perception stats
    perc_stats = compute_perception_stats(frames)

    # ---- Render report ----
    lines.append("")
    lines.append(f"  Snapshot:   t = {snap['t']:.3f}s  (frame {snap_i})")
    lines.append(f"  Collision:  t = {coll_t:.3f}s  (frame {coll_i})" if coll_t
                 else "  Collision: (no SAT-detected contact in JSON)")
    if coll_t and snap['t'] is not None:
        delta = snap['t'] - coll_t
        rel = "BEFORE" if delta < 0 else "AFTER"
        lines.append(f"             snapshot is {abs(delta):.2f}s {rel} collision")

    lines.append("")
    lines.append("  --- AT THE SNAPSHOT FRAME ---")
    if snap['ev'] is not None and snap['nv'] is not None:
        closing = snap['ev'] - (snap['nv'] or 0)
        lines.append(f"    Ego velocity      : {snap['ev']:.3f} m/s")
        lines.append(f"    NPC velocity (gt) : {snap['nv']:.3f} m/s")
        lines.append(f"    Closing rate      : {closing:+.3f} m/s "
                     f"(positive = closing)")
    if snap.get('long_gap_gt') is not None:
        lines.append(f"    Longitudinal gap (ground truth)  : "
                     f"{snap['long_gap_gt']:.3f} m")
    if snap.get('long_gap_perc') is not None:
        err = snap['long_gap_perc'] - snap['long_gap_gt']
        lines.append(f"    Longitudinal gap (perceived)     : "
                     f"{snap['long_gap_perc']:.3f} m  "
                     f"(error = {err:+.3f} m vs ground truth)")
    else:
        lines.append(f"    Longitudinal gap (perceived)     : NPC NOT DETECTED")
    lines.append(f"    Perception objects in this frame : "
                 f"{snap['perc_count']}")

    if coll_state and coll_state['t'] != snap['t']:
        lines.append("")
        lines.append("  --- AT THE COLLISION MOMENT ---")
        if coll_state['ev'] is not None:
            cc = coll_state['ev'] - (coll_state['nv'] or 0)
            lines.append(f"    Ego velocity      : {coll_state['ev']:.3f} m/s")
            lines.append(f"    NPC velocity      : {coll_state['nv']:.3f} m/s")
            lines.append(f"    Closing rate      : {cc:+.3f} m/s")
        if coll_state.get('long_gap_gt') is not None:
            lines.append(f"    Gap at impact     : "
                         f"{coll_state['long_gap_gt']:.3f} m (true)")
        lines.append(f"    Perception objects: {coll_state['perc_count']}")
        # Minimum penetration depth seen by SAT detector
        if coll_info.get('min_gap_m') is not None:
            lines.append(f"    Box center-to-center min : "
                         f"{coll_info['min_gap_m']:.3f} m")

    if pwin is not None:
        n, nw, total_s, lgap = pwin
        lines.append("")
        lines.append(f"  --- PERCEPTION TIMELINE (3s before collision) ---")
        lines.append(f"    Frames in window               : {n} "
                     f"(spanning {total_s:.2f}s)")
        lines.append(f"    Frames with perception data    : {nw}/{n} "
                     f"({100*nw/n if n else 0:.0f}%)")
        lines.append(f"    Longest blackout in window     : {lgap:.2f} s")

    lines.append("")
    lines.append(f"  --- TRAJECTORY-LEVEL PERCEPTION ---")
    lines.append(f"    Overall coverage           : "
                 f"{perc_stats['coverage']:.3f}")
    lines.append(f"    Longest blackout (full run): "
                 f"{perc_stats['max_gap_s']:.2f} s")

    # ---- Cause attribution ----
    # Pull the perceived NPC velocity that was actually fed to the classifier.
    # Field name depends on scenario:
    #   deceleration -> 'w_t'
    #   cutin        -> 'w_cut'
    #   cutout       -> 'v_rev_hat'
    def _safe_float(key):
        try:
            v = csv_row.get(key)
            return float(v) if v not in (None, '') else None
        except (TypeError, ValueError):
            return None
    scen = (csv_row.get('scenario') or '').lower()
    if scen == 'deceleration':
        csv_w_t = _safe_float('w_t')
    elif scen == 'cutin':
        csv_w_t = _safe_float('w_cut')
    elif scen == 'cutout':
        csv_w_t = _safe_float('v_rev_hat')
    else:
        csv_w_t = None

    cause, exps = attribute_cause(snap, None, coll_state, pwin, plane,
                                    csv_w_t=csv_w_t)
    lines.append("")
    lines.append(f"  --- ATTRIBUTION ---")
    lines.append(f"    Primary cause: {cause}")
    for line in exps:
        lines.append(f"    {line}")

    lines.append("")
    return lines


# -----------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------
def find_json_file(json_dir, label):
    """Extract the JSON filename from the label and locate it on disk."""
    # Label is "<filename>@<timestamp>"
    fname = label.split('@')[0]
    p = json_dir / fname
    if p.exists():
        return p
    # Try common variants
    for cand in [json_dir / fname,
                 json_dir / fname.replace('.json', '_data.json')]:
        if cand.exists(): return cand
    return None


def main():
    ap = argparse.ArgumentParser(description="Diagnose PCB false negatives.")
    ap.add_argument("csv", help="CSV file with PCB classifications (must contain "
                                "'collision_occurred' and the snapshot rows).")
    ap.add_argument("json_dir", help="Directory containing original *_data.json files")
    ap.add_argument("--output", default=None,
                    help="Write report to file (default: stdout)")
    args = ap.parse_args()

    json_dir = Path(args.json_dir)

    # Re-classify each row to find which are false negatives.
    fn_rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                col = int(r.get('collision_occurred') or 0)
            except Exception:
                col = 0
            if col != 1: continue
            # Re-run classifier to determine region
            scen = (r.get('scenario') or '').lower()
            def _f(k): return float(r[k]) if r.get(k) not in (None, '') else None
            try:
                if scen == 'deceleration':
                    region = classify_deceleration(_f('p_t'), _f('u_t'), _f('w_t'))
                elif scen == 'cutin':
                    region = classify_cutin(_f('p_long_t'), _f('v_y_rel'),
                                              _f('u_t'), _f('w_cut'), _f('d_y'))
                elif scen == 'cutout':
                    region = classify_cutout(_f('p_rev'), _f('v_y_occ'),
                                               _f('u_t'), _f('v_rev_hat'),
                                               _f('t_blackout'), _f('d_y_occ'))
                else:
                    region = None
            except Exception as e:
                print(f"warning: could not classify row {r.get('label')}: {e}",
                      file=sys.stderr)
                continue
            if region == 'safe':
                fn_rows.append((r, scen))

    out_lines = []
    out_lines.append("PCB FRAMEWORK -- FALSE-NEGATIVE DIAGNOSTIC REPORT")
    out_lines.append("=" * 72)
    out_lines.append(f"Source CSV : {args.csv}")
    out_lines.append(f"JSON dir   : {args.json_dir}")
    out_lines.append(f"False negatives found: {len(fn_rows)}")
    out_lines.append("")

    cause_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for row, scen in fn_rows:
        plane = 'XZ' if scen == 'cutout' else 'XY'
        label = row.get('label', '')
        json_path = find_json_file(json_dir, label)
        if json_path is None:
            out_lines.append(f"[skip] {label}: JSON not found in {args.json_dir}")
            continue
        try:
            with open(json_path) as f:
                d = json.load(f)
        except Exception as e:
            out_lines.append(f"[skip] {label}: failed to load JSON ({e})")
            continue
        frames = d.get('frames') or []
        if not frames:
            out_lines.append(f"[skip] {label}: no frames")
            continue
        report = diagnose_one(row, frames, plane)
        out_lines.extend(report)
        # Find the primary-cause line and count it
        for line in report:
            if 'Primary cause:' in line:
                cause = line.split(':')[-1].strip()
                if cause in cause_counts: cause_counts[cause] += 1
                break

    # Summary table at the end
    out_lines.append("=" * 72)
    out_lines.append("SUMMARY OF CAUSES")
    out_lines.append("=" * 72)
    out_lines.append(f"  [A] Perception failure          : {cause_counts['A']}")
    out_lines.append(f"  [B] Slow-speed accumulation     : {cause_counts['B']}")
    out_lines.append(f"  [C] Distance over-estimation    : {cause_counts['C']}")
    out_lines.append(f"  [E] Velocity over-estimation    : {cause_counts['E']}")
    out_lines.append(f"  [D] Unexplained                 : {cause_counts['D']}")
    out_lines.append("")
    out_lines.append("Causes are not mutually exclusive; primary cause is the")
    out_lines.append("first hypothesis whose evidence threshold was met.")

    output = "\n".join(out_lines)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Wrote diagnostic report to {args.output}")
        print(f"Diagnosed {len(fn_rows)} false negatives.")
        print(f"Cause breakdown: A={cause_counts['A']}, B={cause_counts['B']}, "
              f"C={cause_counts['C']}, E={cause_counts['E']}, "
              f"D={cause_counts['D']}")
    else:
        print(output)


if __name__ == "__main__":
    main()