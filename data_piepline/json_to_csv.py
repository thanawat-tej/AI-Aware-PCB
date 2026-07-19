"""
JSON  ->  PCB-overlay CSV extractor.

Reads JSON files produced by json_1.py (deceleration, cut-in) or
json_1_cutout.py (cut-out) and emits a CSV in the schema expected by
pcb_analysis.py.

Scenario type is auto-detected from the filename prefix:
    decel*   -> deceleration  (XY ground plane, Z up,  npc1 = lead)
    cutin*   -> cut-in        (XY ground plane, Z up,  npc1 = cut-in vehicle)
    cutout*  -> cut-out       (XZ ground plane, Y up,  npc1 = exiting occluder,
                                                       npc2 = revealed vehicle)

Filename suffix encoding:
    decel*-NNNN  -> NPC deceleration in m/s^2 (NNNN/1000)
    cutin*-NN    -> NPC lateral velocity in m/s (NN/10)
    cutout*-NN   -> NPC lateral velocity in m/s (NN/10)

Modes:
    snapshot     -- one row per scenario at the most safety-critical frame
    all          -- one row per frame (large output!)
    downsample   -- one row every Nth frame

Usage:
    python json_to_csv.py INPUT OUTPUT.csv [--mode {snapshot,all,downsample}]
                                           [--eps-bias 1.37]

    --eps-bias 1.37  -> bias-corrected run (debias perceived gaps by b_eps).
    omit / 0         -> uncorrected baseline.
                                            [--every N]
                                            [--scenario auto|deceleration|cutin|cutout]
                                            [--labels labels.csv]
                                            [--collision-label 0|1]
"""
from __future__ import annotations
import argparse, csv, json, math, os, re, sys
from pathlib import Path

# -------------------------------------------------------------------------
# Framework constants (used to estimate t_blackout from perception gaps)
# -------------------------------------------------------------------------
DEFAULT_D_Y_OCC = 2.0   # m, lateral clearance distance assumed for occluder

# Systematic perception offset b_eps (m). The perceived longitudinal gap is, on
# average, ~1.37 m LARGER than the true gap (Section ssec:eps_bias), in the
# dangerous direction. Set EPS_BIAS = 1.37 to run the BIAS-CORRECTED evaluation:
# perceived longitudinal gaps are debiased before the PCB check, matching the
# thesis constraint  p - b_eps - eps_max.  Leave at 0.0 to reproduce the
# uncorrected baseline. Applied ONLY to real perceptions, never to ground-truth
# fallbacks (those carry no perception bias).
EPS_BIAS = 0.0


def _debias(perc_long_abs, was_groundtruth):
    """Remove the systematic perception offset from a perceived longitudinal
    gap. Perceived is biased farther, so debiasing subtracts. No-op when the
    value is a ground-truth fallback or when EPS_BIAS is disabled."""
    if was_groundtruth or EPS_BIAS <= 0.0:
        return perc_long_abs
    return max(0.0, perc_long_abs - EPS_BIAS)


# Minimum penetration depth to count as a real collision. Set to ignore
# sub-millimetre numerical-touch artefacts from simulators that place
# vehicles bumper-to-bumper with tiny floating-point overshoots. 0.01 m
# (1 cm) is well below any meaningful physical contact event while
# eliminating numerical noise.
COLLISION_OVERLAP_M = 0.01

# -------------------------------------------------------------------------
# Filename metadata parsers
# -------------------------------------------------------------------------
_DECEL_RE  = re.compile(r'^decel.*-(\d{3,5})(?:_data)?\.json$', re.IGNORECASE)
_CUTIN_RE  = re.compile(r'^cutin.*-(\d{1,4})(?:_data)?\.json$', re.IGNORECASE)
_CUTOUT_RE = re.compile(r'^cutout.*-(\d{1,4})(?:_data)?\.json$', re.IGNORECASE)


def detect_scenario(filename):
    """Detect 'deceleration' / 'cutin' / 'cutout' from filename prefix."""
    n = filename.lower()
    if n.startswith('decel'):  return 'deceleration'
    if n.startswith('cutin'):  return 'cutin'
    if n.startswith('cutout'): return 'cutout'
    return None


def parse_npc_param_from_filename(filename, scenario):
    """
    Parse the NPC parameter from filenames.
      decel:   last numeric chunk / 1000  -> m/s^2 deceleration
      cutin:   last numeric chunk / 10    -> m/s lateral velocity
      cutout:  last numeric chunk / 10    -> m/s lateral velocity (npc1 only)
    Returns the parsed float, or None if not parseable.
    """
    try:
        if scenario == 'deceleration':
            m = _DECEL_RE.match(filename)
            if m: return float(m.group(1)) / 1000.0
        elif scenario == 'cutin':
            m = _CUTIN_RE.match(filename)
            if m: return float(m.group(1)) / 10.0
        elif scenario == 'cutout':
            m = _CUTOUT_RE.match(filename)
            if m: return float(m.group(1)) / 10.0
    except (ValueError, TypeError):
        return None
    return None


# -------------------------------------------------------------------------
# Coordinate convention dispatch
# -------------------------------------------------------------------------
# Cut-out files are Y-up (XZ ground plane); cut-in / deceleration are
# Z-up (XY ground plane). All geometry routines accept a 'plane' argument.

def _xy(p, plane):
    """Return the 2D position on the ground plane as (h, l) tuple."""
    if plane == 'XZ':
        return (p['x'], p['z'])
    return (p['x'], p['y'])

def _vxy(v, plane):
    """Return the 2D velocity on the ground plane as (vh, vl) tuple."""
    if plane == 'XZ':
        return (v['x'], v['z'])
    return (v['x'], v['y'])


def dist_2d(a, b, plane):
    ah, al = _xy(a, plane)
    bh, bl = _xy(b, plane)
    return math.hypot(ah - bh, al - bl)


def project_onto_heading(ego_pos, target_pos, ego_vel, plane):
    """
    Decompose the (target - ego) vector into (longitudinal, lateral)
    components relative to the ego's velocity direction.

    Returns (longitudinal_gap, lateral_offset, heading_valid).
    'heading_valid' is False when the ego's speed is too low to define a
    heading; in that case the caller should treat the values as unreliable.
    """
    eh, el   = _xy(ego_pos, plane)
    th, tl   = _xy(target_pos, plane)
    veh, vel = _vxy(ego_vel, plane)
    speed = math.hypot(veh, vel)
    if speed < 0.2:
        # Heading undefined; return unsigned distance as longitudinal and 0 lateral
        return math.hypot(th - eh, tl - el), 0.0, False
    fh, fl = veh / speed, vel / speed       # forward unit vector
    lh, ll = -fl, fh                         # lateral unit (rotated +90)
    dh, dl = th - eh, tl - el
    longitudinal = dh * fh + dl * fl
    lateral      = dh * lh + dl * ll
    return longitudinal, lateral, True


def _vel_mag(v):
    if v is None: return None
    if 'magnitude' in v and v['magnitude'] is not None:
        return float(v['magnitude'])
    # Fall back to vector magnitude
    try:
        return math.hypot(*[v[k] for k in ('x', 'y', 'z') if k in v])
    except Exception:
        return None


def _vel_relative_to_ego(target_vel, ego_vel, plane):
    """
    Decompose the target vehicle's velocity (in absolute frame) into
    components along ego's heading: returns (forward, lateral, heading_valid).
    """
    veh, vel = _vxy(ego_vel, plane)
    speed = math.hypot(veh, vel)
    if speed < 0.2:
        return _vel_mag(target_vel), 0.0, False
    fh, fl = veh / speed, vel / speed
    lh, ll = -fl, fh
    tvh, tvl = _vxy(target_vel, plane)
    forward = tvh * fh + tvl * fl
    lateral = tvh * lh + tvl * ll
    return forward, lateral, True


# -------------------------------------------------------------------------
# Collision detection (Separating Axis Theorem for oriented 2D boxes)
# -------------------------------------------------------------------------
def _box_to_2d_corners(corners_3d_or_2d, plane):
    """
    Extract the 4 ground-plane (x, y_or_z) corner pairs from a box.
    Stored boxes are already 2D pairs (length-2 lists) in the ground plane.
    """
    out = []
    for c in corners_3d_or_2d:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            out.append((float(c[0]), float(c[1])))
    return out


def _project_polygon(corners, axis):
    """Project polygon corners onto an axis; return (min, max) scalars."""
    dots = [c[0]*axis[0] + c[1]*axis[1] for c in corners]
    return min(dots), max(dots)


def _polygons_overlap(p1, p2, min_overlap=0.0):
    """
    Test convex-polygon overlap via the Separating Axis Theorem.

    min_overlap : minimum penetration depth (in metres) required to count
                  as a real collision. The default (0.0) flags any overlap,
                  including sub-millimetre numerical edge cases. Use ~0.01
                  (1 cm) to ignore numerical-touch artefacts from simulators
                  that place vehicles bumper-to-bumper with tiny overshoots.
    """
    if len(p1) < 3 or len(p2) < 3:
        return False
    for poly in (p1, p2):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # Edge normal (perpendicular)
            nx, ny = -(y2 - y1), (x2 - x1)
            mag = math.hypot(nx, ny)
            if mag < 1e-12: continue
            axis = (nx / mag, ny / mag)
            mn1, mx1 = _project_polygon(p1, axis)
            mn2, mx2 = _project_polygon(p2, axis)
            # Compute overlap depth on this axis (negative = gap)
            overlap_depth = min(mx1, mx2) - max(mn1, mn2)
            if overlap_depth < min_overlap:
                return False    # separating axis (with threshold)
    return True


def detect_collision_in_trajectory(frames, plane):
    """
    Scan all frames; return True if ego's bounding box ever overlaps any
    NPC's bounding box. Also returns (collision_frame_idx, min_gap_seen)
    so callers can locate the moment of contact for diagnostics.

    Returns: dict with keys
      'collided'        : bool  (any forward-side collision occurred)
      'collision_frame' : int or None (first forward-collision frame)
      'collision_time'  : float or None
      'min_gap_m'       : float (smallest center-to-center XY/XZ distance)
      'min_gap_frame'   : int
      'npc_hit'         : str ('npc1' or 'npc2') or None
      'rear_collision'  : bool  (a rear collision was detected; recorded
                                 separately and NOT counted as a true
                                 collision for safety analysis)
      'rear_collision_frame' : int or None
      'rear_collision_time'  : float or None
    """
    result = {
        'collided': False, 'collision_frame': None, 'collision_time': None,
        'min_gap_m': float('inf'), 'min_gap_frame': -1, 'npc_hit': None,
        'rear_collision': False,
        'rear_collision_frame': None, 'rear_collision_time': None,
    }

    # Track a recent forward direction so we can carry it through frames
    # where the ego is stopped (velocity-based heading undefined).
    last_known_forward = None

    for i, fr in enumerate(frames):
        ego = fr.get('ego') or {}
        ego_box = ego.get('box')
        ego_pos = ego.get('position')
        if not ego_box or not ego_pos:
            continue
        ego_corners = _box_to_2d_corners(ego_box, plane)
        if len(ego_corners) < 3: continue

        # Refresh ego forward direction when we have a reliable velocity
        ev = ego.get('velocity')
        fwd = _ego_forward_direction(ego_corners, ev, plane, last_known_forward)
        if fwd is not None:
            last_known_forward = fwd

        for npc_key in ('npc1', 'npc2'):
            npc = fr.get(npc_key)
            if not npc: continue
            npc_box = npc.get('box')
            npc_pos = npc.get('position')
            if not npc_box or not npc_pos:
                continue

            # Track minimum center-to-center gap
            g = dist_2d(ego_pos, npc_pos, plane)
            if g < result['min_gap_m']:
                result['min_gap_m'] = g
                result['min_gap_frame'] = i

            # Polygon overlap test (only if center gap is small enough
            # to plausibly overlap -- skips most frames quickly)
            if g > 10.0:
                continue

            npc_corners = _box_to_2d_corners(npc_box, plane)
            if _polygons_overlap(ego_corners, npc_corners,
                                  min_overlap=COLLISION_OVERLAP_M):
                # Decide front vs rear by projecting NPC centroid onto
                # ego forward axis.
                is_rear = _is_npc_behind_ego(ego_corners, npc_corners,
                                              last_known_forward, plane)
                if is_rear:
                    if not result['rear_collision']:
                        result['rear_collision'] = True
                        result['rear_collision_frame'] = i
                        result['rear_collision_time'] = fr.get('timestamp')
                else:
                    if not result['collided']:
                        result['collided'] = True
                        result['collision_frame'] = i
                        result['collision_time'] = fr.get('timestamp')
                        result['npc_hit'] = npc_key

    if not math.isfinite(result['min_gap_m']):
        result['min_gap_m'] = None
    return result


def _box_long_axis(corners):
    """
    Given 4 ground-plane corners of a rectangle, return the unit vector
    along the long axis (vehicle forward direction, up to a sign). The
    sign is resolved separately by velocity or by carry-over.
    """
    import math as _m
    # Compute the two distinct edge lengths
    edges = []
    n = len(corners)
    for k in range(n):
        a = corners[k]; b = corners[(k + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = _m.hypot(dx, dy)
        edges.append((L, (dx, dy)))
    # Pick the edge with the largest length
    edges.sort(key=lambda x: -x[0])
    L_long, (dx, dy) = edges[0]
    if L_long < 1e-6: return None
    return (dx / L_long, dy / L_long)


def _ego_forward_direction(ego_corners, ego_vel, plane, last_known):
    """
    Return a unit vector pointing in the ego's forward direction.

    Strategy:
      1. Compute the long axis of the bounding box (unsigned).
      2. Disambiguate sign using ego velocity (if speed >= 0.5 m/s).
      3. If velocity is too small, keep the previous known direction
         (passed in as `last_known`), aligning the box's long axis with it.
      4. If both are unavailable, return None.
    """
    import math as _m
    long_axis = _box_long_axis(ego_corners)
    if long_axis is None:
        return last_known

    speed = 0.0
    if ego_vel:
        speed = _vel_mag(ego_vel) or 0.0

    if speed >= 0.5:
        # Use velocity to disambiguate sign of the long axis
        vx, vy = _vxy(ego_vel, plane)
        v_unit = (vx / speed, vy / speed)
        dot = long_axis[0]*v_unit[0] + long_axis[1]*v_unit[1]
        if dot < 0:
            return (-long_axis[0], -long_axis[1])
        return long_axis

    if last_known is not None:
        # Ego stopped: keep the previous heading direction, but align the
        # long axis to that direction (in case the box has been re-emitted
        # with corners in a different order).
        dot = long_axis[0]*last_known[0] + long_axis[1]*last_known[1]
        if dot < 0:
            return (-long_axis[0], -long_axis[1])
        return long_axis

    return None


def _is_npc_behind_ego(ego_corners, npc_corners, ego_forward, plane):
    """
    Return True if the NPC's centroid is behind the ego along ego's
    forward direction. If forward is unknown, returns False (default:
    treat as forward collision, since we can't be sure).
    """
    if ego_forward is None:
        return False
    # Centroids
    ex = sum(c[0] for c in ego_corners) / len(ego_corners)
    ey = sum(c[1] for c in ego_corners) / len(ego_corners)
    nx = sum(c[0] for c in npc_corners) / len(npc_corners)
    ny = sum(c[1] for c in npc_corners) / len(npc_corners)
    # Project NPC relative position onto ego's forward axis
    rx, ry = nx - ex, ny - ey
    fwd_proj = rx * ego_forward[0] + ry * ego_forward[1]
    return fwd_proj < 0  # NPC behind ego


# -------------------------------------------------------------------------
# Perceived-object matching
# -------------------------------------------------------------------------
def _pick_perceived_match(frame, gt_position, plane):
    """Pick the perceived object closest in 2D to the supplied ground-truth pos."""
    pobjs = frame.get('perception_objects') or []
    if not pobjs or gt_position is None:
        return None
    best, best_d = None, float('inf')
    for o in pobjs:
        p = o.get('position')
        if p is None: continue
        d = dist_2d(p, gt_position, plane)
        if d < best_d:
            best, best_d = o, d
    return best


def _perceived_world_velocity(frames, frame_idx, gt_pos_history, plane,
                                window_seconds=0.3):
    """
    Compute a world-frame perceived velocity for the lead object by finite-
    differencing perceived positions across a small time window.

    Background: the JSON's perception_objects record velocity with
    source='twist', which means the velocity is in the perceived object's
    own body frame (forward speed and lateral speed in the NPC's local
    coordinate system), NOT in world frame. Reading these (vx, vy) values
    as if they were world-frame gives nonsense (e.g., a vehicle moving
    south at 11 m/s appears as moving east at 11 m/s).

    This function reconstructs a proper world-frame velocity by taking
    the perceived position at frame_idx and at an earlier frame within
    the window, then differencing.

    Returns:
        (vx_world, vy_world)  -- world-frame velocity in m/s
        or None if not enough perceived history is available.
    """
    if frame_idx <= 0:
        return None
    fr_now = frames[frame_idx]
    t_now = fr_now.get('timestamp')
    pobjs_now = fr_now.get('perception_objects') or []
    if not pobjs_now: return None
    # Match by closest to ground-truth NPC pos at this frame
    gt_now = gt_pos_history[frame_idx] if frame_idx < len(gt_pos_history) else None
    if gt_now is None: return None
    perc_now = _pick_perceived_match(fr_now, gt_now, plane)
    if not perc_now or not perc_now.get('position'): return None
    p_now = perc_now['position']

    # Scan backward for a frame within the window that has a match AND
    # has measurable displacement (perception sometimes holds the same
    # position across multiple frames, which would give a spurious zero
    # velocity).
    j = frame_idx - 1
    while j >= 0:
        fr_old = frames[j]
        t_old = fr_old.get('timestamp')
        if t_old is None or t_now is None: break
        dt = t_now - t_old
        if dt > window_seconds * 1.5: break
        gt_old = gt_pos_history[j] if j < len(gt_pos_history) else None
        if gt_old is not None:
            perc_old = _pick_perceived_match(fr_old, gt_old, plane)
            if perc_old and perc_old.get('position') and dt > 0.05:
                p_old = perc_old['position']
                if plane == 'XZ':
                    dx = p_now['x'] - p_old['x']
                    dy = p_now['z'] - p_old['z']
                else:
                    dx = p_now['x'] - p_old['x']
                    dy = p_now['y'] - p_old['y']
                # Require minimum displacement of 0.05 m to avoid the
                # case where the perception system held the same value
                # across consecutive frames.
                if math.hypot(dx, dy) >= 0.05:
                    return (dx / dt, dy / dt)
        j -= 1
    return None


def _build_gt_position_history(frames, key='npc1'):
    """Cache of ground-truth positions per frame for perception matching."""
    history = []
    for fr in frames:
        npc = fr.get(key)
        if npc and npc.get('position') is not None:
            history.append(npc['position'])
        else:
            history.append(None)
    return history


# -------------------------------------------------------------------------
# Row builders -- one per scenario
# -------------------------------------------------------------------------
def _blank_row():
    return {
        'scenario': '',
        'p_t': '', 'u_t': '', 'w_t': '',
        'p_long_t': '', 'v_y_rel': '', 'w_cut': '', 'd_y': '',
        'p_rev': '', 'v_y_occ': '', 'v_rev_hat': '',
        't_blackout': '', 'd_y_occ': '',
        'collision_occurred': '',
        'ground_truth_dist': '', 'ground_truth_vel': '',
        'ground_truth_v_y_rel': '','ground_truth_d_y': '', 'ground_truth_v_y_occ': '',
        'v_y_occ_peak': '',
        'npc_actual_decel': '',
        'ego_max_accel': '',
        'ego_accel_prebrake': '',
        'filename_param': '',
        'perception_detected': '',
        'inputs_were_groundtruth': '',
        'perception_coverage': '',
        'perception_max_gap_s': '',
        'staleness_at_decision_s': '',
        'label': '',
    }


def compute_perception_stats(frames):
    """
    Analyse perception coverage across a trajectory.

    Returns a dict with:
      'coverage'   : fraction of active (ego-moving) frames with >=1 perception object
      'max_gap_s'  : longest consecutive period (seconds) with zero perception
                     objects while ego was moving
      'n_active'   : number of frames where ego was moving
    """
    active_frames = []   # (idx, t, has_perception)
    for i, fr in enumerate(frames):
        ev = (fr.get('ego') or {}).get('velocity')
        if not ev: continue
        v_mag = _vel_mag(ev) or 0
        if v_mag < 0.5: continue
        has_perc = bool(fr.get('perception_objects'))
        t = fr.get('timestamp') or 0
        active_frames.append((i, t, has_perc))

    if not active_frames:
        return {'coverage': 0.0, 'max_gap_s': 0.0, 'n_active': 0}

    n_with = sum(1 for _, _, h in active_frames if h)
    coverage = n_with / len(active_frames)

    # Find longest consecutive zero-perception run (in seconds)
    max_gap = 0.0
    cur_start_t = None
    for _, t, has_perc in active_frames:
        if not has_perc:
            if cur_start_t is None:
                cur_start_t = t
            max_gap = max(max_gap, t - cur_start_t)
        else:
            cur_start_t = None

    return {'coverage': coverage,
            'max_gap_s': max_gap,
            'n_active': len(active_frames)}


def staleness_at_decision(frames, frame_idx, plane='XY'):
    """Seconds since the LEAD (npc1) was last matched in perception, as of the
    decision frame frame_idx. 0.0 if the lead is perceived at the decision frame
    itself. Falls back to time-since-scenario-start if the lead was never seen
    up to frame_idx. This is the staleness of the ego's last LEAD sighting --
    the quantity the PCB actually decides on -- not 'was any object visible'."""
    if not frames or frame_idx is None or frame_idx >= len(frames):
        return 0.0
    t_dec = frames[frame_idx].get('timestamp') or 0.0
    for j in range(frame_idx, -1, -1):
        npc = frames[j].get('npc1') or {}
        npc_pos = npc.get('position')
        if npc_pos is None:
            continue
        # lead counts as "seen" only if perception has a match for it this frame
        if lead_perceived(frames[j], npc_pos, plane):
            return max(0.0, t_dec - (frames[j].get('timestamp') or 0.0))
    t0 = frames[0].get('timestamp') or 0.0
    return max(0.0, t_dec - t0)

def lead_perceived(frame, npc_pos, plane='XY', gate=3.5):
    """True only if the nearest perceived object is within `gate` metres of the
    lead's ground-truth position -- i.e. the LEAD was tracked, not just 'some
    object exists'. gate ~ a vehicle half-length; widen if perception error is
    large, since a big gap bias can push a true lead match beyond a tight gate."""
    m = _pick_perceived_match(frame, npc_pos, plane)
    if m is None or m.get('position') is None:
        return False
    return dist_2d(m['position'], npc_pos, plane) <= gate

def build_deceleration_row(frame, source_file, collision_label, npc_decel,
                            plane='XY', perception_stats=None,
                            frames=None, frame_idx=None,
                            gt_pos_history=None):
    ego = frame.get('ego') or {}
    npc = frame.get('npc1') or {}
    ego_pos, ego_vel = ego.get('position'), ego.get('velocity')
    npc_pos, npc_vel = npc.get('position'), npc.get('velocity')
    if ego_pos is None or npc_pos is None or ego_vel is None:
        return None

    u_t = _vel_mag(ego_vel)
    if u_t is None: return None

    # Ground-truth longitudinal gap
    gt_long, _, _ = project_onto_heading(ego_pos, npc_pos, ego_vel, plane)
    gt_long = abs(gt_long)
    npc_speed = _vel_mag(npc_vel)

    # Perceived gap and velocity.
    #
    # For DECELERATION, the framework only needs scalar speed (no direction).
    # The JSON's perception_objects velocity vector is in the OBJECT's body
    # frame (source='twist'), but the 'magnitude' field is frame-independent
    # and equals the NPC's actual speed -- we can use it directly.
    #
    # Finite-difference reconstruction of velocity from positions introduces
    # numerical noise that can occasionally push perceived w_t above ego u_t,
    # triggering the framework's safe-pull-away shortcut and creating fake
    # 'safe' classifications. The body-frame magnitude is more reliable for
    # this scalar use, so we prefer it for deceleration.
    perc = _pick_perceived_match(frame, npc_pos, plane)
    inputs_were_groundtruth = 0
    if perc and perc.get('position') is not None:
        perc_long, _, _ = project_onto_heading(ego_pos, perc['position'], ego_vel, plane)
        perc_long = abs(perc_long)
        # Prefer the perception object's velocity magnitude (body-frame
        # vector but its magnitude is correct as a scalar speed).
        perc_speed = _vel_mag(perc.get('velocity'))
        if perc_speed is None:
            # No usable perceived velocity: fall back to ground truth for w_t
            perc_speed = npc_speed
            inputs_were_groundtruth = 1
        perception_detected = 1
    else:
        # No perception of the lead at all in this frame -- the AV would
        # have had no input. The CSV records ground truth for completeness
        # but flags this so analysis can isolate these cases.
        perc_long = gt_long
        perc_speed = npc_speed
        perception_detected = 0
        inputs_were_groundtruth = 1
    if perc_speed is None: perc_speed = npc_speed

    row = _blank_row()
    row.update({
        'scenario': 'deceleration',
        'p_t':      round(_debias(perc_long, inputs_were_groundtruth), 3),
        'u_t':      round(u_t, 3),
        'w_t':      round(perc_speed, 3) if perc_speed is not None else '',
        'collision_occurred': '' if collision_label is None else int(collision_label),
        'ground_truth_dist': round(gt_long, 3),
        'ground_truth_vel':  round(npc_speed, 3) if npc_speed is not None else '',
        'npc_actual_decel': round(npc_decel, 4) if npc_decel is not None else '',
        'filename_param':  round(npc_decel, 4) if npc_decel is not None else '',
        'perception_detected': perception_detected,
        'inputs_were_groundtruth': inputs_were_groundtruth,
        'perception_coverage': round(perception_stats['coverage'], 3)
                               if perception_stats else '',
        'perception_max_gap_s': round(perception_stats['max_gap_s'], 3)
                                if perception_stats else '',
        'staleness_at_decision_s': round(staleness_at_decision(frames, frame_idx), 3)
                                  if (frames is not None and frame_idx is not None) else '',
        'label': f"{source_file}@{frame.get('timestamp')}",
    })
    return row


def build_cutin_row(frame, source_file, collision_label, npc_lateral_param,
                     plane='XY', perception_stats=None,
                     frames=None, frame_idx=None, gt_pos_history=None):
    """
    Cut-in: npc1 is the cutting-in vehicle. The longitudinal gap is its
    distance ahead of ego along ego heading; the lateral closing speed is
    its velocity component perpendicular to ego heading (positive = closing).
    d_y is the perpendicular distance from npc1 to ego's lane center, which
    we approximate as |lateral_offset|.
    """
    ego = frame.get('ego') or {}
    npc = frame.get('npc1') or {}
    ego_pos, ego_vel = ego.get('position'), ego.get('velocity')
    npc_pos, npc_vel = npc.get('position'), npc.get('velocity')
    if ego_pos is None or npc_pos is None or ego_vel is None or npc_vel is None:
        return None

    u_t = _vel_mag(ego_vel)
    if u_t is None or u_t < 0.5:
        return None   # ego not moving -> heading undefined -> skip

    gt_long, gt_lat, ok = project_onto_heading(ego_pos, npc_pos, ego_vel, plane)
    if not ok:
        return None
    npc_forward_gt, npc_lat_gt, _ = _vel_relative_to_ego(npc_vel, ego_vel, plane)

    # Perceived equivalents.
    # IMPORTANT: as in build_deceleration_row, the JSON's perception_objects
    # velocity is in the OBJECT'S body frame (source='twist'), not world
    # frame. We must compute the world-frame perceived velocity by finite-
    # differencing perceived positions across a short time window.
    perc = _pick_perceived_match(frame, npc_pos, plane)
    inputs_were_groundtruth = 0
    if perc and perc.get('position') is not None:
        perc_long, perc_lat, _ = project_onto_heading(
            ego_pos, perc['position'], ego_vel, plane)
        # Finite-difference perceived velocity (world frame)
        perc_v_world = None
        if frames is not None and frame_idx is not None and gt_pos_history is not None:
            perc_v_world = _perceived_world_velocity(
                frames, frame_idx, gt_pos_history, plane)
        if perc_v_world is not None:
            # Construct a velocity dict mimicking the JSON format so we can
            # reuse _vel_relative_to_ego.
            v_dict = {'x': perc_v_world[0],
                      'y': perc_v_world[1] if plane == 'XY' else 0.0,
                      'z': perc_v_world[1] if plane == 'XZ' else 0.0}
            perc_forward, perc_lat_vel, _ = _vel_relative_to_ego(
                v_dict, ego_vel, plane)
        else:
            # No usable perceived history -- fall back to GT velocity
            perc_forward, perc_lat_vel = npc_forward_gt, npc_lat_gt
            inputs_were_groundtruth = 1
    else:
        perc_long, perc_lat = gt_long, gt_lat
        perc_forward, perc_lat_vel = npc_forward_gt, npc_lat_gt
        inputs_were_groundtruth = 1

    # Sign convention:
    #   v_y_rel > 0  =>  npc closing toward ego's lane (lateral motion toward ego)
    # Take |lateral velocity| as the closing magnitude; sign of lateral_offset
    # tells us which side, but for the PCB we just care about closing speed.
    # Was the lead actually perceived at this frame?
    perception_detected = 1 if (perc and perc.get('position')) else 0

    row = _blank_row()
    row.update({
        'scenario': 'cutin',
        'p_long_t': round(_debias(abs(perc_long), inputs_were_groundtruth), 3),
        'v_y_rel':  round(abs(perc_lat_vel), 3),
        'u_t':      round(u_t, 3),
        'w_cut':    round(abs(perc_forward), 3),
        'd_y':      round(abs(perc_lat), 3),
        'collision_occurred': '' if collision_label is None else int(collision_label),
        'ground_truth_dist': round(abs(gt_long), 3),
        'ground_truth_vel':  round(abs(npc_forward_gt), 3),
        'ground_truth_v_y_rel': round(abs(npc_lat_gt), 3),
        'ground_truth_d_y': round(abs(gt_lat), 3),
        'filename_param':  round(npc_lateral_param, 3) if npc_lateral_param is not None else '',
        'perception_detected': perception_detected,
        'inputs_were_groundtruth': inputs_were_groundtruth,
        'perception_coverage': round(perception_stats['coverage'], 3)
                               if perception_stats else '',
        'perception_max_gap_s': round(perception_stats['max_gap_s'], 3)
                                if perception_stats else '',
        'staleness_at_decision_s': round(staleness_at_decision(frames, frame_idx), 3)
                                  if (frames is not None and frame_idx is not None) else '',
        'label': f"{source_file}@{frame.get('timestamp')}",
    })
    return row


def build_cutout_row(frame, source_file, collision_label, npc_lateral_param,
                      t_blackout, plane='XZ', perception_stats=None,
                      frames=None, frame_idx=None,
                      gt_pos_history_npc1=None, gt_pos_history_npc2=None,
                      v_y_occ_peak=None):
    """
    Cut-out: npc1 is the occluder that EXITS the lane (laterally); npc2 (when
    present) is the revealed vehicle. We compute the cut-out PCB inputs at
    the moment of revelation, defined as the first frame where npc1 is
    moving laterally (cutting out) and npc2 is in front of ego.

      p_rev      = perceived longitudinal gap to npc2 (revealed vehicle)
      v_y_occ    = lateral exit velocity of npc1 (occluder)
      v_rev_hat  = perceived velocity of npc2
      t_blackout = duration the revealed vehicle was occluded (estimated)
      d_y_occ    = lateral clearance distance (fixed parameter)
    """
    ego = frame.get('ego') or {}
    npc1 = frame.get('npc1') or {}
    npc2 = frame.get('npc2') or {}
    ego_pos, ego_vel = ego.get('position'), ego.get('velocity')
    npc1_vel = npc1.get('velocity')
    npc2_pos, npc2_vel = npc2.get('position'), npc2.get('velocity')

    if (ego_pos is None or ego_vel is None or
        npc1_vel is None or npc2_pos is None):
        return None

    u_t = _vel_mag(ego_vel)
    if u_t is None or u_t < 0.5:
        return None

    # Lateral velocity of occluder (npc1)
    _, npc1_lat_gt, ok = _vel_relative_to_ego(npc1_vel, ego_vel, plane)
    if not ok:
        return None
    v_y_occ_gt = abs(npc1_lat_gt)

    # Revealed vehicle (npc2) gap and velocity
    gt_long_rev, _, _ = project_onto_heading(ego_pos, npc2_pos, ego_vel, plane)
    gt_long_rev = abs(gt_long_rev)
    npc2_speed_gt = _vel_mag(npc2_vel)

    # Perceived equivalents.
    # For NPC2 (revealed vehicle): the framework needs scalar speed
    # v_rev_hat, so the body-frame magnitude is correct and avoids
    # finite-difference noise. For NPC1 (occluder): the framework needs
    # the LATERAL component of velocity, so we must finite-difference
    # positions to get a meaningful world-frame vector.
    perc_rev = _pick_perceived_match(frame, npc2_pos, plane)
    inputs_were_groundtruth = 0
    if perc_rev and perc_rev.get('position') is not None:
        perc_long_rev, _, _ = project_onto_heading(
            ego_pos, perc_rev['position'], ego_vel, plane)
        perc_long_rev = abs(perc_long_rev)
        # Use body-frame magnitude (scalar speed is what we need)
        perc_speed_rev = _vel_mag(perc_rev.get('velocity'))
        if perc_speed_rev is None:
            perc_speed_rev = npc2_speed_gt
            inputs_were_groundtruth = 1
    else:
        perc_long_rev = gt_long_rev
        perc_speed_rev = npc2_speed_gt
        inputs_were_groundtruth = 1
    if perc_speed_rev is None: perc_speed_rev = npc2_speed_gt

    # Perceived occluder lateral velocity: finite-difference perceived npc1
    perc_occ = _pick_perceived_match(frame,
                                      npc1.get('position') or {}, plane)
    v_y_occ_perc = None
    if perc_occ and perc_occ.get('position') is not None:
        perc_v_world_occ = None
        if (frames is not None and frame_idx is not None and
                gt_pos_history_npc1 is not None):
            perc_v_world_occ = _perceived_world_velocity(
                frames, frame_idx, gt_pos_history_npc1, plane)
        if perc_v_world_occ is not None:
            v_dict = {'x': perc_v_world_occ[0],
                      'y': perc_v_world_occ[1] if plane == 'XY' else 0.0,
                      'z': perc_v_world_occ[1] if plane == 'XZ' else 0.0}
            _, perc_lat_occ, _ = _vel_relative_to_ego(v_dict, ego_vel, plane)
            v_y_occ_perc = abs(perc_lat_occ)
    if v_y_occ_perc is None:
        v_y_occ_perc = v_y_occ_gt
        inputs_were_groundtruth = 1

    # If occluder appears nearly stationary but the filename says otherwise,
    # use the filename's target lateral velocity as a fallback for v_y_occ.
    if v_y_occ_perc < 0.1 and npc_lateral_param and npc_lateral_param > 0.1:
        v_y_occ_perc = npc_lateral_param

    # Was the revealed vehicle actually perceived at this frame?
    perception_detected_rev = 1 if (perc_rev and perc_rev.get('position')) else 0

    row = _blank_row()
    row.update({
        'scenario': 'cutout',
        'p_rev':       round(_debias(perc_long_rev, inputs_were_groundtruth), 3),
        'v_y_occ':     round(v_y_occ_perc, 3),
        'u_t':         round(u_t, 3),
        'v_rev_hat':   round(perc_speed_rev, 3) if perc_speed_rev is not None else '',
        't_blackout':  round(t_blackout, 3),
        'd_y_occ':     DEFAULT_D_Y_OCC,
        'collision_occurred': '' if collision_label is None else int(collision_label),
        'ground_truth_dist':    round(gt_long_rev, 3),
        'ground_truth_vel':     round(npc2_speed_gt, 3) if npc2_speed_gt is not None else '',
        'ground_truth_v_y_occ': round(v_y_occ_gt, 3),
        'v_y_occ_peak': round(v_y_occ_peak, 3) if v_y_occ_peak is not None else '',
        'filename_param':  round(npc_lateral_param, 3) if npc_lateral_param is not None else '',
        'perception_detected': perception_detected_rev,
        'inputs_were_groundtruth': inputs_were_groundtruth,
        'perception_coverage': round(perception_stats['coverage'], 3)
                               if perception_stats else '',
        'perception_max_gap_s': round(perception_stats['max_gap_s'], 3)
                                if perception_stats else '',
        'staleness_at_decision_s': round(staleness_at_decision(frames, frame_idx), 3)
                                  if (frames is not None and frame_idx is not None) else '',
        'label': f"{source_file}@{frame.get('timestamp')}",
    })
    return row


# -------------------------------------------------------------------------
# Critical-frame selection per scenario
# -------------------------------------------------------------------------
def _pos_of(frame, who):
    obj = frame.get(who) or {}
    return obj.get('position')


def critical_frame_deceleration(frames, plane='XY', collision_frame=None):
    """
    Pick the most safety-critical frame in this trajectory.

    Strategy:
      1. If a collision occurred at `collision_frame`, restrict the search
         to frames at or before that moment. This avoids picking
         post-collision frames where the dynamics are no longer meaningful
         (vehicles bouncing apart, NPC stopped after impact, etc.).
      2. Among the candidate frames, pick the one where the ego is closing
         (rv = ev - nv > 0) AND the gap is smallest. Smallest gap during
         a closing phase is the moment of maximum threat.
      3. If no closing frame exists, fall back to smallest-gap overall.
    """
    upper = collision_frame if collision_frame is not None else len(frames)

    # Strategy 2: smallest gap during a closing phase
    best_idx, best_gap = None, float('inf')
    for i in range(upper):
        fr = frames[i]
        ev = (fr.get('ego') or {}).get('velocity')
        nv = (fr.get('npc1') or {}).get('velocity')
        if ev is None or nv is None: continue
        ev_mag = _vel_mag(ev); nv_mag = _vel_mag(nv)
        if ev_mag is None or nv_mag is None: continue
        rv = ev_mag - nv_mag
        if rv <= 0: continue
        ep = _pos_of(fr, 'ego'); np_ = _pos_of(fr, 'npc1')
        if ep is None or np_ is None: continue
        gap = dist_2d(ep, np_, plane)
        if gap <= 0.1: continue
        if gap < best_gap:
            best_idx, best_gap = i, gap

    if best_idx is not None:
        return best_idx

    # Strategy 3: fallback -- smallest gap overall (in pre-collision range)
    best_idx, best_gap = None, float('inf')
    for i in range(upper):
        fr = frames[i]
        ep = _pos_of(fr, 'ego'); np_ = _pos_of(fr, 'npc1')
        if ep is None or np_ is None: continue
        g = dist_2d(ep, np_, plane)
        if g < best_gap:
            best_idx, best_gap = i, g
    return best_idx


def critical_frame_cutin(frames, plane='XY', collision_frame=None):
    """
    Pick the most safety-critical frame in a cut-in trajectory.

    Strategy:
      1. If a collision occurred, restrict the search to frames at or before
         the collision moment.
      2. Among those frames, prefer the frame closest to the collision
         where: (a) npc1 has crossed into ego's lateral threat range
         (|lateral_offset| < ~2m), AND (b) ego is closing on npc1
         longitudinally, AND (c) longitudinal gap is small.
      3. If no such frame is found, fall back to the original logic
         (first frame npc1 starts lateral motion).
    """
    upper = collision_frame if collision_frame is not None else len(frames)

    # Strategy 2: prefer the frame closest to collision with small gap
    # and active closing, weighted by lateral threat.
    best_idx, best_score = None, -float('inf')
    for i in range(upper):
        fr = frames[i]
        ev = (fr.get('ego') or {}).get('velocity')
        nv = (fr.get('npc1') or {}).get('velocity')
        if ev is None or nv is None: continue
        ev_mag = _vel_mag(ev)
        if ev_mag is None or ev_mag < 0.5: continue
        ep = _pos_of(fr, 'ego'); np_ = _pos_of(fr, 'npc1')
        if ep is None or np_ is None: continue

        # Longitudinal gap and lateral offset relative to ego heading
        long_gap, lat_off, ok = project_onto_heading(ep, np_, ev, plane)
        if not ok or long_gap <= 0.1: continue
        if long_gap > 80.0: continue   # too far to be the moment of danger

        # Longitudinal closing
        fwd_npc, _, _ = _vel_relative_to_ego(nv, ev, plane)
        closing = ev_mag - fwd_npc
        if closing <= 0: continue   # not closing

        # Score: prefer small gap, high closing, lateral within threat zone
        lat_term = max(0.0, 2.0 - abs(lat_off))   # bonus when within 2m laterally
        score = (closing / max(1.0, long_gap)) + 0.1 * lat_term

        if score > best_score:
            best_score, best_idx = score, i

    if best_idx is not None:
        return best_idx

    # Strategy 3: fallback -- first frame with significant lateral motion
    threshold = 0.5
    for i in range(upper):
        fr = frames[i]
        ev = (fr.get('ego') or {}).get('velocity')
        nv = (fr.get('npc1') or {}).get('velocity')
        if ev is None or nv is None: continue
        if (_vel_mag(ev) or 0) < 0.5: continue
        _, lat, ok = _vel_relative_to_ego(nv, ev, plane)
        if not ok: continue
        if abs(lat) >= threshold:
            return i

    # Last resort: any frame
    return 0 if frames else None


def peak_npc1_lateral_velocity(frames, plane='XZ', collision_frame=None):
    """
    Scan all pre-collision frames and return the maximum |lateral velocity|
    of npc1 (the occluder), projected against ego heading. This represents
    the peak/steady-state lateral motion of the cut-out actor and is used
    for plotting the PCB overlay (not for classification, which still uses
    the snapshot's v_y_occ).

    Returns 0.0 if npc1 is never observed moving or ego never moves enough
    for heading to be defined.
    """
    upper = collision_frame if collision_frame is not None else len(frames)
    peak = 0.0
    for i in range(upper):
        fr = frames[i]
        ev = (fr.get('ego') or {}).get('velocity')
        nv1 = (fr.get('npc1') or {}).get('velocity')
        if ev is None or nv1 is None:
            continue
        if (_vel_mag(ev) or 0) < 0.5:
            continue
        _, lat, ok = _vel_relative_to_ego(nv1, ev, plane)
        if ok:
            lat_abs = abs(lat)
            if lat_abs > peak:
                peak = lat_abs
    return peak


def critical_frame_cutout(frames, plane='XZ', collision_frame=None):
    """
    Pick the cut-out 'moment of revelation' frame: the first frame at which
    BOTH npc1 has started its lateral exit (lateral velocity above threshold)
    AND npc2 is visible (ahead of ego). If a collision is provided, the
    search is bounded to pre-collision frames.
    Returns (frame_idx, t_blackout_estimate).
    """
    upper = collision_frame if collision_frame is not None else len(frames)
    lat_threshold = 0.3   # m/s, occluder lateral velocity considered "exiting"

    # First, find when npc1 starts exiting laterally (within pre-collision range)
    npc1_exit_idx = None
    npc1_exit_time = None
    for i in range(upper):
        fr = frames[i]
        ev = (fr.get('ego') or {}).get('velocity')
        nv1 = (fr.get('npc1') or {}).get('velocity')
        if ev is None or nv1 is None: continue
        if (_vel_mag(ev) or 0) < 0.5: continue
        _, lat, ok = _vel_relative_to_ego(nv1, ev, plane)
        if ok and abs(lat) >= lat_threshold:
            npc1_exit_idx = i
            npc1_exit_time = fr.get('timestamp')
            break

    # Second, find first frame where npc2 is visible and ahead of ego
    npc2_visible_idx = None
    npc2_visible_time = None
    for i in range(upper):
        fr = frames[i]
        npc2 = fr.get('npc2')
        if not npc2 or npc2.get('position') is None: continue
        ev = (fr.get('ego') or {}).get('velocity')
        ep = _pos_of(fr, 'ego')
        if ev is None or ep is None: continue
        if (_vel_mag(ev) or 0) < 0.5: continue
        long_gap, _, ok = project_onto_heading(ep, npc2['position'], ev, plane)
        if ok and long_gap > 0:
            npc2_visible_idx = i
            npc2_visible_time = fr.get('timestamp')
            break

    # Use the LATER of the two: the cut-out "happens" when both occur.
    # In real scenarios these are typically simultaneous (the occluder
    # pulling away IS what makes npc2 visible).
    if npc1_exit_idx is not None and npc2_visible_idx is not None:
        critical_idx = max(npc1_exit_idx, npc2_visible_idx)
        critical_time = frames[critical_idx].get('timestamp')
    elif npc2_visible_idx is not None:
        critical_idx = npc2_visible_idx
        critical_time = npc2_visible_time
    elif npc1_exit_idx is not None:
        critical_idx = npc1_exit_idx
        critical_time = npc1_exit_time
    else:
        return None, 0.0

    # t_blackout (data-grounded): the duration npc2 is RELEVANT -- present,
    # ground-truth ahead of ego, within range -- but has NO perception
    # detection within a matching gate, ending at its first clean detection.
    # This is the run of MISSING DETECTIONS preceding the reveal, i.e. the
    # blackout the perception system actually experienced (cf.
    # blackout_measurement.py), rather than a ground-truth-geometry estimate.
    # No artificial clamp: a long blackout is real and simply saturates
    # v_eff_rev to the "assume stopped" regime. The 5 m gate sits well above
    # the localisation offset (~1.37 m), so this detection-gap measurement is
    # independent of the bias.
    GATE = 5.0            # m, detection matching gate
    REL_RANGE = 120.0     # m, range within which npc2 is a relevant threat
    first_relevant_t = None
    last_relevant_t = None
    first_detect_t = None
    for i in range(upper):
        fr = frames[i]
        npc2 = fr.get('npc2')
        if not npc2 or npc2.get('position') is None:
            continue
        ev = (fr.get('ego') or {}).get('velocity')
        ep = _pos_of(fr, 'ego')
        if ev is None or ep is None or (_vel_mag(ev) or 0) < 0.5:
            continue
        long_gap, _, ok = project_onto_heading(ep, npc2['position'], ev, plane)
        if not ok or long_gap <= 0 or long_gap > REL_RANGE:
            continue
        ts = fr.get('timestamp')
        if first_relevant_t is None:
            first_relevant_t = ts
        last_relevant_t = ts
        perc = _pick_perceived_match(fr, npc2['position'], plane)
        if (perc is not None and perc.get('position') is not None and
                dist_2d(perc['position'], npc2['position'], plane) <= GATE):
            first_detect_t = ts
            break
    if first_relevant_t is not None and first_detect_t is not None:
        # detected while relevant: blackout is the gap before the reveal
        t_blackout = max(0.0, first_detect_t - first_relevant_t)
    elif first_relevant_t is not None:
        # never cleanly detected throughout the relevant window: full
        # occlusion. This is large and saturates v_eff_rev to "assume
        # stopped", the correct worst-case treatment of an unseen lead.
        t_blackout = max(0.0, last_relevant_t - first_relevant_t)
    else:
        t_blackout = 0.0

    return critical_idx, t_blackout


# -------------------------------------------------------------------------
# Main per-file processing
# -------------------------------------------------------------------------
def _ego_speed_series(frames, collision_frame=None):
    """Return parallel lists (timestamps, speeds) of the ego up to and
    including the collision frame (or the whole trajectory)."""
    end = len(frames)
    if collision_frame is not None and 0 <= collision_frame < len(frames):
        end = collision_frame + 1
    ts, sp = [], []
    for i in range(end):
        ego = frames[i].get('ego') or {}
        v = ego.get('velocity')
        t = frames[i].get('timestamp')
        s = _vel_mag(v)
        if t is None or s is None:
            continue
        ts.append(float(t))
        sp.append(float(s))
    return ts, sp


def _accel_at(ts, sp, k, min_dt):
    """Forward (or backward at the end) finite-difference acceleration at
    index k, over a window of at least min_dt seconds to suppress jitter."""
    n = len(ts)
    j = k + 1
    while j < n and (ts[j] - ts[k]) < min_dt:
        j += 1
    if j < n:
        dt = ts[j] - ts[k]
        return (sp[j] - sp[k]) / dt if dt > 0 else None
    # near the end: use a backward window
    j2 = k - 1
    while j2 >= 0 and (ts[k] - ts[j2]) < min_dt:
        j2 -= 1
    if j2 < 0:
        return None
    dt = ts[k] - ts[j2]
    return (sp[k] - sp[j2]) / dt if dt > 0 else None


def windowed_prebrake_ego_acceleration(frames, collision_frame=None,
                                        delta_sys=0.7, brake_thresh=0.5,
                                        min_dt=0.1):
    """Maximum FORWARD ego acceleration in the latency window immediately
    BEFORE the ego begins its final braking, in m/s^2.

    This is the empirical analogue of alpha_max as the model actually uses
    it: the acceleration the ego sustains during the delta_sys reaction
    window (Phase A), just before it reacts to the hazard. Unlike
    peak_ego_acceleration (the trajectory-wide peak, which can include
    initial speed-up far from any hazard), this isolates the moments that
    feed into Phase A.

    Method: locate the onset of the final sustained braking phase before
    the critical/collision frame (the last run of frames with
    deceleration steeper than brake_thresh, walked back to its start),
    then take the maximum forward acceleration in the window
    [t_brake - delta_sys, t_brake]. Returns None if it cannot be computed.
    """
    ts, sp = _ego_speed_series(frames, collision_frame)
    n = len(ts)
    if n < 3:
        return None

    # Find the last braking frame (closest to the event), then walk back
    # to the onset of that braking phase.
    brake_idx = None
    for k in range(n - 1, -1, -1):
        a = _accel_at(ts, sp, k, min_dt)
        if a is not None and a < -brake_thresh:
            brake_idx = k
            break

    if brake_idx is None:
        # Ego never brakes hard before the event: the Phase-A window is the
        # delta_sys ending at the critical frame itself.
        t_brake = ts[-1]
    else:
        onset = brake_idx
        while onset - 1 >= 0:
            a = _accel_at(ts, sp, onset - 1, min_dt)
            if a is not None and a < -brake_thresh:
                onset -= 1
            else:
                break
        t_brake = ts[onset]

    lo = t_brake - delta_sys - 1e-9
    peak = None
    for k in range(n):
        if ts[k] < lo:
            continue
        if ts[k] > t_brake + 1e-9:
            break
        a = _accel_at(ts, sp, k, min_dt)
        if a is None:
            continue
        if peak is None or a > peak:
            peak = a
    if peak is None:
        return None
    return max(0.0, peak)


def peak_ego_acceleration(frames, collision_frame=None, min_dt=0.1):
    """Maximum FORWARD (positive) ego acceleration observed over the
    pre-collision portion of the trajectory, in m/s^2.

    This is the empirical quantity to compare against the assumed
    worst-case pre-brake acceleration alpha_max. Acceleration is computed
    from the scalar ego speed (velocity 'magnitude') differenced over a
    short window (>= min_dt seconds) to suppress single-frame finite-
    difference jitter; we then take the maximum positive value.

    Only frames up to `collision_frame` (inclusive) are considered when a
    collision frame is given, so post-impact dynamics do not contaminate
    the estimate. Returns None if it cannot be computed.
    """
    if not frames:
        return None
    end = len(frames)
    if collision_frame is not None and 0 <= collision_frame < len(frames):
        end = collision_frame + 1

    # Build (timestamp, speed) series
    ts, sp = [], []
    for i in range(end):
        fr = frames[i]
        ego = fr.get('ego') or {}
        v = ego.get('velocity')
        t = fr.get('timestamp')
        s = _vel_mag(v)
        if t is None or s is None:
            continue
        ts.append(float(t))
        sp.append(float(s))
    if len(ts) < 2:
        return None

    peak = None
    j = 0
    for i in range(len(ts)):
        # advance j until the time separation is at least min_dt
        if j <= i:
            j = i + 1
        while j < len(ts) and (ts[j] - ts[i]) < min_dt:
            j += 1
        if j >= len(ts):
            break
        dt = ts[j] - ts[i]
        if dt <= 0:
            continue
        a = (sp[j] - sp[i]) / dt
        if peak is None or a > peak:
            peak = a
    # Only positive (forward) acceleration is meaningful for alpha_max;
    # clamp a negative peak (pure braking throughout) to 0.0.
    if peak is None:
        return None
    return max(0.0, peak)


def _resolve_snapshot_idx(frames, critical_idx, anchor='critical',
                          lead_seconds=0.0, collision_frame=None):
    """Choose which frame to snapshot.

    anchor selects the reference event:
      'critical'  -- most safety-critical frame (default; original behaviour)
      'collision' -- frame where the collision is first detected
      'end'       -- last frame of the trajectory (goal / scenario end)
    lead_seconds (>=0) then steps BACK that many seconds from the anchor, landing
    on the closest frame at or before (anchor_time - lead_seconds). lead_seconds=0
    uses the anchor frame itself. Falls back to the critical frame whenever the
    requested anchor is unavailable (e.g. 'collision' on a no-collision run).
    """
    if anchor == 'collision' and collision_frame is not None:
        anchor_idx = collision_frame
    elif anchor == 'end':
        anchor_idx = len(frames) - 1
    else:
        anchor_idx = critical_idx
    if anchor_idx is None:
        anchor_idx = critical_idx
    if anchor_idx is None:
        return None
    if not lead_seconds or lead_seconds <= 0:
        return anchor_idx
    t_anchor = frames[anchor_idx].get('timestamp')
    if t_anchor is None:
        return anchor_idx
    target_t = t_anchor - lead_seconds
    j = anchor_idx
    while j > 0:
        tj = frames[j].get('timestamp')
        if tj is not None and tj <= target_t:
            break
        j -= 1
    return j


def process_json_file(path, mode, every, scenario, collision_label,
                       fname_for_label, anchor='critical', lead_seconds=0.0):
    with open(path) as f:
        d = json.load(f)
    frames = d.get('frames') or []
    if not frames:
        print(f"  [warn] no frames in {path}", file=sys.stderr)
        return [], None

    # Auto-detect scenario if requested
    if scenario == 'auto':
        scenario = detect_scenario(fname_for_label)
        if scenario is None:
            print(f"  [warn] {fname_for_label}: could not detect scenario from name",
                  file=sys.stderr)
            return [], None

    # Coordinate plane convention
    plane = 'XZ' if scenario == 'cutout' else 'XY'

    # Parse filename parameter
    npc_param = parse_npc_param_from_filename(fname_for_label, scenario)

    # Auto-detect collision from trajectory if no explicit label given.
    # Any label provided (via --labels CSV or --collision-label) takes
    # precedence over the auto-detected value.
    auto_collision_info = None
    if collision_label is None:
        auto_collision_info = detect_collision_in_trajectory(frames, plane)
        collision_label = 1 if auto_collision_info['collided'] else 0

    # Compute trajectory-level perception coverage statistics
    perc_stats = compute_perception_stats(frames)

    # Precompute ground-truth position histories for finite-difference
    # perceived-velocity reconstruction.
    gt_hist_npc1 = _build_gt_position_history(frames, 'npc1')
    gt_hist_npc2 = (_build_gt_position_history(frames, 'npc2')
                     if scenario == 'cutout' else None)

    rows = []
    if scenario == 'deceleration':
        if mode == 'snapshot':
            # Pass the auto-detected collision frame (if any) so the snapshot
            # is taken at or before the collision, never after.
            coll_frame = (auto_collision_info or {}).get('collision_frame')
            idx = critical_frame_deceleration(frames, plane, coll_frame)
            if idx is None:
                print(f"  [warn] {fname_for_label}: no critical frame",
                      file=sys.stderr); return [], None
            idx = _resolve_snapshot_idx(frames, idx, anchor, lead_seconds, coll_frame)
            r = build_deceleration_row(frames[idx], fname_for_label,
                                        collision_label, npc_param, plane,
                                        perc_stats,
                                        frames=frames, frame_idx=idx,
                                        gt_pos_history=gt_hist_npc1)
            if r: rows.append(r)
        else:
            stride = max(1, int(every)) if mode == 'downsample' else 1
            for i in range(0, len(frames), stride):
                r = build_deceleration_row(frames[i], fname_for_label,
                                            collision_label, npc_param, plane,
                                            perc_stats,
                                            frames=frames, frame_idx=i,
                                            gt_pos_history=gt_hist_npc1)
                if r: rows.append(r)

    elif scenario == 'cutin':
        if mode == 'snapshot':
            coll_frame = (auto_collision_info or {}).get('collision_frame')
            idx = critical_frame_cutin(frames, plane, coll_frame)
            if idx is None:
                print(f"  [warn] {fname_for_label}: no cut-in frame",
                      file=sys.stderr); return [], None
            idx = _resolve_snapshot_idx(frames, idx, anchor, lead_seconds, coll_frame)
            r = build_cutin_row(frames[idx], fname_for_label,
                                  collision_label, npc_param, plane,
                                  perc_stats,
                                  frames=frames, frame_idx=idx,
                                  gt_pos_history=gt_hist_npc1)
            if r: rows.append(r)
        else:
            stride = max(1, int(every)) if mode == 'downsample' else 1
            for i in range(0, len(frames), stride):
                r = build_cutin_row(frames[i], fname_for_label,
                                      collision_label, npc_param, plane,
                                      perc_stats,
                                      frames=frames, frame_idx=i,
                                      gt_pos_history=gt_hist_npc1)
                if r: rows.append(r)

    elif scenario == 'cutout':
        if mode == 'snapshot':
            coll_frame = (auto_collision_info or {}).get('collision_frame')
            idx, t_blackout = critical_frame_cutout(frames, plane, coll_frame)
            if idx is None:
                print(f"  [warn] {fname_for_label}: no cut-out moment found",
                      file=sys.stderr); return [], None
            idx = _resolve_snapshot_idx(frames, idx, anchor, lead_seconds, coll_frame)
            peak_lat = peak_npc1_lateral_velocity(frames, plane, coll_frame)
            r = build_cutout_row(frames[idx], fname_for_label,
                                   collision_label, npc_param,
                                   t_blackout, plane, perc_stats,
                                   frames=frames, frame_idx=idx,
                                   gt_pos_history_npc1=gt_hist_npc1,
                                   gt_pos_history_npc2=gt_hist_npc2,
                                   v_y_occ_peak=peak_lat)
            if r: rows.append(r)
        else:
            # All / downsample: compute t_blackout once, reuse
            _, t_blackout = critical_frame_cutout(frames, plane)
            peak_lat = peak_npc1_lateral_velocity(frames, plane)
            stride = max(1, int(every)) if mode == 'downsample' else 1
            for i in range(0, len(frames), stride):
                r = build_cutout_row(frames[i], fname_for_label,
                                       collision_label, npc_param,
                                       t_blackout, plane, perc_stats,
                                       frames=frames, frame_idx=i,
                                       gt_pos_history_npc1=gt_hist_npc1,
                                       gt_pos_history_npc2=gt_hist_npc2,
                                       v_y_occ_peak=peak_lat)
                if r: rows.append(r)
    else:
        print(f"  [warn] unknown scenario: {scenario}", file=sys.stderr)

    # Stamp the file-level peak forward ego acceleration onto every row.
    # This is a trajectory-level quantity (the worst-case observed ego
    # acceleration before impact), so it is identical across the rows
    # produced from a single file. It lets bound_consistency.py check the
    # assumed alpha_max against what the ego actually did.
    coll_frame_for_accel = (auto_collision_info or {}).get('collision_frame')
    ego_peak_acc = peak_ego_acceleration(frames, coll_frame_for_accel)
    if ego_peak_acc is not None:
        for r in rows:
            r['ego_max_accel'] = round(ego_peak_acc, 4)
    ego_prebrake_acc = windowed_prebrake_ego_acceleration(
        frames, coll_frame_for_accel)
    if ego_prebrake_acc is not None:
        for r in rows:
            r['ego_accel_prebrake'] = round(ego_prebrake_acc, 4)

    return rows, auto_collision_info


# -------------------------------------------------------------------------
# CSV writing
# -------------------------------------------------------------------------
CSV_COLUMNS = [
    'scenario',
    'p_t', 'u_t', 'w_t',
    'p_long_t', 'v_y_rel', 'w_cut', 'd_y',
    'p_rev', 'v_y_occ', 'v_rev_hat', 't_blackout', 'd_y_occ',
    'collision_occurred',
    'ground_truth_dist', 'ground_truth_vel',
    'ground_truth_v_y_rel', 'ground_truth_d_y','ground_truth_v_y_occ',
    'v_y_occ_peak',
    'npc_actual_decel', 'ego_max_accel', 'ego_accel_prebrake', 'filename_param',
    'perception_detected', 'inputs_were_groundtruth',
    'perception_coverage', 'perception_max_gap_s', 'staleness_at_decision_s',
    'label',
]


def write_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main():
    global COLLISION_OVERLAP_M, EPS_BIAS
    p = argparse.ArgumentParser(
        description="Convert simulation JSONs to PCB-overlay CSV.")
    p.add_argument("input", help=".json file OR directory of .json files")
    p.add_argument("output_csv", help="path to write the CSV")
    p.add_argument("--mode", choices=['snapshot', 'all', 'downsample'],
                   default='snapshot')
    p.add_argument("--every", type=int, default=20)
    p.add_argument("--scenario",
                   choices=['auto', 'deceleration', 'cutin', 'cutout'],
                   default='auto')
    p.add_argument("--collision-label", type=int, default=None)
    p.add_argument("--collision-overlap-m", type=float, default=COLLISION_OVERLAP_M,
                   help="minimum bounding-box penetration depth (m) to count "
                        "as a collision. Default 0.01 (1 cm) ignores "
                        "sub-millimetre numerical-touch artefacts. Set to 0 "
                        "to flag any overlap including touches.")
    p.add_argument("--labels", default=None,
                   help="CSV mapping filename->collision_occurred")
    p.add_argument("--anchor", choices=['critical', 'collision', 'end'],
                   default='critical',
                   help="snapshot-mode reference event: 'critical' (default, "
                        "most safety-critical frame), 'collision' (collision "
                        "frame), or 'end' (last frame / goal end).")
    p.add_argument("--lead-seconds", type=float, default=0.0,
                   help="snapshot-mode seconds BEFORE --anchor to sample "
                        "(0 = the anchor frame itself). E.g. --anchor collision "
                        "--lead-seconds 1.5 samples 1.5 s before impact.")
    p.add_argument("--eps-bias", type=float, default=None,
                   help="Systematic perception offset b_eps (m) to remove from "
                        "perceived longitudinal gaps (bias-corrected run). "
                        "Use 1.37 for the corrected evaluation; omit (or 0) for "
                        "the uncorrected baseline. Applied only to real "
                        "perceptions, never to ground-truth fallbacks.")
    args = p.parse_args()

    # Apply CLI override for collision overlap threshold
    COLLISION_OVERLAP_M = args.collision_overlap_m

    # Apply CLI override for the perception bias offset (falls back to the
    # module default EPS_BIAS when the flag is not given).
    if args.eps_bias is not None:
        EPS_BIAS = args.eps_bias
    if EPS_BIAS > 0.0:
        print(f"[bias-corrected] debiasing perceived gaps by b_eps = "
              f"{EPS_BIAS:.3f} m (real perceptions only)")

    label_map = {}
    if args.labels:
        with open(args.labels) as lf:
            for r in csv.DictReader(lf):
                fn = (r.get('filename') or r.get('file') or '').strip()
                lv = r.get('collision_occurred') or r.get('label') or ''
                if fn and lv != '':
                    label_map[fn] = int(lv)

    in_path = Path(args.input)
    if in_path.is_dir():
        files = sorted(in_path.glob("*_data.json")) or sorted(in_path.glob("*.json"))
    else:
        files = [in_path]
    if not files:
        print(f"No JSON files found at {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(files)} file(s) in mode '{args.mode}' ...")
    all_rows = []
    n_collided, n_safe, n_rear = 0, 0, 0
    for fp in files:
        basename = fp.name
        clabel = label_map.get(basename, args.collision_label)
        rows, coll_info = process_json_file(
            fp, args.mode, args.every, args.scenario, clabel, basename,
            anchor=args.anchor, lead_seconds=args.lead_seconds)
        all_rows.extend(rows)
        scen_detected = detect_scenario(basename) if args.scenario == 'auto' else args.scenario

        # Build a one-line summary including auto-detected collision
        coll_str = ""
        if coll_info is not None:
            if coll_info['collided']:
                coll_str = (f"  COLLIDED@t={coll_info['collision_time']:.2f}"
                            f" with {coll_info['npc_hit']}"
                            f" (min_gap was {coll_info['min_gap_m']:.2f}m)")
                n_collided += 1
            elif coll_info.get('rear_collision'):
                # NPC drove into the back of the ego -- this is the NPC's
                # fault and is not counted as an ego safety failure.
                coll_str = (f"  rear-collision-only@t="
                            f"{coll_info['rear_collision_time']:.2f}"
                            f"  (excluded -- NPC into ego rear)")
                n_safe += 1
                n_rear += 1
            else:
                coll_str = (f"  no-collision  (min_gap={coll_info['min_gap_m']:.2f}m)"
                            if coll_info['min_gap_m'] is not None else "")
                n_safe += 1
        elif clabel is not None:
            coll_str = f"  label={clabel} (provided)"

        print(f"  {basename}: scenario={scen_detected}, rows={len(rows)}{coll_str}")

    write_csv(all_rows, args.output_csv)
    print(f"\nWrote {len(all_rows)} rows to {args.output_csv}")
    if n_collided + n_safe > 0:
        print(f"Auto-detected collisions: {n_collided} forward-collided / "
              f"{n_safe} safe (of {n_collided + n_safe} auto-labeled).")
        if n_rear > 0:
            print(f"  ({n_rear} of those 'safe' had rear-only collisions "
                  f"excluded -- NPC drove into ego's back, treated as "
                  f"no-fault to the ego)")


if __name__ == "__main__":
    main()