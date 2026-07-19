"""
perception_characterization.py
==============================

Dedicated quantitative justification of the perception error budgets
(eps_max, eps_y_max, gamma_max, gamma_y_max).

For each perception parameter this tool:
  1. measures the per-frame error (perceived - true) in the dangerous
     direction, directly from perceived-vs-ground-truth pairs in the JSON;
  2. estimates the SYSTEMATIC BIAS b (the mean error) -- the candidate
     "environmental / calibration parameter" of the AI model;
  3. forms the RESIDUAL NOISE  nu = error - b ;
  4. reports the exceedance of the assumed bound BEFORE and AFTER
     compensating the bias, with full percentiles; and
  5. gives an honest per-parameter verdict:
        - BIAS-DOMINATED  : compensating b brings the residual within the
                            bound (the exceedance was a removable offset);
        - TAIL-DOMINATED  : the median is within the bound but a heavy tail
                            remains after compensation (a transient
                            perception/estimation artefact, not a bias);
        - WITHIN-BOUND    : the raw error already respects the bound.

This is the analysis to back the claim "once the systematic factor is
represented, the observations fall within the assumed bound" with numbers,
and to show honestly where that claim does NOT hold (the residual tail).

The model interpretation it supports:
        perceived = true + b + nu ,     |nu| <= budget
with b an explicit environmental parameter and the budget bounding the
residual noise nu.

USAGE
-----
    # one or more folders / globs / files of *_data.json
    python perception_characterization.py /path/to/json_folder
    python perception_characterization.py folderA folderB "globC/*.json"

    # estimate the bias PER SCENARIO (file) instead of one global bias
    python perception_characterization.py /path/folder --per-scenario-bias

    # override the assumed bounds being tested
    python perception_characterization.py /path/folder \
        --eps 1.0 --eps-y 0.3 --gamma 1.5 --gamma-y 0.5

    # figure output (4-panel raw-vs-residual histograms)
    python perception_characterization.py /path/folder --out perception_char.png

    # text report only
    python perception_characterization.py /path/folder --no-plot

DEPENDENCIES:  numpy, matplotlib   (no project modules)

NOTE ON SIGN CONVENTIONS (dangerous direction). These are encoded per
parameter in PARAM_SPECS below and explained there; confirm they match the
direction your d_PRSS actually uses before quoting the numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np


# ---------------------------------------------------------------------
# Geometry helpers (self-contained; mirror project_onto_heading)
# ---------------------------------------------------------------------
def _xy(p, plane="xy"):
    a, b = plane[0], plane[1]
    return float(p[a]), float(p[b])


def _heading(ego_vel, plane="xy"):
    vh, vl = _xy(ego_vel, plane)
    speed = math.hypot(vh, vl)
    if speed < 0.2:
        return None
    return (vh / speed, vl / speed)


def decompose(vec_h, vec_l, fwd):
    """Decompose a 2D vector into (longitudinal, lateral) given a forward
    unit vector fwd = (fh, fl). Lateral unit = (-fl, fh)."""
    fh, fl = fwd
    longitudinal = vec_h * fh + vec_l * fl
    lateral = vec_h * (-fl) + vec_l * fh
    return longitudinal, lateral


def _vel_relative_to_ego(target_vel, ego_vel, plane="xy"):
    """Return (forward, lateral, heading_valid) in the ego frame.

    This mirrors the frame-aware decomposition used by the CSV extraction
    pipeline, but stays self-contained so the script can run on raw JSON.
    """
    fwd = _heading(ego_vel, plane)
    if fwd is None:
        return None, None, False
    tvh, tvl = _xy(target_vel, plane)
    fh, fl = fwd
    forward = tvh * fh + tvl * fl
    lateral = tvh * (-fl) + tvl * fh
    return forward, lateral, True


# ---------------------------------------------------------------------
# Per-frame perception errors
# ---------------------------------------------------------------------
GATED = object()   # sentinel: nearest detection too far -> mis-association


def frame_errors(frame, plane="xy", match_gate=5.0):
    """Return dict of perception errors for one frame, None where not
    computable, or GATED when the nearest detection is farther than
    match_gate from the true NPC (a detection/association failure, not a
    localization error). Errors are signed in the DANGEROUS direction."""
    ego = frame.get("ego") or {}
    npc = frame.get("npc1") or {}
    ep, ev = ego.get("position"), ego.get("velocity")
    tp, tv = npc.get("position"), npc.get("velocity")
    pobjs = frame.get("perception_objects") or []
    if not (ep and ev and tp and pobjs):
        return None
    fwd = _heading(ev, plane)
    if fwd is None:                       # ego heading undefined
        return None

    # nearest perception object to the TRUE npc position
    best, bd = None, 1e9
    for po in pobjs:
        pp = po.get("position")
        if not pp:
            continue
        d = math.hypot(pp[plane[0]] - tp[plane[0]], pp[plane[1]] - tp[plane[1]])
        if d < bd:
            bd, best = d, po
    if best is None or not best.get("position"):
        return None
    if bd > match_gate:
        return GATED          # nearest detection too far: not a localization
    pp = best["position"]
    pv = best.get("velocity")

    eh, el = _xy(ep, plane)
    th, tl = _xy(tp, plane)
    ph, pl = _xy(pp, plane)

    # ---- position errors ----
    t_long, t_lat = decompose(th - eh, tl - el, fwd)
    p_long, p_lat = decompose(ph - eh, pl - el, fwd)
    out = {
        # longitudinal position: dangerous + = perceives FARTHER than true
        "eps_max": p_long - t_long,
        # lateral position: dangerous + = perceives MORE lateral gap than true
        "eps_y_max": abs(p_lat) - abs(t_lat),
    }

    # ---- velocity error (SPEED MAGNITUDE; frame-invariant) ----
    # The logged velocities use source 'twist' (object body frame), which
    # cannot be reliably decomposed into ego-frame longitudinal/lateral
    # components without each object's orientation -- doing so naively
    # injects a large spurious bias. The SPEED MAGNITUDE is frame-invariant,
    # so gamma_max is characterized as the error in perceived speed. The
    # lateral velocity budget gamma_y_max needs the frame-aware decomposition
    # already implemented in json_to_csv.py / bound_consistency.py and is
    # intentionally NOT recomputed here.
    tm = tv.get("magnitude") if isinstance(tv, dict) else None
    pm = pv.get("magnitude") if isinstance(pv, dict) else None
    if tm is not None and pm is not None:
        # dangerous + = perceives lead FASTER than true (less braking credited)
        out["gamma_max"] = float(pm) - float(tm)
    else:
        out["gamma_max"] = None

    # ---- lateral velocity error (frame-aware magnitude) ----
    # Mirrors the CSV pipeline: the cut-in / cut-out budgets use the
    # magnitude of the lateral component relative to ego heading.
    _, t_lat, ok_t = _vel_relative_to_ego(tv, ev, plane)
    if ok_t and pv and 'y' in pv:
        # pv is already in ego frame, use y-component directly (no decomposition)
        # dangerous + = perceives more lateral closing / exit speed
        out["gamma_y_max"] = abs(float(pv['y'])) - abs(float(t_lat))
    else:
        out["gamma_y_max"] = None
    return out


PARAM_SPECS = [
    ("eps_max",   "longitudinal position error (perceived - true)", "m",
     "dangerous + = perceives farther"),
    ("eps_y_max", "lateral position error (|perc| - |true|)",       "m",
     "dangerous + = perceives more lateral gap"),
    ("gamma_max", "speed error (perceived - true |v|)",             "m/s",
     "dangerous + = perceives lead faster; frame-invariant magnitude"),
    ("gamma_y_max", "lateral speed error (|perc| - |true|)",        "m/s",
     "dangerous + = perceives more lateral closing / exit speed"),
]


# ---------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------
def find_frames(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("frames", "data", "timesteps", "steps"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        for v in obj.values():
            r = find_frames(v)
            if r:
                return r
    return None


def discover_files(inputs):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            files.extend(sorted(glob.glob(os.path.join(inp, "*_data.json"))))
            files.extend(sorted(glob.glob(os.path.join(inp, "*.json"))))
        elif any(c in inp for c in "*?["):
            files.extend(sorted(glob.glob(inp)))
        else:
            files.append(inp)
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def label_for(path):
    b = os.path.basename(path)
    for suf in ("_data.json", ".json"):
        if b.endswith(suf):
            return b[: -len(suf)]
    return b


# ---------------------------------------------------------------------
# Stats + verdict
# ---------------------------------------------------------------------
def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def exceed_frac(err, bound):
    """Fraction exceeding the bound in the dangerous (+) direction."""
    return float(np.mean(err > bound)) if len(err) else float("nan")


def verdict(raw, residual, bound):
    raw_ex = exceed_frac(raw, bound)
    res_ex = exceed_frac(residual, bound)
    if raw_ex <= 0.05:
        return "WITHIN-BOUND", raw_ex, res_ex
    if res_ex <= 0.05:
        return "BIAS-DOMINATED", raw_ex, res_ex
    return "TAIL-DOMINATED", raw_ex, res_ex


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Characterize perception error as bias + bounded noise, "
                    "and test bound-consistency before/after compensation.")
    ap.add_argument("inputs", nargs="+", help="Folders, globs, or files.")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--eps-y", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=1.5)
    ap.add_argument("--gamma-y", type=float, default=0.5)
    ap.add_argument("--per-scenario-bias", action="store_true",
                    help="Estimate the bias per source file rather than one "
                         "global bias.")
    ap.add_argument("--plane", default="xy")
    ap.add_argument("--match-gate", type=float, default=5.0,
                    help="Max distance (m) between the nearest detection and "
                         "the true NPC for the frame to count as a localization "
                         "(else it is a mis-association/non-detection).")
    ap.add_argument("--out", default="perception_characterization.png")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    bounds = {"eps_max": args.eps, "eps_y_max": args.eps_y,
              "gamma_max": args.gamma, "gamma_y_max": args.gamma_y}

    files = discover_files(args.inputs)
    if not files:
        print("No JSON files found.", file=sys.stderr)
        return 1

    # collect per-file lists per parameter
    per_file = {p: {} for p, *_ in PARAM_SPECS}
    n_frames = 0
    n_gated = 0
    for path in files:
        try:
            with open(path) as fh:
                frames = find_frames(json.load(fh))
        except Exception as e:
            print(f"  [skip] {path}: {e}", file=sys.stderr)
            continue
        if not frames:
            continue
        lbl = label_for(path)
        for fr in frames:
            errs = frame_errors(fr, args.plane, match_gate=args.match_gate)
            if errs is None:
                continue
            if errs is GATED:
                n_gated += 1
                continue
            n_frames += 1
            for p in per_file:
                v = errs.get(p)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    per_file[p].setdefault(lbl, []).append(v)

    # build raw + residual arrays per parameter. Use the MEDIAN as the bias
    # estimator -- it is robust to the mis-association outliers that the gate
    # does not catch; the mean is reported alongside for transparency.
    results = {}
    for p, desc, unit, note in PARAM_SPECS:
        files_d = {k: np.array(v) for k, v in per_file[p].items() if len(v)}
        if not files_d:
            continue
        raw = np.concatenate(list(files_d.values()))
        if args.per_scenario_bias:
            bias_used = {k: float(np.median(v)) for k, v in files_d.items()}
            residual = np.concatenate([v - bias_used[k]
                                       for k, v in files_d.items()])
            bias_report = ("per-scenario median (avg "
                           f"{np.mean(list(bias_used.values())):+.3f})")
        else:
            b = float(np.median(raw))
            residual = raw - b
            bias_report = f"{b:+.3f}  (mean {np.mean(raw):+.3f})"
        results[p] = dict(raw=raw, residual=residual, bias=bias_report,
                          desc=desc, unit=unit, note=note,
                          bound=bounds[p], files=files_d)

    # ---- text report ----
    print("=" * 74)
    print("PERCEPTION CHARACTERIZATION  (bias + bounded-noise model)")
    print("=" * 74)
    print(f"frames used: {n_frames}   scenarios: {len(files)}")
    _tot = n_frames + n_gated
    if _tot:
        print(f"mis-associated/non-detection frames gated out: {n_gated} "
              f"({100*n_gated/_tot:.1f}% of detections, gate "
              f"{args.match_gate} m) -- a separate detection-failure mode, "
              f"not a localization error")
    print(f"bias estimator: "
          f"{'per-scenario median' if args.per_scenario_bias else 'global median'}")
    print("model:  perceived = true + b + nu,   bound tests |nu| (residual)")
    print()
    for p, desc, unit, note in PARAM_SPECS:
        if p not in results:
            print(f"[{p}] no data\n")
            continue
        r = results[p]
        raw, res, bnd = r["raw"], r["residual"], r["bound"]
        v, raw_ex, res_ex = verdict(raw, res, bnd)
        print("-" * 74)
        print(f"{p}  (bound = {bnd} {unit})   {desc}")
        print(f"    dangerous direction: {note}")
        print(f"    n = {len(raw)}   estimated bias b = {r['bias']} {unit}")
        print(f"    RAW error      : mean {np.mean(raw):+.3f}  "
              f"p50 {pct(raw,50):+.3f}  p95 {pct(raw,95):+.3f}  "
              f"p99 {pct(raw,99):+.3f}  max {np.max(raw):+.3f}")
        print(f"      exceed (raw  > {bnd}): {100*raw_ex:5.1f}%")
        print(f"    RESIDUAL nu    : mean {np.mean(res):+.3f}  "
              f"p50 {pct(res,50):+.3f}  p95 {pct(res,95):+.3f}  "
              f"p99 {pct(res,99):+.3f}  max {np.max(res):+.3f}")
        print(f"      exceed (|nu| > {bnd}): "
              f"{100*np.mean(np.abs(res) > bnd):5.1f}%   "
              f"(dangerous-side {100*res_ex:.1f}%)")
        print(f"    VERDICT: {v}")
        if v == "BIAS-DOMINATED":
            print(f"      -> exceedance {100*raw_ex:.0f}% -> {100*res_ex:.0f}% "
                  f"after representing b; adopt b = {r['bias']} {unit} as an "
                  f"environmental parameter, bound holds for the noise.")
        elif v == "TAIL-DOMINATED":
            print(f"      -> median within bound but a heavy tail remains after "
                  f"compensation; NOT a removable bias (transient artefact).")
        print()

    print("-" * 74)
    # ---- figure ----
    if not args.no_plot and results:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[plot skipped: {e}]", file=sys.stderr)
            return 0
        keys = [p for p, *_ in PARAM_SPECS if p in results]
        ncol = 2
        nrow = int(math.ceil(len(keys) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 4.4 * nrow))
        axes = np.array(axes).reshape(-1)
        for ax, p in zip(axes, keys):
            r = results[p]
            raw, res, bnd, unit = r["raw"], r["residual"], r["bound"], r["unit"]
            lo = min(raw.min(), res.min())
            hi = max(raw.max(), res.max())
            bins = np.linspace(lo, hi, 60)
            ax.hist(raw, bins=bins, alpha=0.5, color="#c44", label="raw error")
            ax.hist(res, bins=bins, alpha=0.6, color="#27a", label="residual $\\nu$")
            ax.axvline(bnd, color="#222", ls="--", lw=1.2,
                       label=f"+bound {bnd} {unit}")
            ax.axvline(-bnd, color="#222", ls="--", lw=1.2)
            ax.axvline(0, color="#888", lw=0.6)
            v, rex, sex = verdict(raw, res, bnd)
            ax.set_title(f"{p}  ({v})\nbias {r['bias']} {unit};  "
                         f"exceed {100*rex:.0f}% \u2192 {100*sex:.0f}%",
                         fontsize=10)
            ax.set_xlabel(f"error ({unit})")
            ax.set_ylabel("frames")
            ax.legend(fontsize=7)
        for ax in axes[len(keys):]:
            ax.axis("off")
        fig.suptitle("Perception error characterized as bias + bounded noise",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(args.out, dpi=140, bbox_inches="tight")
        print(f"figure -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())