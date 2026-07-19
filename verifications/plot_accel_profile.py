"""
plot_accel_profiles.py
======================

Plot the ego's acceleration -> deceleration profile over time for each
scenario, one line per scenario, all on one graph. Intended as empirical
support for the continuous-ramp (linear-ramp) modelling choice: it shows
that the real actuation transition from accelerating to braking is a
finite-duration ramp, not an instantaneous step.

The y-axis is the ego's realized longitudinal acceleration (m/s^2),
derived from the logged ego speed by a smoothed finite difference.
Positive = accelerating, negative = braking.

By default each scenario is aligned to its own BRAKE ONSET (t = 0 at the
moment the ego begins its final sustained braking), so the ramps overlay
and are directly comparable. Use --align start to plot raw time from the
start of each scenario instead.

USAGE
-----
    # all JSONs in a folder, aligned to brake onset (default)
    python plot_accel_profiles.py /path/to/json_folder

    # explicit files
    python plot_accel_profiles.py a_data.json b_data.json c_data.json

    # raw time from start instead of brake-onset alignment
    python plot_accel_profiles.py /path/to/folder --align start

    # show the COMMANDED acceleration (control_cmds) instead of realized
    python plot_accel_profiles.py /path/to/folder --commanded

    # custom window (onset mode), output name, and smoothing
    python plot_accel_profiles.py /path/to/folder \
        --window -1.5 3.0 --out accel.png --smooth 0.12

    # if you point it at thousands of files, sample a readable subset
    python plot_accel_profiles.py /path/to/folder --sample 25

OPTIONS
-------
  --align {onset,start}   x-axis reference (default: onset)
  --window LO HI          time window in seconds for onset mode
                          (default: -1.0 2.5)
  --commanded             plot commanded accel from control_cmds instead
                          of the realized accel from speed
  --smooth SECONDS        finite-difference half-window (default: 0.10)
  --brake-thresh M/S^2    decel magnitude that counts as braking
                          (default: 0.5)
  --sample N              randomly sample N files if more are found
  --out PATH              output figure (default: ego_accel_profiles.png)

DEPENDENCIES
------------
  numpy, matplotlib  (standard scientific Python; no project modules).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------
def find_frames(obj):
    """Locate the list of frames in a loaded JSON object."""
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


def vel_mag(v):
    """Speed from a velocity dict: prefer 'magnitude', else sqrt(x^2+y^2)."""
    if not isinstance(v, dict):
        return None
    m = v.get("magnitude")
    if m is not None:
        return float(m)
    x, y = v.get("x"), v.get("y")
    if x is None or y is None:
        return None
    return float(np.hypot(x, y))


def load_series(path, commanded=False):
    """Return (timestamps, signal) for one scenario file.
    signal is the commanded accel if commanded=True, else the ego speed
    (acceleration is derived from speed later)."""
    with open(path) as fh:
        d = json.load(fh)
    frames = find_frames(d)
    if not frames:
        return None, None
    ts, sig = [], []
    for f in frames:
        t = f.get("timestamp")
        if t is None:
            continue
        if commanded:
            c = (((f.get("control_cmds") or {}).get("longitudinal") or {})
                 .get("acceleration"))
            if c is None:
                continue
            ts.append(float(t))
            sig.append(float(c))
        else:
            s = vel_mag((f.get("ego") or {}).get("velocity"))
            if s is None:
                continue
            ts.append(float(t))
            sig.append(s)
    if len(ts) < 3:
        return None, None
    return np.array(ts), np.array(sig)


# ---------------------------------------------------------------------
# Acceleration + brake-onset detection
# ---------------------------------------------------------------------
def realized_accel(ts, sp, half=0.10):
    """Centered finite-difference acceleration over a +/- half-second
    window, to suppress single-frame jitter."""
    n = len(ts)
    a = np.full(n, np.nan)
    for i in range(n):
        j = i + 1
        while j < n and ts[j] - ts[i] < half:
            j += 1
        k = i - 1
        while k >= 0 and ts[i] - ts[k] < half:
            k -= 1
        if j < n and k >= 0 and ts[j] - ts[k] > 0:
            a[i] = (sp[j] - sp[k]) / (ts[j] - ts[k])
    return a


def brake_onset_time(ts, accel, thresh=0.5):
    """Time of the onset of the final sustained braking phase: find the
    last frame with accel < -thresh, then walk back to where that braking
    run began. Falls back to the deepest-decel time, then to t0."""
    n = len(ts)
    last = None
    for i in range(n - 1, -1, -1):
        if not np.isnan(accel[i]) and accel[i] < -thresh:
            last = i
            break
    if last is None:
        # no clear braking: align at the deepest deceleration instead
        if np.all(np.isnan(accel)):
            return ts[0]
        return ts[int(np.nanargmin(accel))]
    onset = last
    while onset - 1 >= 0 and not np.isnan(accel[onset - 1]) \
            and accel[onset - 1] < -thresh:
        onset -= 1
    return ts[onset]


# ---------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------
def discover_files(inputs):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            files.extend(sorted(glob.glob(os.path.join(inp, "*_data.json"))))
            files.extend(sorted(glob.glob(os.path.join(inp, "*.json"))))
        elif any(ch in inp for ch in "*?["):
            files.extend(sorted(glob.glob(inp)))
        else:
            files.append(inp)
    # de-duplicate, preserve order
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def label_for(path):
    base = os.path.basename(path)
    for suf in ("_data.json", ".json"):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Plot ego acceleration->deceleration profiles, one "
                    "line per scenario, on a single graph.")
    ap.add_argument("inputs", nargs="+",
                    help="Folder, glob, or explicit *_data.json files.")
    ap.add_argument("--align", choices=["onset", "start"], default="onset",
                    help="x-axis reference (default: onset).")
    ap.add_argument("--window", nargs=2, type=float, default=[-1.0, 2.5],
                    metavar=("LO", "HI"),
                    help="Time window (s) for onset alignment.")
    ap.add_argument("--commanded", action="store_true",
                    help="Plot commanded accel (control_cmds) not realized.")
    ap.add_argument("--smooth", type=float, default=0.10,
                    help="Finite-difference half-window in seconds.")
    ap.add_argument("--brake-thresh", type=float, default=0.5,
                    help="Decel magnitude (m/s^2) counted as braking.")
    ap.add_argument("--sample", type=int, default=None,
                    help="Randomly sample N files if more are found.")
    ap.add_argument("--out", default="ego_accel_profiles.png",
                    help="Output figure path.")
    args = ap.parse_args()

    files = discover_files(args.inputs)
    if not files:
        print("No JSON files found.", file=sys.stderr)
        return 1
    if args.sample and len(files) > args.sample:
        random.seed(0)
        files = sorted(random.sample(files, args.sample))
        print(f"Sampled {len(files)} of the discovered files.")
    if len(files) > 40:
        print(f"[warn] plotting {len(files)} lines on one graph may be "
              f"unreadable; consider --sample.", file=sys.stderr)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    cmap = plt.get_cmap("turbo")
    n_ok = 0
    for idx, path in enumerate(files):
        ts, sig = load_series(path, commanded=args.commanded)
        if ts is None:
            print(f"  [skip] {path} (no usable frames)", file=sys.stderr)
            continue
        if args.commanded:
            accel = sig
        else:
            accel = realized_accel(ts, sig, half=args.smooth)

        if args.align == "onset":
            t0 = brake_onset_time(ts, accel, thresh=args.brake_thresh)
            x = ts - t0
            mask = (x >= args.window[0]) & (x <= args.window[1])
            xx, yy = x[mask], accel[mask]
        else:  # start
            xx, yy = ts - ts[0], accel

        color = cmap(idx / max(1, len(files) - 1))
        ax.plot(xx, yy, lw=1.4, alpha=0.85, color=color, label=label_for(path))
        n_ok += 1

    ax.axhline(0, color="#444", lw=0.8)
    if args.align == "onset":
        ax.axvline(0, color="#444", lw=0.8, ls=":")
        ax.set_xlabel("time relative to brake onset (s)")
    else:
        ax.set_xlabel("time from scenario start (s)")
    ax.set_ylabel(("commanded" if args.commanded else "realized")
                  + " ego acceleration (m/s$^2$)")
    ax.set_title("Ego acceleration \u2192 deceleration per scenario\n"
                 "(positive = accelerating, negative = braking)",
                 fontweight="bold")
    ax.grid(alpha=0.25)
    # legend only if a readable number of lines
    if n_ok <= 25:
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Plotted {n_ok} scenarios -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())