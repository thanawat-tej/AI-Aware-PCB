"""
blackout_measurement.py
=======================

Measure the cut-out *blackout / occlusion duration* directly from the
perception logs, instead of treating it as an assumed scenario input.

For a target object (the hidden vehicle, npc2 by default), a frame counts as a
DETECTION if some entry in ``perception_objects`` lies within a gate of the
target's true position; otherwise the target is OCCLUDED in that frame. A
"blackout" is a run of consecutive occluded frames. The data-grounded
occlusion duration the C_cut-out model needs is

        t_occ  =  the non-detection span preceding the reveal,

which this tool reports per scenario, together with the detection rate and the
distribution of blackout lengths across scenarios (to justify a conservative
worst-case value for design).

Why this is independent of the perception OFFSET b:
  Detection is a presence/absence event, not a position estimate, so the
  systematic offset b does not change whether a detection fires. The matching
  gate (default 5 m) is well above the measured offset (~1.37 m), so an object
  that is detected-but-biased still counts as detected. Blackout (a DETECTION
  failure, cause A) and the offset (a LOCALISATION bias, cause C) are therefore
  measured as separate, non-interacting quantities -- which is the point.

USAGE
-----
    python blackout_measurement.py /path/to/json_folder
    python blackout_measurement.py cutout*.json --range 80 --min-blackout 1.5
    python blackout_measurement.py /path/folder --target npc2 --gate 5.0 --out blackout.csv

DEPENDENCIES: standard library only (numpy optional, not required).
"""

from __future__ import annotations
import argparse, glob, json, math, os, sys


def discover(inputs):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            files += sorted(glob.glob(os.path.join(inp, "*_data.json")))
            files += sorted(glob.glob(os.path.join(inp, "*.json")))
        elif any(c in inp for c in "*?["):
            files += sorted(glob.glob(inp))
        else:
            files.append(inp)
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); out.append(f)
    return out


def frames_of(obj):
    if isinstance(obj, dict):
        for k in ("frames", "data", "timesteps", "steps"):
            if isinstance(obj.get(k), list):
                return obj[k]
    return obj if isinstance(obj, list) else None


def label(path):
    b = os.path.basename(path)
    for s in ("_data.json", ".json"):
        if b.endswith(s):
            return b[:-len(s)]
    return b


def dist(a, b, plane="XY"):
    if plane == "XZ":            # cut-out: Y is vertical, ground plane is X-Z
        return math.hypot(a["x"] - b["x"], a["z"] - b["z"])
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def pctile(vals, q):
    if not vals:
        return float("nan")
    v = sorted(vals)
    k = (len(v) - 1) * q / 100.0
    lo = int(math.floor(k)); hi = int(math.ceil(k))
    if lo == hi:
        return v[lo]
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def analyse(frames, target, gate, rng, plane):
    """Return per-scenario blackout statistics, or None if target absent."""
    # frame interval
    ts = [f.get("timestamp") for f in frames if f.get("timestamp") is not None]
    dt = 0.025
    if len(ts) > 5:
        diffs = sorted(ts[i + 1] - ts[i] for i in range(len(ts) - 1))
        dt = diffs[len(diffs) // 2] or 0.025

    present = 0          # target present (ground truth) in frame
    relevant = 0         # present AND within range of ego (if rng set)
    detected = 0         # a perception object within gate of target
    occ_flags = []       # 1 = occluded (relevant but undetected)
    first_det_idx = None
    for i, f in enumerate(frames):
        tgt = (f.get(target) or {}).get("position")
        if not tgt:
            occ_flags.append(0)
            continue
        present += 1
        ego = (f.get("ego") or {}).get("position")
        in_range = True
        if rng is not None and ego is not None:
            in_range = dist(ego, tgt, plane) <= rng
        if not in_range:
            occ_flags.append(0)
            continue
        relevant += 1
        pobs = f.get("perception_objects") or []
        det = any(po.get("position") and dist(po["position"], tgt, plane) <= gate
                  for po in pobs)
        if det:
            detected += 1
            if first_det_idx is None:
                first_det_idx = i
            occ_flags.append(0)
        else:
            occ_flags.append(1)

    if present == 0:
        return None

    # runs of consecutive occluded (relevant) frames
    runs = []
    cur = 0
    for flag in occ_flags:
        if flag:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    run_secs = [r * dt for r in runs]
    longest = max(run_secs) if run_secs else 0.0

    # occlusion before the reveal: leading occluded span up to first detection
    if first_det_idx is not None:
        lead = 0
        for flag in occ_flags[:first_det_idx]:
            lead = lead + 1 if flag else lead  # count occluded frames before reveal
        reveal_occ = lead * dt
        reveal_time = first_det_idx * dt - (frames[0].get("frame_index", 0) * dt)
    else:
        reveal_occ = (relevant if rng is not None else present) * dt  # never seen
        reveal_time = float("inf")

    denom = relevant if rng is not None else present
    return dict(
        dt=dt, present=present, relevant=relevant, detected=detected,
        det_rate=(detected / denom if denom else 0.0),
        longest_blackout=longest, reveal_occ=reveal_occ,
        n_runs=len(runs), ever_detected=first_det_idx is not None,
        run_secs=run_secs,
    )


def main():
    ap = argparse.ArgumentParser(description="Measure cut-out blackout/occlusion "
                                             "from missing detections.")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--target", default="auto",
                    help="Object whose occlusion to measure (default: npc2 if "
                         "present, else npc1).")
    ap.add_argument("--gate", type=float, default=5.0,
                    help="Detection matching gate in metres (default 5).")
    ap.add_argument("--range", type=float, default=None,
                    help="Only count frames where ego-target distance <= this "
                         "(the 'relevant' window). Default: whole scenario.")
    ap.add_argument("--min-blackout", type=float, default=1.5,
                    help="Threshold (s) for flagging a significant blackout.")
    ap.add_argument("--plane", default="auto", choices=["auto", "XY", "XZ"],
                    help="Ground plane. auto: XZ for cutout* files, else XY.")
    ap.add_argument("--out", default=None, help="Optional CSV output path.")
    args = ap.parse_args()

    files = discover(args.inputs)
    if not files:
        print("No JSON files found.", file=sys.stderr); return 1

    rows = []
    print("=" * 78)
    print("BLACKOUT / OCCLUSION MEASUREMENT  (from missing detections)")
    print("=" * 78)
    print(f"detection gate: {args.gate} m   "
          f"relevant range: {args.range if args.range else 'whole scenario'}   "
          f"target: {args.target}")
    print(f"{'scenario':<26}{'det%':>6}{'longest':>9}{'reveal':>9}{'seen?':>7}")
    print(f"{'':26}{'':6}{'blackout':>9}{'occ':>9}")
    print("-" * 78)

    for path in files:
        try:
            frames = frames_of(json.load(open(path)))
        except Exception as e:
            print(f"  [skip] {path}: {e}", file=sys.stderr); continue
        if not frames:
            continue
        tgt = args.target
        if tgt == "auto":
            tgt = "npc2" if any("npc2" in f for f in frames) else "npc1"
        plane = args.plane
        if plane == "auto":
            plane = "XZ" if "cutout" in os.path.basename(path).lower() else "XY"
        r = analyse(frames, tgt, args.gate, args.range, plane)
        if r is None:
            continue
        rows.append((label(path), tgt, r))
        seen = "yes" if r["ever_detected"] else "NEVER"
        rv = f"{r['reveal_occ']:.2f}" if r["reveal_occ"] != float("inf") else "inf"
        print(f"{label(path):<26}{100*r['det_rate']:>5.1f}%"
              f"{r['longest_blackout']:>8.2f}s{rv:>8}s{seen:>7}")

    # aggregate
    longs = [r["longest_blackout"] for _, _, r in rows]
    flagged = [r for _, _, r in rows if r["longest_blackout"] >= args.min_blackout]
    never = [r for _, _, r in rows if not r["ever_detected"]]
    print("-" * 78)
    if longs:
        print(f"scenarios: {len(rows)}   "
              f"with blackout >= {args.min_blackout}s: {len(flagged)} "
              f"({100*len(flagged)/len(rows):.0f}%)   "
              f"never detected: {len(never)}")
        print(f"longest-blackout distribution (s):  "
              f"median {pctile(longs,50):.2f}   p95 {pctile(longs,95):.2f}   "
              f"max {max(longs):.2f}")
        print()
        print("Use the per-scenario reveal-occlusion as t_occ in C_cut-out for "
              "evaluation;")
        print("use a conservative upper value (e.g. p95) of this distribution as "
              "the design bound.")
        print("Blackout here is a DETECTION gap (cause A); it is measured "
              "independently of")
        print(f"the localisation offset b (gate {args.gate} m >> b ~1.37 m), so "
              "the two stay separate.")
    if args.out and rows:
        with open(args.out, "w") as fh:
            fh.write("scenario,target,det_rate,longest_blackout_s,reveal_occ_s,"
                     "n_runs,ever_detected\n")
            for name, tgt, r in rows:
                fh.write(f"{name},{tgt},{r['det_rate']:.4f},"
                         f"{r['longest_blackout']:.3f},{r['reveal_occ']:.3f},"
                         f"{r['n_runs']},{int(r['ever_detected'])}\n")
        print(f"\nCSV -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())