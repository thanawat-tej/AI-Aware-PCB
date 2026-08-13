# AI-Aware PCB (Preventable Collision Boundary)

A comprehensive safety verification framework for autonomous driving systems that combines perception characterization with preventable collision boundary modeling.

## Overview

AI-Aware PCB provides tools to:
- **Characterize perception errors** from ground truth and AI perception data
- **Generate preventable collision boundaries** accounting for uncertainty
- **Evaluate safety models** (PCB, RSS, JRC) across multiple scenarios
- **Validate and verify** motion planning and collision avoidance logic

The framework operates across three autonomous driving scenarios:
- **Deceleration**: Lead vehicle braking scenarios
- **Cut-in**: Lane change intrusion scenarios  
- **Cut-out**: Occluded vehicle scenarios

## Project Structure

```
AI-Aware-PCB/
├── Evaluation/              # Safety model evaluation and analysis
│   ├── compare_boundaries.py     # PCB, RSS, JRC classifier implementations
│   ├── perception_characterization.py  # Perception error quantification
│   ├── plot_boundaries.py        # Visualization of safe/unsafe regions
│   ├── diagnose_false_negatives.py # FN analysis and debugging
│   ├── preventability_check.py    # Preventability assessment
│   └── fn_analyze.py             # False negative statistics
│
├── boundary_generator/      # Collision boundary visualization
│   └── plot_regions.py           # Multi-scenario boundary plots
│
├── data_pipeline/           # Data processing and transformation
│   ├── yaml_to_json.py           # YAML scenario → JSON conversion
│   ├── json_to_csv.py            # JSON → CSV extraction
│   ├── map_to_corrected.py       # Data correction pipeline
│   └── blackout_measurement.py   # Perception blackout analysis
│
├── verifications/           # Mathematical verifications
│   ├── verify_collision_boundary.py   # Boundary correctness
│   ├── verify_model_b.py              # Model B verification
│   ├── verify_continuous_ramp.py      # Ramp continuity
│   ├── verify_partial_phase.py        # Phase analysis
│   └── plot_accel_profile.py          # Acceleration visualization
│
├── CODE_STYLE.md            # Code conventions
├── requirements.txt
├── setup.py
└── output/                  # Generated figures (see "Output Sets" below)
    ├── AI-PCB/                   # 110 analytic boundary plots (no data overlaid)
    │   ├── deceleration/         #   10
    │   ├── cut-in/               #   45
    │   └── cut-out/              #   55
    └── experiment/               # data-overlaid evaluation figures
        ├── full_autoware/        #  118 binned 3-model panels, full dataset
        ├── sanitize_data/        #  101 binned 3-model panels, sanitized dataset
        └── full_autoware_AI-PCB/ # 7529 exact-speed single-model panels
```

## Scenario Data

The Autoware scenario recordings are **not distributed in this repository**. Each
run is a YAML holding the ego's estimated kinematics, the ground-truth vehicle
states and the perception objects, and the full set is far past what a Git
repository should carry (~16 GB), so it is hosted separately.

Expected layout once obtained — the pipeline reads from these directories:

```
Autoware_data/
└── cutout_yaml/          # 1,638 scenario YAMLs (cut-out); likewise for the
                          #   deceleration and cut-in families
```

Convert to the CSV the evaluation tools consume:

```bash
python3 data_pipeline/yaml_to_json.py  Autoware_data/<family>_yaml/  JSON_Data_<family>/
python3 data_pipeline/json_to_csv.py   JSON_Data_<family>/  sut.csv  --mode snapshot
```

Use `--mode snapshot` for contact and impact statistics (one row per run) and
`--mode all` for per-frame region occupancy; mixing them silently corrupts both.

## Output Sets

The four populated output folders answer different questions and are **not**
interchangeable. The panel count differs by an order of magnitude between them
because the *grouping rule* differs, not because data is missing.

| Folder | Source | Grouping | Panels | Rows shown |
|--------|--------|----------|--------|------------|
| `AI-PCB/` | none (analytic) | one plot per nominal speed pair | 110 | n/a |
| `experiment/full_autoware/` | `sut_new.csv` (10,914 rows) | speed **bins** of 2 m/s, `--min-rows 1` | 118 | **100%** |
| `experiment/sanitize_data/` | `sut_sanitized.csv` (5,324 rows) | speed **bins** of 2 m/s, `--min-rows 1` | 101 | **100%** |
| `experiment/full_autoware_AI-PCB/` | `sut_new.csv` | **exact** measured speed pair, `--min-rows 1` | 7,529 | 100% |

The two `experiment/` sets are the **3-model comparison** (RSS | JRC | AI-PCB,
`--zones-all`). `--all-pairs` merges every ego/lead speed inside a 2 m/s bin into
one panel; `--min-rows 1` keeps every bin, however thin, so no row is dropped.
The default `--min-rows 30` would discard bins under 30 rows and lose coverage:

| `--min-rows` | panels | rows shown |
|---|---|---|
| 1 (used here) | 118 | **100%** |
| 5 | 72 | 99.2% |
| 10 | 60 | 98.5% |
| 30 (default) | 37 | 95.2% |

Binning is what makes the comparison informative: because a panel spans a *range*
of speeds, the best/median/worst envelope has real width and each model shows its
speed-range uncertainty.

`experiment/full_autoware_AI-PCB/` is a different question — a **single-model** (`--pcb-only`)
view grouped on the **exact** measured `(u_t, w)` floats. Those almost never
repeat, so it is close to one panel per row: 7,529 files for 10,914 rows. With one
exact speed per panel the envelope collapses to a single curve, and the median
panel holds one data point.

### Regenerating

```bash
# binned 3-model panels (RSS | JRC | AI-PCB), all scenarios
python3 plot_boundaries_all.py csv_new/sut_new.csv --scenario all --gap perceived \
    --outdir output/experiment/full_autoware \
    --all-pairs --min-rows 1 --zones-all --data-boundary

python3 plot_boundaries_all.py csv_new/sut_sanitized.csv --scenario all --gap perceived \
    --outdir output/experiment/sanitize_data \
    --all-pairs --min-rows 1 --zones-all --data-boundary

# exact-speed single-model panels
python3 plot_boundaries_all.py csv_new/sut_new.csv --scenario all --gap perceived \
    --pcb-only --all-pairs --exact-pairs --min-rows 1 --max-plots 9000 \
    --outdir output/experiment/full_autoware_AI-PCB
```

> **The scripts under `Evaluation/` have diverged from the working copies that
> produced the current figures.** `plot_boundaries.py` is 157 lines here versus
> 538 in the working tree; `compare_boundaries.py` and
> `diagnose_false_negatives.py` also differ. Only `json_to_csv.py` is identical.
> The repository copies do **not** have `--scenario all`, `--exact-pairs`,
> `--cutout-y`, the cut-out boundary, the JRC best/median/worst envelope, or the
> zero-anchored axes. Sync them before treating this README's commands as
> reproducible from a clean clone.

## Installation

### Requirements
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy

### Setup

```bash
git clone https://github.com/thanawat-tej/AI-Aware-PCB.git
cd AI-Aware-PCB
pip install -r requirements.txt
```

## Quick Start

### 1. Characterize Perception Errors

```bash
cd Evaluation
python3 perception_characterization.py /path/to/json_data/ \
  --eps 0.97 --eps-y 0.32 --gamma 0.7 --gamma-y 0.18 \
  --out perception_char.png
```

Outputs bias estimates and residual statistics for:
- `eps_max`: Longitudinal position error
- `eps_y_max`: Lateral position error
- `gamma_max`: Speed magnitude error
- `gamma_y_max`: Lateral velocity error

### 2. Generate Collision Boundaries

```bash
cd boundary_generator
python3 plot_regions.py --scenario deceleration --yaxis ego --w 50
```

Creates safety region plots (green=safe, yellow=unsafe, red=collision) for:
- Deceleration scenarios
- Cut-in intrusion scenarios
- Cut-out occlusion scenarios

### 3. Evaluate Safety Models

```bash
cd Evaluation
python3 plot_boundaries.py data.csv --scenario cutin --umin 15 --umax 16 --wmin 9 --wmax 11
```

Compares PCB, RSS, and JRC boundaries against real data:
- Identifies false negatives (collisions in safe zone)
- Measures exceedance rates before/after bias compensation
- Generates confusion matrices

### 4. Batch Generate Plots

For systematic evaluation:

```bash
# Deceleration: ego speed vs lead speeds 10-100 km/h
for w in 10 20 30 40 50 60 70 80 90 100; do
  python3 plot_regions.py --scenario deceleration --yaxis ego --w $w \
    --out "output/AI-PCB/deceleration/decel_ego_lead${w}kmh.png"
done

# Cut-in: all ego > intruder pairs
for ego in 20 30 40 50 60 70 80 90 100; do
  for intruder in $(seq 10 10 $((ego-10))); do
    python3 plot_regions.py --scenario cutin --yaxis lateral \
      --u $ego --w $intruder \
      --out "output/AI-PCB/cut-in/cutin_lateral_ego${ego}_intruder${intruder}.png"
  done
done
```

## Data Pipeline

### Input: YAML Scenario Format

```yaml
metadata:
  total_frames: 2045
frames:
  - timestamp: 16.345
    ego:
      position: {x: ..., y: ..., z: ...}
      velocity: {x: ..., y: ..., z: ..., magnitude: ...}
    npc1:
      position: {x: ..., y: ..., z: ...}
      velocity: {x: ..., y: ..., z: ..., magnitude: ...}
    perception_objects:
      - position: {x: ..., y: ...}
        velocity: {x: ..., y: ...}
```

### Processing Steps

1. **YAML → JSON**: `yaml_to_json.py` converts YAML to JSON format
2. **JSON → CSV**: `json_to_csv.py` extracts perception errors and metrics
3. **Correction**: `map_to_corrected.py` applies bias compensation
4. **Analysis**: Evaluation tools analyze corrected data

## Key Concepts

### PCB (Predictive Collision Boundary)

An AI-aware safety model accounting for perception uncertainty:

```
perceived = true + bias + noise
|noise| ≤ budget
```

**Key parameters:**
- `DELTA_SYS` (0.15 s): System delay
- `RHO_ACT` (0.15 s): Actuator response time  
- `EPS_MAX` (0.97 m): Longitudinal position margin
- `EPS_Y_MAX` (0.32 m): Lateral position margin
- `GAMMA_MAX` (0.70 m/s): Speed magnitude margin
- `GAMMA_Y_MAX` (0.18 m/s): Lateral velocity margin

### Perception Error Characterization

For each error type, the framework estimates:
- **Bias (b)**: Systematic offset (mean error)
- **Residual (ν)**: Bounded noise after bias removal
- **Verdict**:
  - BIAS-DOMINATED: Exceedance drops below 5% after compensation
  - TAIL-DOMINATED: Median within bound but heavy tail remains
  - WITHIN-BOUND: Raw error already meets bound

### Safety Zones

Each boundary generates three regions:

| Zone | PCB Status | Color | Meaning |
|------|-----------|-------|---------|
| Safe | d ≥ d_req | 🟢 Green | Safe to proceed |
| Unsafe | d_coll < d < d_req | 🟡 Yellow | Risky; requires braking |
| Collision | d ≤ d_coll | 🔴 Red | Unavoidable crash |

## Safety Models

### 1. PCB (Predictive Collision Boundary)
- Accounts for perception uncertainty
- Three-phase braking model (delay, ramp, max deceleration)
- 6 independent error margins per scenario

### 2. RSS (Responsibility-Sensitive Safety)
- Perfect perception baseline
- Lemma-based stopping distances
- Single response time parameter

### 3. JRC (Juridical Responsibility and Causation)
- Closed-loop CC driver model
- Exact cut-in crash frontier via simulation
- Comparative baseline for model validation

## Usage Examples

### Example 1: Analyze Deceleration Scenario

```python
from Evaluation.compare_boundaries import PCB, RSS, JRC, cls_decel

# Define speeds (m/s)
ego_speed = 15  # 54 km/h
lead_speed = 10 # 36 km/h

# Test gaps
for gap in [5, 10, 20, 30]:
    pcb_result = cls_decel(gap, ego_speed, lead_speed, PCB)
    rss_result = cls_decel(gap, ego_speed, lead_speed, RSS)
    print(f"Gap {gap}m: PCB={pcb_result}, RSS={rss_result}")
```

### Example 2: Generate Perception Error Report

```bash
cd Evaluation
python3 perception_characterization.py \
  JSON_Data_cutin_new JSON_Data_decel_new JSON_Data_cutout_new \
  --eps 0.97 --eps-y 0.32 --gamma 0.7 --gamma-y 0.18 \
  --out perception_report.png
```

### Example 3: Compare Models on Data

```bash
cd Evaluation
python3 plot_boundaries.py corrected.csv \
  --scenario cutin \
  --umin 15 --umax 16 \
  --wmin 9 --wmax 11 \
  --gap truth \
  --out cutin_truth.png
```

## Output Interpretation

### Perception Characterization Report

```
[gamma_y_max]  (bound = 0.18 m/s)   lateral speed error (|perc| - |true|)
    n = 3630868   estimated bias b = -0.001  (mean -0.049) m/s
    RAW error      : mean -0.049  p50 -0.001  p95 +0.022  p99 +0.078
      exceed (raw  > 0.18):   0.1%
    RESIDUAL nu    : mean -0.049  p50 +0.000  p95 +0.023  p99 +0.079
      exceed (|nu| > 0.18):   7.4%   (dangerous-side 0.1%)
    VERDICT: WITHIN-BOUND
```

- **WITHIN-BOUND**: Error stays within margin even before compensation ✓
- **BIAS-DOMINATED**: Systematic offset removed by compensation
- **TAIL-DOMINATED**: Residual tail remains; heavier margin needed

### Boundary Plots

Each plot shows:
- **Color regions**: PCB safety classification
- **Dashed line**: PCB d_req boundary
- **Solid line**: PCB d_coll boundary
- **Dotted lines**: RSS or other baseline comparisons
- **Data points**: Actual scenario outcomes (green/red for safe/collision)

## License & Usage

This project is provided as-is under the MIT License. This is a research project and is **not open for external contributions**. 

If you find this work useful, please cite it in your research (see Citation section above).

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use AI-Aware PCB in your research, please cite:

```bibtex
@software{ai_aware_pcb_2026,
  title={AI-Aware Preventable Collision Boundary},
  author={TEJAPIJAYA Thanawat},
  year={2026},
  url={https://github.com/thanawat-tej/AI-Aware-PCB}
}
```

## Support & Questions

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: See `docs/` folder for detailed guides

## References

- RSS framework: Shalev-Shwartz et al.
- JRC: https://github.com/ec-jrc/JRC-FSM 
- Autoware integration: Tier IV

---

**Last Updated**: 13 August 2026  
**Status**: Figures current; `Evaluation/` scripts pending sync with the working tree
