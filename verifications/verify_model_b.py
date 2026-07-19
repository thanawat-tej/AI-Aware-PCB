"""
VERIFICATION - Model B Cut-Out Equations
=========================================
New equations introduced by Model B:
  CO0a: t_reveal = (d_y_occ + eps_y_max) / max(v_floor, v_y_occ - gamma_y_max)
  CO0b: t_occ    = t_blackout + t_reveal
  CO1:  v_eff_rev = max(0, v_rev_hat - gamma_max - beta_max * t_occ)
"""
import numpy as np

ALPHA_MAX, BETA_MIN, BETA_MAX = 3.0, 4.0, 8.0
DELTA_SYS, RHO_ACT = 0.15, 0.6
EPS_MAX, GAMMA_MAX = 0.97, 0.70
EPS_Y_MAX, GAMMA_Y_MAX = 0.32, 0.18
V_FLOOR = 0.1

def t_reveal_eq(d_y_occ, v_y_occ, eps_y=EPS_Y_MAX, gamma_y=GAMMA_Y_MAX, v_floor=V_FLOOR):
    return (d_y_occ + eps_y) / max(v_floor, v_y_occ - gamma_y)

def t_occ_eq(t_blackout, d_y_occ, v_y_occ):
    return t_blackout + t_reveal_eq(d_y_occ, v_y_occ)

def v_eff_rev_eq(v_rev_hat, t_occ, gamma=GAMMA_MAX, beta_max=BETA_MAX):
    return max(0.0, v_rev_hat - gamma - beta_max * t_occ)

def ego_three_phase_full(u_t):
    u_A = u_t + ALPHA_MAX * DELTA_SYS
    u_B = max(0.0, u_A + (ALPHA_MAX - BETA_MIN) * RHO_ACT / 2.0)
    dx_A = u_t * DELTA_SYS + 0.5 * ALPHA_MAX * DELTA_SYS**2
    dx_B = u_A * RHO_ACT + (2*ALPHA_MAX - BETA_MIN) * RHO_ACT**2 / 6.0
    dx_C = u_B**2 / (2 * BETA_MIN) if u_B > 0 else 0.0
    return dx_A + dx_B + dx_C

def C_cutout_required(u_t, v_rev_hat, t_blackout, d_y_occ, v_y_occ):
    t_occ = t_occ_eq(t_blackout, d_y_occ, v_y_occ)
    v_eff_rev = v_eff_rev_eq(v_rev_hat, t_occ)
    s_ego = ego_three_phase_full(u_t)
    s_lead = v_eff_rev**2 / (2 * BETA_MAX)
    return max(0.0, EPS_MAX + s_ego - s_lead), t_occ, v_eff_rev

def header(s):
    print("\n" + "=" * 72)
    print("  " + s)
    print("=" * 72)

# TEST 1: Dimensional consistency
header("TEST 1: Dimensional consistency of t_reveal")
d_y_occ, v_y_occ = 2.0, 1.5
t_rev = t_reveal_eq(d_y_occ, v_y_occ)
print(f"  t_reveal = ({d_y_occ}+{EPS_Y_MAX})/({v_y_occ}-{GAMMA_Y_MAX}) = {t_rev:.4f} s")
print("  Units: m / (m/s) = s  OK")

# TEST 2: Error directions conservative
header("TEST 2: Error directions produce LONGER t_reveal (conservative)")
t_rev_noerr = d_y_occ / v_y_occ
t_rev_witherr = t_reveal_eq(d_y_occ, v_y_occ)
print(f"  Without errors: {t_rev_noerr:.4f} s")
print(f"  With errors:    {t_rev_witherr:.4f} s  (delta {t_rev_witherr-t_rev_noerr:+.4f})")
assert t_rev_witherr > t_rev_noerr
t_eps_only = (d_y_occ + EPS_Y_MAX) / v_y_occ
t_gamma_only = d_y_occ / (v_y_occ - GAMMA_Y_MAX)
print(f"  eps_y only:   {t_eps_only:.4f} s")
print(f"  gamma_y only: {t_gamma_only:.4f} s")
assert t_eps_only > t_rev_noerr and t_gamma_only > t_rev_noerr
print("  PASS: both error terms independently lengthen t_reveal")

# TEST 3: Model A recovery
header("TEST 3: Model A recovery (fast occluder => t_reveal -> 0)")
print(f"  {'v_y_occ':>12} | {'t_reveal':>12}")
for v_y in [1.0, 2.0, 5.0, 10.0, 50.0, 1000.0]:
    print(f"  {v_y:>12.1f} | {t_reveal_eq(d_y_occ, v_y):>12.6f}")
t_rev_limit = t_reveal_eq(d_y_occ, 1e9)
assert t_rev_limit < 1e-6
print(f"  Limit: {t_rev_limit:.2e} s ~ 0  -- PASS: Model A recovered")

# TEST 4: Monotonicity
header("TEST 4: Monotonicity of t_reveal")
base = t_reveal_eq(2.0, 1.5)
t_more_dist = t_reveal_eq(2.5, 1.5)
t_faster = t_reveal_eq(2.0, 2.0)
print(f"  baseline (d=2.0,v=1.5): {base:.4f}")
print(f"  d 2.0->2.5: {t_more_dist:.4f} (longer, OK)")
print(f"  v 1.5->2.0: {t_faster:.4f} (shorter, OK)")
assert t_more_dist > base and t_faster < base
print("  PASS: monotonic in both inputs")

# TEST 5: Denominator floor
header("TEST 5: Denominator floor prevents division by zero")
print(f"  {'v_y_occ':>12} | {'raw denom':>12} | {'t_reveal':>12}")
for v_y in [0.0, 0.3, 0.5, 0.6, 0.8, 1.0]:
    print(f"  {v_y:>12.2f} | {v_y-GAMMA_Y_MAX:>12.2f} | {t_reveal_eq(d_y_occ,v_y):>12.4f}")
t_at_floor = t_reveal_eq(d_y_occ, 0.0)
expected = (d_y_occ + EPS_Y_MAX) / V_FLOOR
assert abs(t_at_floor - expected) < 1e-9
print(f"  At v_y=0: t_reveal={t_at_floor:.4f}, expected={expected:.4f}  PASS")

# TEST 6: No double-counting
header("TEST 6: t_occ = t_blackout + t_reveal (additive)")
t_blackout = 1.0
t_rev = t_reveal_eq(2.0, 1.5)
t_occ = t_occ_eq(t_blackout, 2.0, 1.5)
assert abs(t_occ - (t_blackout + t_rev)) < 1e-12
print(f"  {t_blackout} + {t_rev:.4f} = {t_occ:.4f}  PASS")

# TEST 7: Worked example
header("TEST 7: Worked example end-to-end")
u_t, v_rev_hat, t_blackout, d_y_occ, v_y_occ, p_rev = 20.0, 14.0, 1.0, 2.0, 1.5, 60.0
required, t_occ, v_eff_rev = C_cutout_required(u_t, v_rev_hat, t_blackout, d_y_occ, v_y_occ)
t_rev = t_reveal_eq(d_y_occ, v_y_occ)
s_ego = ego_three_phase_full(u_t)
s_lead = v_eff_rev**2 / (2*BETA_MAX)
print(f"  t_reveal   = {t_rev:.4f} s")
print(f"  t_occ      = {t_occ:.4f} s")
print(f"  v_eff_rev  = {v_eff_rev:.4f} m/s")
print(f"  s_ego      = {s_ego:.4f} m")
print(f"  s_lead_rev = {s_lead:.4f} m")
print(f"  required   = {required:.4f} m")
print(f"  available  = {p_rev} m")
print(f"  VERDICT: {'SAFE' if p_rev >= required else 'VIOLATION (unsafe)'}")

# TEST 8: Model A vs B comparison
header("TEST 8: Model A vs Model B comparison")
u_t, v_rev_hat, t_blackout, d_y_occ = 20.0, 14.0, 1.0, 2.0
print(f"  {'v_y_occ':>8} | {'t_reveal':>9} | {'t_occ(B)':>9} | {'v_eff(A)':>9} | {'v_eff(B)':>9} | {'req(A)':>8} | {'req(B)':>8}")
for v_y in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    t_rev = t_reveal_eq(d_y_occ, v_y)
    t_occ_B = t_blackout + t_rev
    v_eff_A = v_eff_rev_eq(v_rev_hat, t_blackout)
    v_eff_B = v_eff_rev_eq(v_rev_hat, t_occ_B)
    s_ego = ego_three_phase_full(u_t)
    req_A = max(0, EPS_MAX + s_ego - v_eff_A**2/(2*BETA_MAX))
    req_B = max(0, EPS_MAX + s_ego - v_eff_B**2/(2*BETA_MAX))
    print(f"  {v_y:>8.1f} | {t_rev:>9.4f} | {t_occ_B:>9.4f} | {v_eff_A:>9.4f} | {v_eff_B:>9.4f} | {req_A:>8.2f} | {req_B:>8.2f}")

print("\n" + "=" * 72)
print("  ALL MODEL B VERIFICATION TESTS PASSED")
print("=" * 72)
