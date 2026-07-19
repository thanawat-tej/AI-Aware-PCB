"""
Verification of partial-phase ego projection for cut-in.
Three cases:
  Case 1: T_entry <= delta_sys  (cut-in completes during Phase A)
  Case 2: delta_sys < T_entry <= delta_sys + rho_act  (during Phase B)
  Case 3: T_entry > delta_sys + rho_act  (during Phase C)

For each case, derive closed-form formulas and verify by RK4 simulation.
"""
import numpy as np
import sympy as sp

# Parameters
ALPHA_MAX = 3.0
BETA_MIN  = 4.0
DELTA_SYS = 0.15
RHO_ACT   = 0.6
U_T       = 16.0


def a_ego(t):
    """Ego acceleration profile (continuous ramp Phase B)."""
    if t < DELTA_SYS:
        return ALPHA_MAX
    elif t < DELTA_SYS + RHO_ACT:
        tau = t - DELTA_SYS
        return ALPHA_MAX - (ALPHA_MAX + BETA_MIN) * tau / RHO_ACT
    else:
        return -BETA_MIN


def project_ego_closed_form(u_t, T_entry, alpha=ALPHA_MAX, beta=BETA_MIN,
                              dsys=DELTA_SYS, rho=RHO_ACT):
    """
    Closed-form ego projection handling all three cases.
    Returns: (u_proj, dx_proj, case_name)
    """
    if T_entry <= 0:
        return u_t, 0.0, 'invalid'
    
    if T_entry <= dsys:
        # CASE 1: cut-in completes during Phase A
        u_proj = u_t + alpha * T_entry
        dx_proj = u_t * T_entry + 0.5 * alpha * T_entry**2
        return u_proj, dx_proj, 'Case 1 (during Phase A)'
    
    # At least Phase A is complete
    u_A = u_t + alpha * dsys
    dx_A = u_t * dsys + 0.5 * alpha * dsys**2
    
    if T_entry <= dsys + rho:
        # CASE 2: cut-in completes during Phase B
        tau = T_entry - dsys  # time elapsed within Phase B
        # Velocity at end of partial Phase B
        u_proj = u_A + alpha * tau - (alpha + beta) * tau**2 / (2 * rho)
        # Distance during partial Phase B
        dx_B = u_A * tau + alpha * tau**2 / 2 - (alpha + beta) * tau**3 / (6 * rho)
        dx_proj = dx_A + dx_B
        return u_proj, dx_proj, 'Case 2 (during Phase B)'
    
    # Phase A and B complete
    u_B = u_A + (alpha - beta) * rho / 2
    dx_B_full = u_A * rho + (2*alpha - beta) * rho**2 / 6
    
    # CASE 3: cut-in completes during Phase C
    dt_C = T_entry - dsys - rho
    # Check if ego stops before T_entry
    if u_B > 0:
        t_to_stop = u_B / beta
        if dt_C <= t_to_stop:
            u_proj = u_B - beta * dt_C
            dx_C = u_B * dt_C - 0.5 * beta * dt_C**2
        else:
            u_proj = 0.0
            dx_C = u_B**2 / (2 * beta)
    else:
        u_proj = 0.0
        dx_C = 0.0
    
    dx_proj = dx_A + dx_B_full + dx_C
    return u_proj, dx_proj, 'Case 3 (during Phase C)'


def project_ego_rk4(u_t, T_entry, dt=1e-5):
    """RK4 numerical integration as ground truth."""
    t = 0.0
    u = u_t
    x = 0.0
    
    while t < T_entry:
        step = min(dt, T_entry - t)
        # RK4
        k1u = a_ego(t)
        k1x = u
        k2u = a_ego(t + step/2)
        k2x = u + step/2 * k1u
        k3u = a_ego(t + step/2)
        k3x = u + step/2 * k2u
        k4u = a_ego(t + step)
        k4x = u + step * k3u
        u = u + step/6 * (k1u + 2*k2u + 2*k3u + k4u)
        x = x + step/6 * (k1x + 2*k2x + 2*k3x + k4x)
        # physical clamp: a braking vehicle stops, it does not reverse. The
        # closed form clamps at standstill by definition, so the reference must
        # too, else it integrates the deceleration into negative speed.
        if u <= 0.0:
            u = 0.0
            break
        t = t + step
    
    return u, x


def verify(T_entry, label):
    u_cf, dx_cf, case = project_ego_closed_form(U_T, T_entry)
    u_rk4, dx_rk4 = project_ego_rk4(U_T, T_entry)
    
    u_err = abs(u_cf - u_rk4)
    x_err = abs(dx_cf - dx_rk4)
    
    status = '✓' if (u_err < 1e-4 and x_err < 1e-4) else '✗ FAIL'
    
    print(f"  T_entry = {T_entry:.3f}s  ({case}) {label}")
    print(f"    Closed-form: u = {u_cf:.6f},  dx = {dx_cf:.6f}")
    print(f"    RK4:         u = {u_rk4:.6f},  dx = {dx_rk4:.6f}")
    print(f"    Errors:      u = {u_err:.2e},  dx = {x_err:.2e}  {status}")
    print()
    return u_err < 1e-4 and x_err < 1e-4


print("=" * 72)
print("VERIFICATION: Partial-Phase Cut-In Projection")
print("=" * 72)
print(f"Parameters: u_t={U_T}, alpha={ALPHA_MAX}, beta={BETA_MIN},")
print(f"            delta_sys={DELTA_SYS}, rho_act={RHO_ACT}")
print()

results = []
# Case 1 tests
print("CASE 1: T_entry < delta_sys (cut-in during Phase A)")
print("-" * 72)
results.append(verify(0.1, "very fast cut-in"))
results.append(verify(0.3, "fast cut-in"))
results.append(verify(0.6, "exactly at Phase A boundary"))

# Case 2 tests
print("CASE 2: delta_sys < T_entry < delta_sys + rho_act (cut-in during Phase B)")
print("-" * 72)
results.append(verify(0.65, "just past Phase A"))
results.append(verify(0.7, "mid-Phase-B"))
results.append(verify(0.8, "end of Phase B"))

# Case 3 tests
print("CASE 3: T_entry > delta_sys + rho_act (cut-in during Phase C)")
print("-" * 72)
results.append(verify(1.0, "slightly into Phase C"))
results.append(verify(2.0, "well into Phase C"))
results.append(verify(5.0, "long after Phase C starts"))

# Symbolic verification of Case 2 formula
print("=" * 72)
print("SYMBOLIC VERIFICATION OF CASE 2 (Partial Phase B)")
print("=" * 72)
tau_sym, alpha_sym, beta_sym, rho_sym, u_A_sym = sp.symbols(
    'tau alpha beta rho u_A', positive=True, real=True)

# Acceleration profile during Phase B
sigma = sp.symbols('sigma', positive=True)
a_B_sigma = alpha_sym - (alpha_sym + beta_sym) * sigma / rho_sym

# Velocity at time tau (within Phase B)
u_tau = u_A_sym + sp.integrate(a_B_sigma, (sigma, 0, tau_sym))
u_tau = sp.simplify(u_tau)
print(f"\n  Velocity at tau:  u(tau) = {sp.expand(u_tau)}")

# Distance during partial Phase B
dx_tau = sp.integrate(u_tau, (tau_sym, 0, tau_sym))
# Wait — that's wrong, integrating over tau as both bound and inside
# Let me redo with proper variable separation
tau_var = sp.symbols('tau_var', positive=True)
u_at_tau_var = u_A_sym + sp.integrate(a_B_sigma, (sigma, 0, tau_var))
dx_partial = sp.integrate(u_at_tau_var, (tau_var, 0, tau_sym))
dx_partial = sp.simplify(sp.expand(dx_partial))
print(f"\n  Distance (partial Phase B): dx_B(tau) = {dx_partial}")

# My claim: u_A * tau + alpha*tau^2/2 - (alpha+beta)*tau^3/(6*rho)
my_claim_u = u_A_sym + alpha_sym * tau_sym - (alpha_sym + beta_sym) * tau_sym**2 / (2 * rho_sym)
my_claim_dx = u_A_sym * tau_sym + alpha_sym * tau_sym**2 / 2 - (alpha_sym + beta_sym) * tau_sym**3 / (6 * rho_sym)

u_diff = sp.simplify(u_tau - my_claim_u)
dx_diff = sp.simplify(dx_partial - my_claim_dx)
print(f"\n  Difference u(tau) - my_claim: {u_diff}")
print(f"  Difference dx_B(tau) - my_claim: {dx_diff}")

assert u_diff == 0, "Velocity formula mismatch"
assert dx_diff == 0, "Distance formula mismatch"
print("\n  ✓ Symbolic formulas confirmed:")
print(f"    u(tau)  = u_A + alpha*tau - (alpha+beta)*tau^2/(2*rho)")
print(f"    dx_B(tau) = u_A*tau + alpha*tau^2/2 - (alpha+beta)*tau^3/(6*rho)")

# Boundary continuity check
print()
print("=" * 72)
print("BOUNDARY CONTINUITY CHECKS")
print("=" * 72)

# At T_entry = delta_sys (Case 1 → Case 2 boundary)
u1, x1, _ = project_ego_closed_form(U_T, DELTA_SYS - 1e-9)
u2, x2, _ = project_ego_closed_form(U_T, DELTA_SYS + 1e-9)
print(f"\n  At T_entry = delta_sys:")
print(f"    Case 1 limit: u = {u1:.6f}, dx = {x1:.6f}")
print(f"    Case 2 limit: u = {u2:.6f}, dx = {x2:.6f}")
print(f"    Jump in u: {abs(u2-u1):.2e}  (should be ~0)")
print(f"    Jump in dx: {abs(x2-x1):.2e} (should be ~0)")

# At T_entry = delta_sys + rho_act (Case 2 → Case 3 boundary)
u3, x3, _ = project_ego_closed_form(U_T, DELTA_SYS + RHO_ACT - 1e-9)
u4, x4, _ = project_ego_closed_form(U_T, DELTA_SYS + RHO_ACT + 1e-9)
print(f"\n  At T_entry = delta_sys + rho_act:")
print(f"    Case 2 limit: u = {u3:.6f}, dx = {x3:.6f}")
print(f"    Case 3 limit: u = {u4:.6f}, dx = {x4:.6f}")
print(f"    Jump in u: {abs(u4-u3):.2e}  (should be ~0)")
print(f"    Jump in dx: {abs(x4-x3):.2e} (should be ~0)")

print()
print("=" * 72)
if all(results) and abs(u2-u1)<1e-3 and abs(u4-u3)<1e-3:
    print("ALL VERIFICATION TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("=" * 72)
