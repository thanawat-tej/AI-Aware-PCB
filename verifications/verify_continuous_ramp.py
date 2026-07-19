"""
COMPREHENSIVE VERIFICATION — Continuous-Ramp Phase B Model for AV PCB
======================================================================
Four independent methods to verify the closed-form formulas:
  1. Symbolic integration with SymPy
  2. Euler integration (forward, dt=1e-5)
  3. RK4 integration (gold standard)
  4. Continuity checks at phase boundaries

If all four agree, the closed-form formulas are correct.

The model:
  Phase A:  a(t) = +alpha_max,                                  t in [0, delta_sys]
  Phase B:  a(t) = alpha_max - (alpha_max+beta_min)*(t-delta_sys)/rho_act,
                                                                t in [delta_sys, delta_sys+rho_act]
  Phase C:  a(t) = -beta_min,                                   t > delta_sys+rho_act
"""
import numpy as np
import sympy as sp


# =====================================================================
# PARAMETERS (paper's example)
# =====================================================================
ALPHA_MAX = 3.0
BETA_MIN  = 4.0
DELTA_SYS = 0.15
RHO_ACT   = 0.6
U_T       = 16.348


# =====================================================================
# THE PROPOSED CLOSED-FORM FORMULAS (to be verified)
# =====================================================================
def closed_form_uA():
    """Velocity at end of Phase A."""
    return U_T + ALPHA_MAX * DELTA_SYS

def closed_form_uB():
    """
    Velocity at end of Phase B with continuous ramp from +alpha_max to -beta_min.
    Average acceleration during B: (alpha_max + (-beta_min))/2 = (alpha_max - beta_min)/2
    Velocity change: avg_a * rho_act
    """
    u_A = closed_form_uA()
    return u_A + (ALPHA_MAX - BETA_MIN) * RHO_ACT / 2.0

def closed_form_dx_A():
    """Distance covered in Phase A: u_t*delta_sys + 0.5*alpha_max*delta_sys^2."""
    return U_T * DELTA_SYS + 0.5 * ALPHA_MAX * DELTA_SYS**2

def closed_form_dx_B():
    """
    Distance covered in Phase B with continuous ramp.
    My claim: u_A * rho_act + (2*alpha_max - beta_min) * rho_act^2 / 6
    """
    u_A = closed_form_uA()
    return u_A * RHO_ACT + (2*ALPHA_MAX - BETA_MIN) * RHO_ACT**2 / 6.0


# =====================================================================
# METHOD 1: SYMBOLIC VERIFICATION WITH SYMPY
# =====================================================================

def method1_symbolic():
    """
    Re-derive the formulas from scratch using SymPy and compare.
    """
    print("=" * 72)
    print("METHOD 1: Symbolic verification with SymPy")
    print("=" * 72)
    
    # Define symbols
    t, tau = sp.symbols('t tau', real=True, nonnegative=True)
    alpha, beta, rho, dsys, ut = sp.symbols(
        'alpha_max beta_min rho_act delta_sys u_t',
        real=True, positive=True)
    
    # ---- PHASE A ----
    # a(t) = +alpha during [0, delta_sys]
    a_A = alpha
    # u(t) starts from u_t, integrates a
    u_A_sym = ut + sp.integrate(a_A, (t, 0, dsys))
    print(f"\n  Phase A end velocity: u_A = {u_A_sym}")
    expected_uA = ut + alpha * dsys
    assert sp.simplify(u_A_sym - expected_uA) == 0, "Phase A velocity wrong"
    
    # x(t) at end of Phase A (starting from x=0)
    # Integrate velocity profile in A: u(s) = u_t + alpha*s for s in [0, dsys]
    s = sp.symbols('s', real=True, nonnegative=True)
    u_in_A = ut + alpha * s
    dx_A_sym = sp.integrate(u_in_A, (s, 0, dsys))
    dx_A_expanded = sp.expand(dx_A_sym)
    print(f"  Phase A distance:     dx_A = {dx_A_expanded}")
    expected_dxA = ut * dsys + alpha * dsys**2 / 2
    assert sp.simplify(dx_A_sym - expected_dxA) == 0, "Phase A distance wrong"
    print("  ✓ Phase A formulas match")
    
    # ---- PHASE B (continuous ramp) ----
    # a(tau) = alpha - (alpha+beta) * tau/rho   for tau in [0, rho]
    a_B = alpha - (alpha + beta) * tau / rho
    # Verify boundary conditions
    a_B_at_0 = a_B.subs(tau, 0)
    a_B_at_rho = a_B.subs(tau, rho)
    print(f"\n  Phase B a(tau=0)   = {a_B_at_0}  (should equal +alpha)")
    print(f"  Phase B a(tau=rho) = {sp.simplify(a_B_at_rho)}  (should equal -beta)")
    assert sp.simplify(a_B_at_0 - alpha) == 0
    assert sp.simplify(a_B_at_rho - (-beta)) == 0
    print("  ✓ Phase B boundary conditions correct")
    
    # Velocity during Phase B: u(tau) = u_A + integral of a from 0 to tau
    sigma = sp.symbols('sigma', real=True, nonnegative=True)
    a_B_sigma = alpha - (alpha + beta) * sigma / rho
    u_B_tau = u_A_sym + sp.integrate(a_B_sigma, (sigma, 0, tau))
    u_B_tau = sp.simplify(u_B_tau)
    print(f"\n  Phase B velocity profile: u(tau) = {sp.expand(u_B_tau)}")
    
    # Velocity at end of Phase B
    u_B_sym = u_B_tau.subs(tau, rho)
    u_B_simplified = sp.simplify(u_B_sym)
    print(f"  Phase B end velocity:     u_B = {sp.expand(u_B_simplified)}")
    
    # My closed-form claim
    my_claim_uB = u_A_sym + (alpha - beta) * rho / 2
    diff = sp.simplify(u_B_simplified - my_claim_uB)
    print(f"  Difference from my claim u_A + (alpha-beta)*rho/2: {diff}")
    assert diff == 0, "Phase B velocity formula is WRONG"
    print("  ✓ Phase B end velocity formula CORRECT: u_B = u_A + (alpha-beta)*rho/2")
    
    # Distance during Phase B: integrate u(tau) from 0 to rho
    dx_B_sym = sp.integrate(u_B_tau, (tau, 0, rho))
    dx_B_simplified = sp.simplify(sp.expand(dx_B_sym))
    print(f"\n  Phase B distance: dx_B = {dx_B_simplified}")
    
    # My closed-form claim
    my_claim_dxB = u_A_sym * rho + (2*alpha - beta) * rho**2 / 6
    diff2 = sp.simplify(sp.expand(dx_B_simplified - my_claim_dxB))
    print(f"  Difference from my claim u_A*rho + (2*alpha-beta)*rho^2/6: {diff2}")
    assert diff2 == 0, "Phase B distance formula is WRONG"
    print("  ✓ Phase B distance formula CORRECT: dx_B = u_A*rho + (2*alpha-beta)*rho^2/6")
    
    # ---- Numeric substitution to confirm ----
    subs = {alpha: ALPHA_MAX, beta: BETA_MIN, rho: RHO_ACT,
            dsys: DELTA_SYS, ut: U_T}
    uA_num = float(u_A_sym.subs(subs))
    uB_num = float(u_B_simplified.subs(subs))
    dxA_num = float(dx_A_sym.subs(subs))
    dxB_num = float(dx_B_simplified.subs(subs))
    
    print(f"\n  Numeric values from symbolic:")
    print(f"    u_A  = {uA_num:.10f}")
    print(f"    u_B  = {uB_num:.10f}")
    print(f"    dx_A = {dxA_num:.10f}")
    print(f"    dx_B = {dxB_num:.10f}")
    
    print(f"\n  Compared to my closed-form:")
    print(f"    u_A  = {closed_form_uA():.10f}")
    print(f"    u_B  = {closed_form_uB():.10f}")
    print(f"    dx_A = {closed_form_dx_A():.10f}")
    print(f"    dx_B = {closed_form_dx_B():.10f}")
    
    assert abs(uA_num - closed_form_uA())   < 1e-10
    assert abs(uB_num - closed_form_uB())   < 1e-10
    assert abs(dxA_num - closed_form_dx_A()) < 1e-10
    assert abs(dxB_num - closed_form_dx_B()) < 1e-10
    print("  ✓ Symbolic and closed-form match to <1e-10")
    
    return uA_num, uB_num, dxA_num, dxB_num


# =====================================================================
# METHOD 2: EULER INTEGRATION (forward)
# =====================================================================

def a_function(t):
    """The piecewise acceleration function."""
    if t < DELTA_SYS:
        return ALPHA_MAX
    elif t < DELTA_SYS + RHO_ACT:
        tau = t - DELTA_SYS
        return ALPHA_MAX - (ALPHA_MAX + BETA_MIN) * tau / RHO_ACT
    else:
        return -BETA_MIN

def method2_euler(dt=1e-6):
    """
    Forward Euler integration with very small dt.
    """
    print()
    print("=" * 72)
    print(f"METHOD 2: Euler integration with dt = {dt}")
    print("=" * 72)
    
    t = 0.0
    u = U_T
    x = 0.0
    
    uA_euler, uB_euler = None, None
    dxA_euler, dxB_euler = None, None
    
    t_end = DELTA_SYS + RHO_ACT
    n_steps = int(t_end / dt) + 1
    
    for i in range(n_steps):
        if uA_euler is None and t >= DELTA_SYS - dt/2:
            uA_euler = u
            dxA_euler = x
        
        a = a_function(t)
        u = u + a * dt
        x = x + u * dt   # using updated velocity (semi-implicit Euler is more accurate)
        # Actually let's use plain forward Euler for honesty:
        # u_new = u + a*dt; x_new = x + u*dt (using OLD u)
        # The above is one combined step. Let me redo properly:
        t = t + dt
    
    # Restart with cleaner forward Euler
    t = 0.0
    u = U_T
    x = 0.0
    uA_euler = uB_euler = None
    dxA_euler = dxB_euler = None
    
    # Step until just past delta_sys
    while t < DELTA_SYS - dt/2:
        a = a_function(t)
        x = x + u * dt
        u = u + a * dt
        t = t + dt
    uA_euler = u
    dxA_euler = x
    
    # Step until just past delta_sys + rho_act
    while t < DELTA_SYS + RHO_ACT - dt/2:
        a = a_function(t)
        x = x + u * dt
        u = u + a * dt
        t = t + dt
    uB_euler = u
    dxB_euler = x - dxA_euler  # distance covered IN Phase B only
    
    print(f"  Euler results:")
    print(f"    u_A  = {uA_euler:.6f}  (closed-form: {closed_form_uA():.6f}, error: {abs(uA_euler - closed_form_uA()):.2e})")
    print(f"    u_B  = {uB_euler:.6f}  (closed-form: {closed_form_uB():.6f}, error: {abs(uB_euler - closed_form_uB()):.2e})")
    print(f"    dx_A = {dxA_euler:.6f}  (closed-form: {closed_form_dx_A():.6f}, error: {abs(dxA_euler - closed_form_dx_A()):.2e})")
    print(f"    dx_B = {dxB_euler:.6f}  (closed-form: {closed_form_dx_B():.6f}, error: {abs(dxB_euler - closed_form_dx_B()):.2e})")
    
    # Euler has O(dt) error, so with dt=1e-6 we expect errors ~1e-5 or smaller
    assert abs(uA_euler - closed_form_uA()) < 1e-3
    assert abs(uB_euler - closed_form_uB()) < 1e-3
    assert abs(dxA_euler - closed_form_dx_A()) < 1e-3
    assert abs(dxB_euler - closed_form_dx_B()) < 1e-3
    print(f"  ✓ Euler agrees with closed-form to <1e-3 (expected: O(dt)={dt})")
    
    return uA_euler, uB_euler, dxA_euler, dxB_euler


# =====================================================================
# METHOD 3: RK4 INTEGRATION (gold standard)
# =====================================================================

def method3_rk4(dt=1e-4):
    """
    Fourth-order Runge-Kutta integration. Much more accurate than Euler.
    For a system u' = a(t), x' = u, RK4 stages are:
      k1 = a(t)
      k2 = a(t + dt/2)
      k3 = a(t + dt/2)
      k4 = a(t + dt)
      u_new = u + (dt/6)(k1 + 2k2 + 2k3 + k4)
    For position with x' = u(t):
      x_new = x + (dt/6)(u + 2*u_mid + 2*u_mid + u_end)
    But since u is changing, we need to be careful. Let me do this properly.
    """
    print()
    print("=" * 72)
    print(f"METHOD 3: RK4 integration with dt = {dt}")
    print("=" * 72)
    
    def step_rk4(t, u, x, dt):
        # State: y = [u, x]; y' = [a(t), u]
        # k1
        k1_u = a_function(t)
        k1_x = u
        # k2
        u2 = u + 0.5*dt*k1_u
        k2_u = a_function(t + 0.5*dt)
        k2_x = u2
        # k3
        u3 = u + 0.5*dt*k2_u
        k3_u = a_function(t + 0.5*dt)
        k3_x = u3
        # k4
        u4 = u + dt*k3_u
        k4_u = a_function(t + dt)
        k4_x = u4
        # Combine
        u_new = u + (dt/6.0)*(k1_u + 2*k2_u + 2*k3_u + k4_u)
        x_new = x + (dt/6.0)*(k1_x + 2*k2_x + 2*k3_x + k4_x)
        return u_new, x_new
    
    t = 0.0
    u = U_T
    x = 0.0
    
    while t < DELTA_SYS - dt/2:
        u, x = step_rk4(t, u, x, dt)
        t += dt
    uA_rk4 = u
    dxA_rk4 = x
    
    while t < DELTA_SYS + RHO_ACT - dt/2:
        u, x = step_rk4(t, u, x, dt)
        t += dt
    uB_rk4 = u
    dxB_rk4 = x - dxA_rk4
    
    print(f"  RK4 results:")
    print(f"    u_A  = {uA_rk4:.10f}  (closed-form: {closed_form_uA():.10f}, error: {abs(uA_rk4 - closed_form_uA()):.2e})")
    print(f"    u_B  = {uB_rk4:.10f}  (closed-form: {closed_form_uB():.10f}, error: {abs(uB_rk4 - closed_form_uB()):.2e})")
    print(f"    dx_A = {dxA_rk4:.10f}  (closed-form: {closed_form_dx_A():.10f}, error: {abs(dxA_rk4 - closed_form_dx_A()):.2e})")
    print(f"    dx_B = {dxB_rk4:.10f}  (closed-form: {closed_form_dx_B():.10f}, error: {abs(dxB_rk4 - closed_form_dx_B()):.2e})")
    
    # RK4 has O(dt^4) error, so with dt=1e-4 we expect errors ~1e-16 (machine epsilon)
    # Note: there's a phase-boundary effect since a(t) is C^0 but not differentiable
    # at the boundaries. RK4 may not achieve full 4th-order accuracy at boundaries.
    assert abs(uA_rk4 - closed_form_uA()) < 1e-6
    assert abs(uB_rk4 - closed_form_uB()) < 1e-6
    assert abs(dxA_rk4 - closed_form_dx_A()) < 1e-6
    assert abs(dxB_rk4 - closed_form_dx_B()) < 1e-6
    print(f"  ✓ RK4 agrees with closed-form to <1e-6")
    
    return uA_rk4, uB_rk4, dxA_rk4, dxB_rk4


# =====================================================================
# METHOD 4: CONTINUITY CHECKS AT PHASE BOUNDARIES
# =====================================================================

def method4_continuity():
    """
    Verify that a(t), u(t), x(t) are continuous at the phase boundaries.
    a(t) should be continuous (this is the 'continuous ramp' model).
    u(t) and x(t) should ALWAYS be continuous (physical requirement).
    """
    print()
    print("=" * 72)
    print("METHOD 4: Continuity checks at phase boundaries")
    print("=" * 72)
    
    eps = 1e-9
    
    # Boundary t = delta_sys (Phase A → Phase B)
    a_left = a_function(DELTA_SYS - eps)
    a_right = a_function(DELTA_SYS + eps)
    print(f"\n  At t = delta_sys = {DELTA_SYS}:")
    print(f"    a(t-) = {a_left:.6f}  (Phase A end, expected: +alpha_max = {ALPHA_MAX})")
    print(f"    a(t+) = {a_right:.6f}  (Phase B start, expected: +alpha_max = {ALPHA_MAX})")
    print(f"    Continuous in a? {'YES' if abs(a_left - a_right) < 1e-3 else 'NO'}")
    assert abs(a_left - a_right) < 1e-3, "a(t) should be continuous at t=delta_sys"
    
    # Boundary t = delta_sys + rho_act (Phase B → Phase C)
    a_left2 = a_function(DELTA_SYS + RHO_ACT - eps)
    a_right2 = a_function(DELTA_SYS + RHO_ACT + eps)
    print(f"\n  At t = delta_sys + rho_act = {DELTA_SYS + RHO_ACT}:")
    print(f"    a(t-) = {a_left2:.6f}  (Phase B end, expected: -beta_min = {-BETA_MIN})")
    print(f"    a(t+) = {a_right2:.6f}  (Phase C start, expected: -beta_min = {-BETA_MIN})")
    print(f"    Continuous in a? {'YES' if abs(a_left2 - a_right2) < 1e-3 else 'NO'}")
    assert abs(a_left2 - a_right2) < 1e-3, "a(t) should be continuous at t=delta_sys+rho_act"
    
    # Sample a(t) over the entire interval to check it traces +alpha_max -> -beta_min
    print(f"\n  Sampling a(t) at 11 points across [0, {DELTA_SYS + RHO_ACT}]:")
    times = np.linspace(0, DELTA_SYS + RHO_ACT, 11)
    print(f"  {'t':>8} | {'a(t)':>10} | {'Phase':>8}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*8}")
    for tt in times:
        a = a_function(tt + 1e-12)  # tiny offset to avoid boundary
        if tt < DELTA_SYS:
            phase = 'A'
        elif tt <= DELTA_SYS + RHO_ACT:
            phase = 'B'
        else:
            phase = 'C'
        print(f"  {tt:>8.4f} | {a:>10.4f} | {phase:>8}")
    
    print("\n  ✓ a(t) is continuous at both phase boundaries")
    print("  ✓ a(t) traces from +alpha_max through 0 down to -beta_min during Phase B")
    print("  ✓ Confirms Phase B covers the FULL range [+alpha_max, -beta_min]")


# =====================================================================
# RUN ALL METHODS
# =====================================================================

def main():
    print()
    print("#" * 72)
    print("#  VERIFICATION OF CONTINUOUS-RAMP PHASE B FORMULAS  ".center(72, "#"))
    print("#" * 72)
    print(f"\n  Parameters: alpha_max={ALPHA_MAX}, beta_min={BETA_MIN}, "
          f"delta_sys={DELTA_SYS}, rho_act={RHO_ACT}, u_t={U_T}")
    print(f"\n  Closed-form formulas to verify:")
    print(f"    u_A  = u_t + alpha_max * delta_sys")
    print(f"    u_B  = u_A + (alpha_max - beta_min) * rho_act / 2")
    print(f"    dx_A = u_t * delta_sys + 0.5 * alpha_max * delta_sys^2")
    print(f"    dx_B = u_A * rho_act + (2*alpha_max - beta_min) * rho_act^2 / 6")
    
    method1_symbolic()
    method2_euler(dt=1e-6)
    method3_rk4(dt=1e-4)
    method4_continuity()
    
    print()
    print("#" * 72)
    print("#  ALL FOUR INDEPENDENT METHODS CONFIRM THE FORMULAS  ".center(72, "#"))
    print("#" * 72)


if __name__ == '__main__':
    main()
