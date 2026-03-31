# Analytical steady-state solution for the drogued drifter

## Summary

The full 8-DOF ODE for the buoy+drogue system can be replaced by an
analytical steady-state solution at each parcels timestep. The drift
velocity is a fixed weighted average of the surface and drogue-depth
currents, and the effective drogue depth is found by a simple 1D
fixed-point iteration. No ODE integration is needed.

## Derivation

### Setup

The drogued drifter consists of a surface buoy (drag coefficient k_b)
connected by a rigid pole of length l to a subsurface drogue (drag
coefficient k_d). Both experience quadratic drag from the surrounding
water. The drogue has net negative buoyancy W = (m_d - m_hat_d) * g.

The ocean velocity field is U(z), V(z) — an arbitrary horizontal velocity
vector that varies with depth. The buoy sits at z=0 and the drogue at
depth z_eff, which depends on the pole tilt angle.

### Steady-state force balance (drift velocity)

At steady state, the buoy and drogue move together at constant velocity
(u_drift, v_drift) with the pole at fixed angles (theta, phi). All
accelerations and angular velocities are zero.

The horizontal force balance is:

    k_b |v_slip_b| v_slip_b + k_d |v_slip_d| v_slip_d = 0

where v_slip_b = (u_drift - U(0), v_drift - V(0)) and
v_slip_d = (u_drift - U(z_eff), v_drift - V(z_eff)).

For this to hold, the two slip vectors must be antiparallel. This means
u_drift lies on the line between (U(0), V(0)) and (U(z_eff), V(z_eff))
in velocity space:

    u_drift = U(z_eff) + α (U(0) - U(z_eff))
    v_drift = V(z_eff) + α (V(0) - V(z_eff))

Substituting back:

    v_slip_b = -(1-α) * shear
    v_slip_d =  α     * shear

where shear = (U(0) - U(z_eff), V(0) - V(z_eff)).

The speed norms are |v_slip_b| = (1-α)|S| and |v_slip_d| = α|S| where
|S| = |shear|. The force balance becomes:

    k_b (1-α)² |S|² = k_d α² |S|²

which gives:

    **α = √k_b / (√k_b + √k_d)**

This is independent of the velocity profile, the shear magnitude, and
the shear direction. For the default Callies et al. parameters
(k_b=12, k_d=154): **α = 0.2182**.

The drift velocity is equivalently:

    u_drift = (1-α) U(z_eff) + α U(0)
    v_drift = (1-α) V(z_eff) + α V(0)

i.e., a weighted average with 78% weight on the drogue current and 22%
on the surface current.

### Why √k not k

With linear drag (F = -k v), the steady-state weighting would be
k_b/(k_b+k_d) ≈ 0.072. But quadratic drag (F = -k|v|v) means the
drag force scales as v², so the effective resistance is k*|v|. At
steady state, the buoy has a larger slip speed than the drogue
(it's being pulled away from the fast surface current), so the buoy drag
"punches above its weight". The square root arises from balancing
k_b (1-α)² = k_d α².

### Steady-state torque balance (pole tilt and drogue depth)

The azimuthal angle phi aligns with the shear vector (from F[3] = 0):

    phi = atan2(V(0) - V(z_eff), U(0) - U(z_eff))

The polar angle theta is determined by the balance between gravity
(restoring) and drag torque (tilting) on the pole. Defining the tilt
from vertical as δ = θ - π, the torque balance (F[2] = 0) gives:

    tan(δ) = k_d α² S(z_eff)² / W

where:
- S(z_eff) = |(U(0) - U(z_eff), V(0) - V(z_eff))| is the shear
  magnitude
- W = (m_d - m_hat_d) g is the net drogue weight

The effective drogue depth is:

    z_eff = l cos(δ)

This is implicit because S depends on z_eff. The solution is a 1D
fixed-point iteration:

    1. Start with z_eff = l (pole vertical)
    2. Compute S² = (U(0) - U(z_eff))² + (V(0) - V(z_eff))²
    3. Compute δ = arctan(k_d α² S² / W)
    4. Compute z_eff = l cos(δ)
    5. Repeat until converged

This converges because tilting the pole shallower typically reduces
the shear (velocity profiles decay with depth), which reduces the
tilt — a contracting map. Convergence is typically in 10–35 iterations
from a cold start, 1–2 iterations with warm starting.

### Verified against the full ODE

| Velocity profile          | z_eff (ODE) | z_eff (analytical) | Error |
|---------------------------|-------------|-------------------|-------|
| Exponential decay H=3m    | 2.587 m     | 2.587 m           | 0.1 mm|
| Exponential decay H=0.5m  | 1.580 m     | 1.581 m           | 0.7 mm|
| Ekman spiral H=2m 45°/e   | 2.023 m     | 2.023 m           | 0.1 mm|
| Depth-intensified flow    | 2.648 m     | 2.648 m           | 0.1 mm|

The drift velocity α was verified to be constant (0.2180 ± 0.001) across:
- Shear magnitudes from 0.01 to 5.0 m/s
- Rotation angles from 0° to 180°
- Depth-intensified, reversed, and opposing currents
- All combinations of (U_b, V_b, U_d, V_d)

## Implementation in the parcels kernel

### Current approach (ODE-based)

Each parcels timestep:
1. Sample U, V at z=0 and z=drogue_depth (fixed)
2. Call `get_final_drift_batch` → solve_ivp on (8N,) ODE system
3. Apply drift velocities

Cost: ~150ms per timestep for N=210 particles. Dominated by ODE solver.

### Proposed approach (analytical)

Precompute once:
```python
alpha = np.sqrt(k_b) / (np.sqrt(k_b) + np.sqrt(k_d))
W = (m_d - m_hat_d) * g
```

Each parcels timestep, for each particle:
```python
# Fixed-point iteration for z_eff (warm-started from previous timestep)
for _ in range(n_iter):  # ~2 iterations with warm start
    U_d, V_d = fieldset.UV[time, z_eff, lat, lon, particles]
    S2 = (U_b - U_d)**2 + (V_b - V_d)**2
    delta = np.arctan2(k_d * alpha**2 * S2, W)
    z_eff = l * np.cos(delta)

# Drift velocity
u_drift = (1 - alpha) * U_d + alpha * U_b
v_drift = (1 - alpha) * V_d + alpha * V_b
```

This is fully vectorized (no per-particle loop, no ODE) and requires
only 2–3 field sampling calls per iteration. Each iteration is O(N)
array arithmetic.

### Expected speedup

The ODE solver currently takes ~150ms for 210 particles. The analytical
approach replaces it with ~3 field samplings + arithmetic, which should
take <1ms. The parcels field interpolation and bookkeeping would then
dominate. Expected speedup: **100x or more** on the drifter kernel.

### What is lost

1. **Transient dynamics**: The analytical solution assumes the system is
   always at steady state. The ODE captures the ~10-30s relaxation
   transient. This is valid when the parcels timestep (300s) >> the
   relaxation time.

2. **Pole oscillations**: The ODE captures the pendulum-like oscillation
   of the pole during adjustment. The analytical solution jumps directly
   to equilibrium. For trajectory computation this doesn't matter.

3. **The full state vector**: theta, phi, thetad, phid are no longer
   tracked. Only z_eff (from delta) and (u_drift, v_drift) are computed.
   This is sufficient for the parcels use case.

### Assumptions

- The ocean velocity varies slowly compared to the mechanical relaxation
  time (~10–30s). Valid for typical ocean flows and parcels timesteps of
  minutes to hours.
- The velocity profile is vertically smooth so that the fixed-point
  iteration converges. Pathological profiles (e.g., velocity increasing
  sharply at exactly the drogue depth) could cause non-convergence, but
  this is not physically realistic.
- The pole is always in the lower hemisphere (drogue below buoy). If
  the surface shear is strong enough to lift the drogue to the surface
  (delta → pi/2, z_eff → 0), the iteration converges to z_eff = 0 and
  the drifter simply follows the surface current.
