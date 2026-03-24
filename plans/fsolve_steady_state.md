# Plan: Replace ODE solver with fsolve on F=0 at steady state

## Background

The drogued drifter EOMs have the form M·q̈ = F. At steady state q̈ = 0,
so F = 0 — four algebraic equations in (u, v, θ, φ). Solving these with
`scipy.optimize.fsolve` takes ~150 μs per particle, vs ~60 ms for the
full ODE integration. That's a **400x speedup** per particle.

Benchmarked against the full ODE, fsolve matches to 4 significant figures
across uniform shear, rotated shear, opposing currents, and
depth-intensified flow.

## What we have now

- Full Lagrangian derivation in `lagrange_model.py` (sympy)
- M·q̈ = F decomposition via `sp.linear_eq_to_matrix`
- Steady-state F = 0 derived and lambdified in notebook 04
- Numerical verification that fsolve(F=0) matches the ODE
- Analytical result α = √k_b/(√k_b+√k_d) confirmed by both fsolve and
  ODE across all test cases

## Implementation plan

### 1. Add `get_steady_state_drift` to DroguedDrifter

```python
def get_steady_state_drift(self, *, U_func, V_func, x0=None):
    """Solve F=0 for steady-state drift given a velocity profile.

    Args:
        U_func: callable(z) -> eastward velocity [m/s]
        V_func: callable(z) -> northward velocity [m/s]
        x0: initial guess (u, v, theta, phi), or None for default

    Returns:
        (u_drift, v_drift, theta, phi, z_eff)
    """
```

Internally:
- Calls the lambdified F_ss with U_b=U_func(0), V_b=V_func(0),
  U_d=U_func(z_d), V_d=V_func(z_d) where z_d = -l·cos(θ)
- The self-consistent coupling between θ and z_d is handled by fsolve's
  Newton iteration — no separate fixed-point loop needed
- Warm start: pass previous (u, v, θ, φ) as x0

### 2. Lambdify F_ss once at init

The sympy derivation is slow (~15s). Cache the lambdified F_ss function,
same as the existing `_derive_and_lambdify()` pattern in `lagrange_model.py`.

### 3. Wire into parcels kernel

Per particle per timestep:

```python
# 1. Sample velocity profile at particle position (one fieldset call)
z_levels = fieldset.depth_levels  # or np.linspace(0, l, 10)
U_prof = fieldset.U[time, z_levels, lat, lon, particle]
V_prof = fieldset.V[time, z_levels, lat, lon, particle]
U_func = interp1d(z_levels, U_prof)
V_func = interp1d(z_levels, V_prof)

# 2. Solve F=0 (~150 μs, warm-started from previous timestep)
u, v, theta, phi, z_eff = dd.get_steady_state_drift(
    U_func=U_func, V_func=V_func, x0=prev_state,
)

# 3. Apply drift
dlon += u * dt
dlat += v * dt
```

The fieldset sampling is the most expensive part. The fsolve itself is
negligible (~5 Newton iterations × 1 lambdified evaluation each).

### 4. Vectorized variant

For N particles, fsolve must be called per-particle (each sees a different
velocity profile). But at ~150 μs/particle, N=210 takes ~30 ms — already
negligible compared to the parcels overhead.

If needed later: the α formula + vectorized fixed-point iteration on z_d
can process all particles simultaneously in ~1 ms total. This uses the
analytical result (α = √k_b/(√k_b+√k_d)) which is confirmed but not
derived assumption-free by sympy. The fsolve approach is more general.

## Expected performance

| Approach | Per particle | 210 particles × 288 steps | Total |
|---|---|---|---|
| Full ODE (solve_ivp) | 60 ms | 3.6 M ms | ~60 min |
| Batched ODE (current) | ~0.5 ms | 144 s | ~2:24 |
| fsolve(F=0) | 0.15 ms | 9 s | ~9 s |
| α + fixed-point | ~0.005 ms | 0.3 s | ~0.3 s |

The fsolve approach gives ~15x speedup over the current batched ODE with
no loss of generality. The α + fixed-point gives another ~30x but relies
on the analytical result.

## Notebook 04 status

The current notebook (04_steady_state_derivation) was critically reviewed.
Key issues to fix:

1. **Theta solve fails** — sympy cannot solve F_theta=0. Replace with:
   display the simplified equation, solve numerically via fsolve, and
   show the result matches tan(δ) = k_d·α²·S²/W.

2. **Alpha check silently fails** — Piecewise wrapper from Abs breaks
   simplify. Use numerical verification instead of symbolic assert.

3. **Ansatz not derived** — The drift ansatz is assumed. Add a cell that
   shows F_x and F_y share the same scalar factor (sympy can show this
   after ansatz substitution), proving the ansatz is the unique solution
   form. Alternatively, just solve numerically and observe that the
   result matches the ansatz.

4. **phi solutions not simplified** — `2*atan(tan(φ'/2))` is sympy's
   representation of φ = φ'. Note this in markdown.

5. **Add fsolve-based numerical solution** — solve F=0 directly for
   (u, v, θ, φ) without any ansatz, compare against analytical formulas.
   This is the honest path: derive F=0 symbolically, solve numerically,
   observe the patterns, verify analytically.

## What's not changing

- The full dynamic ODE solver (DroguedDrifter class) stays for transient
  dynamics and validation
- The sympy Lagrangian derivation stays as the source of truth
- The parcels integration pattern (custom kernel, fieldset sampling)
  stays the same

## Open questions

- Should `get_steady_state_drift` live on DroguedDrifter or be a
  standalone function? It doesn't need the get_uv callback pattern
  since it takes U_func/V_func directly.
- How to handle the fieldset z-levels in parcels? The fieldset may have
  non-uniform depth levels. Sample at the native levels and interpolate,
  or sample at a fixed set of levels?
- The fsolve initial guess matters for convergence. With warm starting
  from the previous timestep, convergence should be fast. But the first
  timestep needs a cold start — use (U_d, V_d, π, 0) as default.
