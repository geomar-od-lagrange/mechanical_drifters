# Roadmap: Drogued drifter simulations in the Baltic

## Track A: Parcels integration

Standalone: get the drogued drifter model running in parcels, from
idealized flows to real CMEMS data. No wave effects here.

### A1. Polished idealized example (notebook)
- Convert `examples/parcels_3d_flow.py` into a notebook in
  `examples/baltic_drifters/`
- Opposing meandering jets, multiple particles, z_eff diagnostic
- Human-readable: explain the kernel, the coupling, the physics

### A2. Point-particle runs in CMEMS data
- Load CMEMS Baltic physics into a parcels FieldSet
- Run passive tracers at surface, 1.5m, 3m in the Kiel Bight
- No drifter model, just AdvectionRK4 — proof of concept for the
  parcels + CMEMS pipeline

### A3. Drogued drifter kernel in CMEMS data
- Wire the DroguedDrifter kernel into the CMEMS FieldSet
- Sample U(z) profile at each particle, feed to the drifter model
- Proof of concept — no comparison against observations yet

## Track B: Building the right sheared current

Standalone: understand the wave-driven and Eulerian contributions to
near-surface velocity shear. Produce effective current fields. No
parcels here.

### B1. Wave and current analysis
- Consolidate notebooks 05-07: Stokes drift and Eulerian currents in
  the southern Kiel Bight
- Focus on the deployment period (Apr 20 – May 9, 2023)
- Key result: Stokes is ~80% of Eulerian at the surface but drops to
  ~20% by 3m depth

### B2. Wave orbital effects
- Notebook 08: drogued drifter in explicit monochromatic and 3-component
  (CMEMS partition) wave fields
- Key result: Stokes profile overestimates drift by ~60 mm/s because the
  pole pendulum (T_p=27s) can't follow wave-frequency forcing
- Pendulum eigenfrequency analysis explains the low-pass filtering
- This informs how much error the Stokes profile approximation introduces

### B3. Stokes drift profile builder
- Utility that takes CMEMS wave partition data (Hs_WW, T_WW, dir_WW,
  Hs_SW1, ...) and returns the effective current field:
  Eulerian(z) + Stokes(z) at each grid point
- This produces the input fields for parcels runs — but is independent
  of parcels itself

## Track C: Bringing it together

C is where Track A (parcels) and Track B (effective currents) converge.

### C1. Clean drifter dataset
- Filter each of the 6 drifters to their science phase (after
  deployment jump, before beaching/pickup)
- Flag/remove military zone kinks, trash truck rides, beaching periods
- Output: clean trajectories with timestamps, suitable for IC and
  validation
- Store in `examples/baltic_drifters/data/`

### C2. Drifter simulations in effective currents
- Combine A3 (parcels + drifter kernel) with B3 (effective current
  fields) for the Kiel Bight deployment period
- Deploy virtual drifters at observed locations/times
- First real comparison against observations

### C3. Validation: deployment simulations
- Simulate from deployment location/time, run for the full drift period
- Compare against observed trajectories
- Metric: separation distance vs lead time

### C4. Validation: re-seeded simulations
- Lagged re-initialization: every N hours, re-deploy virtual drifters at
  the observed positions
- Run each re-seeded ensemble forward for a fixed window (24h, 48h)
- Compute separation distance as function of lead time, averaged over
  all re-seedings
- Literature: Liu & Weisberg (2011) cumulative Lagrangian separation
  skill score

### C5. Parameter sensitivity
- Vary drifter parameters (k_b, k_d, added masses) within physically
  plausible ranges
- Check whether the Callies et al. defaults are adequate or tuning is
  needed

## Track D: Code quality

### D1. Sympy → numpy code generation
- Replace hand-coded `_rhs_batch` with sympy-generated numpy code
- Keep the full chain: Lagrangian → sympy → generated functions
- Enables changing the physics without manual re-derivation

### D2. Phi regularization
- Decide: smooth regularization vs coordinate transform
- Current smooth version works but parameters are ad-hoc

### D3. What to defer (keep in plans/)
- Analytical steady-state (α formula, z_eff iteration, fsolve)
- Precomputed drift velocity field approach
- Performance optimization beyond what's needed for ~1000 particles

## Immediate next steps

A and B proceed in parallel, converging at C.

**Track A:**
1. Set up the idealized notebook (A1)
2. Point-particle runs in CMEMS (A2)
3. Drifter kernel in CMEMS (A3)

**Track B:**
1. Consolidate wave/current analysis (B1)
2. Document wave orbital effects (B2)
3. Build the Stokes profile utility (B3)

**Then C:**
4. Clean drifter observations (C1)
5. First combined simulation (C2)
6. Validation (C3, C4)
