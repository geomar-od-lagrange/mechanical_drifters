# Branch summary: wr/go-parcels-v4

## 1. Parcels v4 integration
- Added parcels v4 (from GitHub main) to the pixi environment, plus dependencies (netcdf4, pyproj, cartopy, papermill)
- Created `examples/parcels_3d_flow.py` — custom kernel coupling DroguedDrifter to parcels v4
- Built up from simple exponential flow → Ekman rotation → meandering Gaussian jet → two opposing jets
- 210 particles, 24h simulations, OOB particle deletion
- Added `z_eff` diagnostic (effective drogue depth) as a custom particle variable

FB: Parcels v4 integration is _the_ central pivot point. Let's set up a notebook (! for human consumption) with the idealized opposing jet flow?  

## 2. Performance optimization
- **Vectorized `_rhs_batch`**: Hand-coded the mass matrix M and force vector F as numpy array operations, verified against sympy-lambdified scalar version
- **Batched `get_final_drift_batch`**: Stacks N particles into one `(8N,)` solve_ivp call — 9.3x speedup for N=21
- **Convergence detection**: Event-based early termination when drift accelerations drop below threshold
- **Warm starting**: Pass previous solution as initial condition
- **Smooth phi regularization**: Replaced hard cutoff near θ=π with continuous damping (`phi_reg_eps`, `phi_reg_nu`), eliminating the discontinuous RHS

FB: The vectorized method needs a firmer base. Let's have sympy create the numpy code and use it in a ufunc? I don't want to break the chain from sympy to what we actually use in the end as it's the only reliable way forward for, e.g., incorporating different physics / objects.

## 3. Analytical steady-state solution
- **α = √k_b/(√k_b+√k_d)**: Derived that the drift velocity is a fixed weighted average of buoy and drogue currents. Verified numerically across all flow conditions.
- **Torque balance for z_eff**: `tan(δ) = k_d α² S² / W` — implicit 1D equation, solved by fixed-point iteration
- **fsolve on F=0**: 150μs per particle (400x faster than ODE). Lambdified the steady-state equations, verified against full ODE.
- **Notebook 04**: Sympy derivation from the Lagrangian — M·q̈=F → F=0 at steady state → solve for α, φ, θ. Uses U(z), V(z) as abstract functions and S(z_d) throughout. Critically reviewed by opus agent (issues identified: theta solve fails in sympy, ansatz not derived).

FB: Let's defer the alpha and the z_eff approach. Maybe we just leave it in the plans/ dir. And let's for now also defer the steady-state F=0 way. This is relatively easy to incroporate later once we have the full-dynamics stokes profile parcels integration.

## 4. Plans
- `plans/parcels_v4_integration.md` — original design for the parcels coupling
- `plans/analytical_steady_state.md` — the α formula, z_eff iteration, and what it means
- `plans/fsolve_steady_state.md` — fsolve approach, precomputed drift field option, interpolation error analysis (1-2% on typical grids), performance table

## 5. Baltic Sea wave/current analysis
- **Notebook 05** (`05_baltic_stokes_drift`): CMEMS Baltic wave model, Kiel Bight, Stokes drift maps (mean streamlines, max, RMS)
- **Notebook 06** (`06_baltic_surface_currents`): CMEMS Baltic physics, surface currents at depth 0
- **Notebook 07** (`07_stokes_vs_currents`): Southern Kiel Bight comparison. Stokes is 80% of current on average. 2×2 panels at 0m, 1.5m, 3m depth. Time series with blue=Stokes, red=current styling. Key finding: Stokes drift drops 80% by 3m depth while currents barely change.

## 6. Drifter in explicit waves
- **Notebook 08** (`08_drifter_in_waves`): Full ODE in monochromatic waves, JONSWAP spectrum, and 3-component CMEMS wave field
- Key finding: **Stokes profile approach overestimates drift by ~60 mm/s (5 km/day)** because the pole pendulum (T_p=27s) can't follow wave-frequency forcing (T_wave=2-4s). The drifter acts as a low-pass filter; effective α→0 for Baltic wind seas.
- Pendulum eigenfrequency analysis explains the filtering
- Multi-seed verification (10 seeds): −62 ± 3 mm/s bias, robust

FB: (for 5 and 6 together) Besides the minimal / idealized flow parcels integration, this is another branch we _need_ to get right before bringing together the actual drifter simulations with Parcels in the CMEMS flow. Main takeaways are that using the stokes profile is okay-ish, and a recipe for how to build an effective current (currents + stokes profile) for use in Parcels.

## 7. Drifter observation data
- **Notebook 09** (`09_collect_drifter_data`): Collected 3866 CSV snapshots → 12,264 records, 6 drifters (298-303), Apr 20 – May 9 2023
- Cartopy/OSM map of trajectories
- Detailed analysis of drifter 303: deployment → drift → beaching on Langeland → trash truck → facility management depot

FB: This needs to be condensed into a drifter dataset for the clearly drifting phases of the 6 drifters. We'll then use this for initial conditions (at deployment but likely also along the way) and for validation of the simulated tracks. Note that there's some uncertainty in the added mass, buoyancy correction and drag coefficients of the drogued-drifter components. Once everything is running, we'll have to do some parameter revisions / tuning / validations.

## 8. Code changes to `drifter.py`
- Added `_rhs_batch()` — vectorized RHS
- Added `get_final_drift_batch()` — batched solve_ivp with convergence detection and warm starting
- Changed `rhs()` — smooth phi regularization (replacing hard cutoff)
- New constructor params: `phi_reg_eps`, `phi_reg_nu`

FB: Let's stay on a more conceptual level. Maybe some of these changes will be kept. Maybe they will be dropped and reimplemented. Let's feel free to remove anything we don't feel is worth keeping.

## Files created/modified

**New files:**
- `examples/parcels_3d_flow.py`
- `examples/output/` (zarr stores, plots)
- `notebooks/04_steady_state_derivation.ipynb`
- `notebooks/05_baltic_stokes_drift.ipynb`
- `notebooks/06_baltic_surface_currents.ipynb`
- `notebooks/07_stokes_vs_currents.ipynb`
- `notebooks/08_drifter_in_waves.ipynb`
- `notebooks/09_collect_drifter_data.ipynb`
- `plans/parcels_v4_integration.md`
- `plans/analytical_steady_state.md`
- `plans/fsolve_steady_state.md`
- `data/drifters_kiel_bight.csv`

**Modified files:**
- `pyproject.toml` (added parcels, dependencies, hatch direct-references)
- `pixi.lock`
- `src/drogued_drifters/drifter.py` (batch methods, regularization)
