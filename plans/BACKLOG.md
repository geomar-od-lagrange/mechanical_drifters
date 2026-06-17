# Backlog

Ideas worth remembering but not worth planning yet. Items graduate to
the roadmap (and get their own plan file) when they become timely.

## Performance

- **Hashable xarray DataArrays for lru_cache.** Parcels' interpolation
  call chain (`_get_corner_data_Agrid` etc.) takes xr.DataArray args,
  blocking trivial `lru_cache` decorators. A hashable wrapper (or
  keying on `id(data)` + index `.tobytes()`) could cache corner data
  across calls with identical grid cell indices. See
  [parcels-interpolation-optimization.md](parcels-interpolation-optimization.md) for the full analysis.

- **Analytical steady-state solution.** Replace the ODE solve with a
  closed-form expression for the steady-state drift velocity. Would
  eliminate the ODE entirely. See
  [analytical_steady_state.md](analytical_steady_state.md).

- **fsolve on F=0.** Replace ODE integration with a nonlinear solve
  for the steady state (q_dd = 0). Cheaper than time-stepping but
  still numerical. See [fsolve_steady_state.md](fsolve_steady_state.md).

- **Time-averaged drift velocity.** The ODE final-step snapshot
  includes buoy oscillation about equilibrium. Averaging xd, yd over
  the last N seconds of integration would give a cleaner steady-state
  estimate and a better warm-start initial condition.

- **Warm-start + fixed-step ODE.** With warm-starting, the adaptive
  solver wastes effort on near-equilibrium states. A fixed-step
  Euler/RK4 with short t_span (~10 steps of 1s) would give
  predictable cost. Requires the time-averaged drift velocity work
  above for a meaningful convergence metric.

- **Batch depth levels into one `fieldset.UV.eval()` call.** Tile
  lat/lon D times and call eval once with `(D*N,)` arrays instead of
  D times with `(N,)` arrays. Would eliminate D-1 redundant grid
  searches. First attempt failed: Parcels' `_get_corner_data_Agrid`
  produces mismatched index array shapes when npart changes. Needs
  investigation into how Parcels handles the time dimension indexing
  with varying particle counts.

## Developer experience

- **Speed up test suite.** The symbolic derivation + lambdification
  during test setup dominates wall time. The EOM cache helps on
  repeated runs, but a cold cache (or cache invalidation after code
  changes) triggers multi-minute derivations. Worth profiling to find
  whether the bottleneck is sympy derivation, lambdification, or
  something else — and whether test-scoped fixtures can share more
  work.

## Upstream (Parcels)

- **`fieldset.UV.eval_profile(time, lat, lon, z_levels)`** — sample a
  full vertical profile in one call, reusing the horizontal grid
  search across depth levels. First-class profile sampling.

- **Truly 2D particles.** Particles that don't require artificial
  z=0 placement for surface-only models.

- **Depth as a broadcasting dimension.** Treat depth as an extra
  xarray dim rather than a particle coordinate, for cases where z is
  not a degree of freedom.

- **numpy fast-path in `_get_corner_data_Agrid`.** Skip xr.DataArray
  construction when data is already in memory. ~10-line change, would
  benefit all Parcels users. See
  [parcels-interpolation-optimization.md](parcels-interpolation-optimization.md) Phase 3.

- **Parcels v4 zarr output bug.** Custom kernels can produce
  incorrect output under certain conditions. See
  [parcels_v4_output_bug.md](parcels_v4_output_bug.md).

## Architecture

- **Parcels-integrated EOM stepping.** The current kernel solves the
  internal ODE to steady state each Parcels timestep, then hands back a
  drift velocity. An alternative mode would expose the EOM accelerations
  directly and let Parcels handle the time integration — the kernel
  evaluates `qdd` from the current state, updates generalized velocities
  via particle attributes (e.g. `pv_u`, `pv_v`, `qd_*`), and sets
  `dlon`/`dlat` for position. This keeps the full transient dynamics
  (no steady-state assumption) and lets Parcels control the timestep.
  Requires custom particle classes with velocity/state attributes.
  Interesting for inertial particles that never reach steady state, or
  for models where the steady-state assumption breaks down (fast-changing
  currents relative to the drifter response time).

- **SparBuoy pole-tilt dynamics.** `SparBuoySimple` ([spar-buoy.md](../docs/spar-buoy.md))
  exists and runs with depth-averaged drag. The remaining step is adding
  azimuth and zenith as generalized coordinates, deriving drag torques
  symbolically, and extending the EOM so the pole tilts under shear
  rather than remaining vertical.

- **SparBuoy wind forcing via signed-z fieldset.** The current Parcels
  coupling (`parcels._extract_profiles`) samples only the water column —
  it reads depths up to `_max_depth` (= draft), flips them to z-positive-up,
  and *extrapolates that water profile into the air*, so the above-surface
  levels never see wind. A proper wind-forced spar buoy needs a fieldset
  that provides water velocities at z ≤ 0 and air (wind) velocities at
  z > 0, plus a glue path that samples both media. Requires constructing
  a Parcels-v4 fieldset that merges ocean and atmosphere fields on a
  common signed-z axis and extending `_extract_profiles` to sample the
  air levels. A prototype notebook (`02_test_uv_profile`, an
  air-above/water-below SGRID fieldset) was dropped during the #21/#22
  consolidation because it tripped exactly this gap — revive it as the
  example once the glue handles air.

## Science

- **BSH-HBM integration.** Replace CMEMS Nemo-Nordic (~1.85 km,
  hourly, A-grid) with BSH-HBMnoku (~900 m, 15 min, C-grid) for the
  Kiel Bight drifter simulations. This would exercise the C-grid
  vector rotation path in the kernel. See
  [outlook_bsh_hbm_integration.md](outlook_bsh_hbm_integration.md).
