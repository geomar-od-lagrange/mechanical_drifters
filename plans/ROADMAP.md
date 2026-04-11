# Roadmap: Drogued drifter simulations in the Baltic

## Track A: Parcels integration ✓

### A1. Polished idealized example ✓
- `examples/idealized_flow/02_sheared_jet_parcels.ipynb`

### A2. Point-particle runs in CMEMS data ✓
- `examples/baltic_drifters/02_cmems_point_particles.ipynb`

### A3. Drogued drifter kernel in CMEMS data ✓
- `examples/baltic_drifters/03_cmems_drogued_drifter.ipynb`
- Parcels v4 coupling via `DDAdvectEE` kernel — [docs/parcels-v4-coupling.md](../docs/parcels-v4-coupling.md)

## Track B: Building the right sheared current ✓

### B1. Wave and current analysis ✓
- `examples/baltic_drifters/04_stokes_analysis.ipynb`

### B2. Wave orbital effects ✓
- `examples/baltic_drifters/05_wave_orbital_effects.ipynb`

### B3. Stokes drift profile builder ✓
- `examples/baltic_drifters/06_effective_current_fields.ipynb`

## Track C: Bringing it together

### C1. Clean drifter dataset ✓
- `examples/baltic_drifters/07_clean_drifter_data.ipynb`

### C2. Drifter simulations in effective currents ✓
- `examples/baltic_drifters/08_drifter_in_effective_currents.ipynb`

### C3. Validation: deployment simulations ✓
- `examples/baltic_drifters/09a_simulation.ipynb` (simulation)
- `examples/baltic_drifters/09b_validation_plots.ipynb` (plots + metrics)
- DD: 0.96 km avg separation over ~44h

### C4. Validation: re-seeded simulations
- Plan: [c4-reseeded-validation.md](c4-reseeded-validation.md)

### C5. Parameter sensitivity
- Plan: [c5-parameter-sensitivity.md](c5-parameter-sensitivity.md)

### C6. Along-track velocity validation ✓
- `examples/baltic_drifters/10_along_track_validation.ipynb`
- α = √k_b/(√k_b+√k_d) is the best velocity predictor (0.197 m/s RMSE)

## Track D: Code quality ✓

All items complete. Plans in [done/](done/).

- D-I: Parcels isolation — [docs/parcels-v4-coupling.md](../docs/parcels-v4-coupling.md)
- D-II: DrifterPhysics naming ✓
- D-III: u/v → u_stereo/v_stereo ✓
- Deferred items → [BACKLOG.md](BACKLOG.md)

## Track E: Release wrap-up

### E1. Optional numba backend for qdd evaluation
- 25x speedup on the lambdified EOM function (105 → 4 µs/call, N=6)
- 2.1x end-to-end in Baltic simulation (287 → 135s)
- Implementation: `make_dd_kernel(dd, backend="numba")` raises if
  numba not installed, no silent fallback
- Notes: [numba-acceleration.md](numba-acceleration.md)

### E2. README update
- Current README references old notebook names and doesn't mention
  the Parcels kernel or `parcels_v4.py`
