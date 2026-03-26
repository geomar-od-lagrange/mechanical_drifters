# Roadmap: Drogued drifter simulations in the Baltic

## Track A: Parcels integration ✓

### A1. Polished idealized example ✓
- `examples/baltic_drifters/01_idealized_flow.ipynb`

### A2. Point-particle runs in CMEMS data ✓
- `examples/baltic_drifters/02_cmems_point_particles.ipynb`

### A3. Drogued drifter kernel in CMEMS data ✓
- `examples/baltic_drifters/03_cmems_drogued_drifter.ipynb`
- deg/s conversion, z=0 layer, surface + 3m + DD compared

## Track B: Building the right sheared current ✓

### B1. Wave and current analysis ✓
- `examples/baltic_drifters/04_stokes_analysis.ipynb`

### B2. Wave orbital effects ✓
- `examples/baltic_drifters/05_wave_orbital_effects.ipynb`
- Stokes profile overestimates by ~60 mm/s (pendulum filtering)

### B3. Stokes drift profile builder ✓
- `examples/baltic_drifters/06_effective_current_fields.ipynb`

## Track C: Bringing it together

### C1. Clean drifter dataset ✓
- `examples/baltic_drifters/07_clean_drifter_data.ipynb`
- 6 drifters, science phases extracted (boat ride, beaching, pickups removed)
- `examples/baltic_drifters/data/drifters_clean.csv`

### C2. Drifter simulations in effective currents ✓
- `examples/baltic_drifters/08_drifter_in_effective_currents.ipynb`

### C3. Validation: deployment simulations ✓
- `examples/baltic_drifters/09a_simulation.ipynb` (simulation)
- `examples/baltic_drifters/09b_validation_plots.ipynb` (plots + metrics)
- CMEMS effective coastline from static mask dataset
- Beaching detection + area-between-tracks metric (shapely)
- DD: 0.96 km avg separation over ~44h
- A-grid beaching limits forward simulation utility

### C4. Validation: re-seeded simulations
- Lagged re-initialization: every N hours, re-deploy virtual drifters at
  the observed positions
- Run each re-seeded ensemble forward for a fixed window (24h, 48h)
- Compute separation distance as function of lead time, averaged over
  all re-seedings
- Literature: Liu & Weisberg (2011, doi:10.1029/2010JC006837)

### C5. Parameter sensitivity
- Vary drifter parameters (k_b, k_d, added masses) within physically
  plausible ranges
- Check whether the Callies et al. defaults are adequate or tuning is
  needed

### C6. Along-track velocity validation ✓
- `examples/baltic_drifters/10_along_track_validation.ipynb`
- α-weighted prediction has lowest RMSE for every drifter (0.197 m/s)
- The α = √k_b/(√k_b+√k_d) formula is the best velocity predictor

## Track D: Code quality

### D1. Sympy → numpy code generation
### D2. Phi regularization
### D3. Notebook cleanup (final refactor)
- Papermill-ready parameters (primitives only, explicit random seeds)
- Drop summaries with hardcoded numbers, drop custom figsize
- "tracer" → "point particle" everywhere

### D4. Document the α formula properly
### D5. What to defer (keep in plans/)
- Analytical steady-state, precomputed drift field, performance optimization
