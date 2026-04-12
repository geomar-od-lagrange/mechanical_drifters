# Multi-model first-class citizenship

Rename the package to `mechanical_drifters`. Elevate SparBuoy to equal
standing. Add a PointSurfaceDrifter as a third model that validates the
full Lagrangian machinery. Reorganize examples by model type.

## 1. Package rename: `drogued_drifters` → `mechanical_drifters`

Pre-alpha, no downstream users, internal API changes are free.

### What changes

| What | From | To |
|------|------|----|
| Package dir | `src/drogued_drifters/` | `src/mechanical_drifters/` |
| pyproject.toml name | `drogued_drifters` | `mechanical_drifters` |
| All imports | `drogued_drifters` | `mechanical_drifters` |
| README title | "2025 Drogued Drifters" | "Mechanical Ocean Drifters" |

One `git mv` + project-wide find-and-replace in all `.py` and `.md`
files. Sync all notebooks after.

## 2. Drop backward-compat aliases

Delete `make_dd_kernel`. The generic entry point is `make_kernel(model)`.
No backward compatibility is needed.

## 3. Naming sweep: model-specific types

`DrifterPhysics` and `EOMState` are DroguedDrifter-specific. The SparBuoy
will eventually have its own full Lagrangian derivation with its own
Physics and State types. Current naming is fine (`DrifterPhysics` clearly
belongs to the drogued drifter), but ensure:

- They are NOT re-exported from `__init__.py` as if they were
  package-wide types. They should be importable from
  `mechanical_drifters.models.drogued_drifter` or from the top-level
  for convenience, but documented as DroguedDrifter-specific.
- The `__init__.py` groups exports by model so it's clear which types
  belong to which model.
- `base.py` never references `DrifterPhysics` or `EOMState` — it
  works with `model.Physics` and `model.State` generically. (Already
  true.)

## 4. Add PointSurfaceDrifter

A point particle at the surface. No pole, no drogue, no tilt dynamics.
Drifts with the surface current. Useful as:

- A baseline for comparison ("what would a point particle do?").
- A validation that the full Lagrangian machinery (symbolic derivation,
  cache, lambdify, qdd_func, steady_state_batch, make_kernel) works
  for a model other than DroguedDrifter.

Physics: trivial. One body with mass `m` and drag coefficient `k` at
z = 0. Two generalized coordinates (x, y). The Lagrangian is
`L = ½(m + m_tilde) (xd² + yd²)`, the drag force is
`F = -k |v - u| (v - u)`. The EOM reduce to
`(m + m_tilde) qdd = -k |v - u| (v - u)` which at steady state gives
`v = u` (the particle tracks the current exactly).

Goes through the full `_derive_symbolic` → cache → `_make_qdd_func`
pipeline. The steady-state result is trivially verifiable: drift = surface
current.

`models/point_surface_drifter.py` with `PointSurfacePhysics`,
`PointSurfaceState`, `PointSurfaceDrifter`.

## 5. Example directory restructure

Current:
```
examples/
  eom_study/             # DroguedDrifter EOM exploration
  idealized_flow/        # DroguedDrifter in idealized flows
  baltic_drifters/       # DroguedDrifter validation with real data
```

Proposed:
```
examples/
  drogued_drifter/       # all DroguedDrifter examples
    01_eom_exploration
    02_synthetic_flow_profiles
    03_sheared_jet_parcels
    04_wave_orbitals
  spar_buoy/             # SparBuoy examples
    01_sheared_flow
  point_drifter/         # PointSurfaceDrifter examples
    01_surface_tracking
  baltic_validation/     # multi-model validation with real data
    00_extract_science_periods
    01_fetch_cmems_data
    ...
```

The drogued drifter examples merge `eom_study/` and `idealized_flow/`
into one directory. Baltic validation stays separate (it's a pipeline,
not a single-model demo) but gets a model-neutral name.

Each model's example directory has a self-contained notebook that
demonstrates the model standalone — no Parcels, no real data. Parcels
integration examples go in the model directory or in the validation
pipeline.

## 6. README and docs

- README: title "Mechanical Ocean Drifters", subtitle mentions all
  three models, quick-start shows DroguedDrifter and SparBuoy side by
  side.
- `docs/drifter-model.md`: stays DroguedDrifter-specific. Add a note
  at the top that the package includes other models.
- `docs/parcels-v4-coupling.md`: note that `make_kernel(model)` works
  for any model. Drop references to `make_dd_kernel`.
- `base.py` docstring: mention all three models.

## 7. Checklist

- [ ] `git mv src/drogued_drifters src/mechanical_drifters`
- [ ] Find-and-replace `drogued_drifters` → `mechanical_drifters`
- [ ] Update `pyproject.toml`
- [ ] Delete `make_dd_kernel` from `parcels.py`
- [ ] Implement `PointSurfaceDrifter` in `models/point_surface_drifter.py`
- [ ] Write tests for `PointSurfaceDrifter`
- [ ] Restructure `examples/` directories
- [ ] Write `examples/spar_buoy/01_sheared_flow.md`
- [ ] Write `examples/point_drifter/01_surface_tracking.md`
- [ ] Update README, docs, base.py docstring
- [ ] Sync and execute all notebooks
- [ ] Run full test suite
- [ ] Update PR #15 checklist
