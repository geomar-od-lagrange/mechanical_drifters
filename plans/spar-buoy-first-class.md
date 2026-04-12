# Package rename and PointSurfaceDrifter

Rename `drogued_drifters` → `mechanical_drifters`. Drop SparBuoy (it was
a depth-average hack that doesn't belong in the Lagrangian mechanics
hierarchy). Add PointSurfaceDrifter as a second model that validates the
full pipeline. Clean up tech debt.

## 1. Drop SparBuoy

The current SparBuoy doesn't use Lagrangian mechanics — it just averages
velocity over depth. That forced it to fake the base class contract
(dummy State, null `_qdd_func`, duplicated `__init__` validation). When
real SparBuoy physics arrive (with actual EOM), it enters as a proper
`LagrangianMechanicsModel` subclass. Until then, depth-averaged drift is
a utility, not a model.

- Delete `models/spar_buoy.py`
- Delete `tests/test_spar_buoy.py`
- Remove SparBuoy exports from `__init__.py`

## 2. Package rename: `drogued_drifters` → `mechanical_drifters`

Pre-alpha, no downstream users, internal API changes are free.

| What | From | To |
|------|------|----|
| Package dir | `src/drogued_drifters/` | `src/mechanical_drifters/` |
| pyproject.toml name | `drogued_drifters` | `mechanical_drifters` |
| All imports | `drogued_drifters` | `mechanical_drifters` |
| README title | "2025 Drogued Drifters" | "Mechanical Ocean Drifters" |

One `git mv` + project-wide find-and-replace in all `.py` and `.md`
files. Sync all notebooks after.

## 3. Tech debt cleanup

- Delete `make_dd_kernel` from `parcels.py`. Use `make_kernel(model)`
  everywhere.
- Fix unused `StatusCode` imports in `test_drifter_parcels.py`.
- Narrow `except Exception` → `except (OSError, pickle.UnpicklingError,
  KeyError)` in `eom.py` cache loading.
- Group `__init__.py` exports by model so DroguedDrifter-specific types
  (`DrifterPhysics`, `EOMState`) are clearly scoped.

## 4. Add PointSurfaceDrifter

A point particle at the surface with quadratic drag. Validates that the
full Lagrangian machinery works for a model other than DroguedDrifter.

Physics: one body with mass `m` and drag coefficient `k` at z = 0.
Two generalized coordinates (x, y). The Lagrangian is
`L = ½(m + m_tilde)(xd² + yd²)`, drag force `F = -k|v-u|(v-u)`.
At steady state: drift = surface current (trivially verifiable).

Goes through the full `_derive_symbolic` → cache → `_make_qdd_func`
pipeline.

`models/point_surface_drifter.py` with `PointSurfacePhysics`,
`PointSurfaceState`, `PointSurfaceDrifter`.

## 5. Example directory restructure

Current:
```
examples/
  eom_study/             # DroguedDrifter EOM exploration
  idealized_flow/        # DroguedDrifter in idealized flows
  baltic_drifters/       # DroguedDrifter validation pipeline
```

Proposed:
```
examples/
  drogued_drifter/       # all DroguedDrifter examples
    01_eom_exploration
    02_synthetic_flow_profiles
    03_sheared_jet_parcels
    04_wave_orbitals
  point_drifter/         # PointSurfaceDrifter examples
    01_surface_tracking
  baltic_validation/     # validation pipeline with real data
    00_extract_science_periods
    01_fetch_cmems_data
    ...
```

Merge `eom_study/` and `idealized_flow/` into `drogued_drifter/`.
Rename `baltic_drifters/` → `baltic_validation/` (it's a validation
pipeline, not a model demo).

## 6. README and docs

- README title: "Mechanical Ocean Drifters"
- README subtitle: mention DroguedDrifter and PointSurfaceDrifter.
- `docs/drifter-model.md`: stays DroguedDrifter-specific.
- `docs/parcels-v4-coupling.md`: drop `make_dd_kernel` references,
  note `make_kernel(model)` works for any model.
- `base.py` docstring: mention both models.

## 7. Checklist

- [ ] Delete `models/spar_buoy.py`, `tests/test_spar_buoy.py`
- [ ] Remove SparBuoy from `__init__.py`
- [ ] `git mv src/drogued_drifters src/mechanical_drifters`
- [ ] Find-and-replace `drogued_drifters` → `mechanical_drifters`
- [ ] Update `pyproject.toml`
- [ ] Delete `make_dd_kernel` from `parcels.py` and all call sites
- [ ] Fix `StatusCode` imports, narrow exception handler
- [ ] Implement `PointSurfaceDrifter`
- [ ] Write tests for `PointSurfaceDrifter`
- [ ] Restructure `examples/` directories
- [ ] Write `examples/point_drifter/01_surface_tracking.md`
- [ ] Update README, docs, `base.py` docstring
- [ ] Sync and execute all notebooks
- [ ] Run full test suite
- [ ] Update PR #15 checklist
