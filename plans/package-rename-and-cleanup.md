# Package rename and PointSurfaceDrifter

Rename `drogued_drifters` → `mechanical_drifters`. Drop SparBuoy (it was
a depth-average hack that doesn't belong in the Lagrangian mechanics
hierarchy). Add PointSurfaceDrifter as a second model that validates the
full pipeline. Clean up tech debt. Update all documentation.

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

## 6. Documentation

Per CLAUDE.md: docs describe what *is*, not what *was*. Each doc should
make sense on its own. No changelogs, no "previously we had X".

### README.md

Rewrite top section:
- Title: "Mechanical Ocean Drifters"
- Subtitle: what the package does (Lagrangian mechanics models for
  ocean drifters), what models it includes (DroguedDrifter,
  PointSurfaceDrifter), how they work (sympy → scipy → xarray).
- Quick start: show DroguedDrifter (the primary model).
- Parcels section: use `make_kernel(model)` (no `make_dd_kernel`).
- Examples section: reorganized to match the new directory structure.

### docs/drifter-model.md

Stays DroguedDrifter-specific — this is the physics and API reference
for the drogued drifter model. Updates needed:

- All import paths: `mechanical_drifters` instead of `drogued_drifters`.
- All function names: `eval_qdd`, `eval_M`, `eval_F` (already done in
  the v5 commit but verify).
- Add a brief note at the top that the package also contains
  PointSurfaceDrifter, with a link to its doc.

### docs/parcels-v4-coupling.md → docs/parcels-coupling.md

Rename (drop the "v4" — it's not about the Parcels version, it's about
how this package couples to Parcels). Updates:

- All import paths: `mechanical_drifters`.
- Drop all `make_dd_kernel` references. The entry point is
  `make_kernel(model)`.
- The doc is already mostly generic (it describes how profile extraction,
  ODE integration, and position update work for any model). Verify that
  no DroguedDrifter-specific assumptions leak through.
- Add a sentence noting that `make_kernel` works for any
  `LagrangianMechanicsModel` subclass.

### docs/stokes-drift.md

Already generic (no model-specific references). Only change: import
path update `mechanical_drifters.stokes` instead of
`drogued_drifters.stokes`.

### NEW: docs/point-surface-drifter.md

Short doc for the PointSurfaceDrifter:
- What it is: a point particle at the surface with quadratic drag.
- Why it exists: baseline comparison, pipeline validation.
- Physics: the trivial Lagrangian derivation (inline, not referencing
  sympy code).
- API: constructor, `steady_state_batch`, `make_kernel`.
- Steady-state property: drift equals surface current exactly.

### NEW: docs/architecture.md

Overview of the package architecture. Covers:
- `LagrangianMechanicsModel` base class: what subclasses must provide
  (Physics, State, n_q, `_drift_velocity_indices`, `default_physics`,
  `_derive_symbolic`, `_rhs_batch`, `_max_depth`), what they get for
  free (`steady_state_batch`, `make_kernel`, `state_size`, `_cache_path`).
- The EOM pipeline: symbolic derivation → pickle cache → lambdify with
  CSE → `_build_packer` → `_make_qdd_func`. How caching works (keyed
  by class name, invalidated on source or sympy version change).
- How to add a new model: implement four class attributes and four
  methods, get ODE integration and Parcels coupling for free.
- Module layout and dependency graph.

This replaces the architectural knowledge currently buried in the plan
files and in code comments. Per CLAUDE.md: agents get context by reading
`docs/*.md`.

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
- [ ] Rewrite README.md
- [ ] Update `docs/drifter-model.md` (import paths, add note)
- [ ] Rename and update `docs/parcels-v4-coupling.md` → `docs/parcels-coupling.md`
- [ ] Update `docs/stokes-drift.md` (import path)
- [ ] Write `docs/point-surface-drifter.md`
- [ ] Write `docs/architecture.md`
- [ ] Update `base.py` docstring
- [ ] Sync and execute all notebooks
- [ ] Run full test suite
- [ ] Update PR #15 checklist
