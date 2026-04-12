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

### docs/parcels-v4-coupling.md

Keeps the "v4" — the coupling approach (numpy kernels,
`fieldset.UV.eval()`) requires Parcels v4 and wouldn't work with v3.
Updates:

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

## 7. Execution phasing

Each phase depends on the previous one completing. Within a phase,
steps run sequentially unless noted otherwise.

### Phase 1 — Cleanup and restructure (Sonnet subagents, sequential)

Mechanical changes. Run tests after each step to confirm nothing broke.

- [ ] **1a. Drop SparBuoy** — delete `models/spar_buoy.py`,
  `tests/test_spar_buoy.py`, remove SparBuoy exports from `__init__.py`
- [ ] **1b. Package rename** — `git mv src/drogued_drifters
  src/mechanical_drifters`, find-and-replace in all `.py`, `.md`,
  `pyproject.toml`
- [ ] **1c. Tech debt** — delete `make_dd_kernel` from `parcels.py` and
  all call sites, fix `StatusCode` imports, narrow `except Exception` in
  `eom.py`
- [ ] **1d. Examples restructure** — merge `eom_study/` +
  `idealized_flow/` → `examples/drogued_drifter/`, rename
  `baltic_drifters/` → `examples/baltic_validation/`, update all
  cross-references in notebooks and docs

### Phase 2 — PointSurfaceDrifter (TDD, two rounds)

Red/Green agents, Opus reviews. Two rounds: interface then behavior.
Directory structure is final (Phase 1d done), so the notebook lands in
its permanent location.

- [ ] **2a. Red 1 (interface)** — write tests: class exists, has
  `PointSurfacePhysics`/`PointSurfaceState`, correct `n_q=2`,
  `_drift_velocity_indices`, `state_size`, constructs with default physics
- [ ] **2b. Green 1** — implement skeleton: class, dataclasses, class
  attributes. No physics yet. Tests from 2a pass.
- [ ] **2c. Red 2 (behavior + integration)** — write tests:
  `_derive_symbolic` produces correct EOM, `steady_state_batch` returns
  surface current exactly, `make_kernel` produces a callable. Also write
  `examples/point_drifter/01_surface_tracking.md` as an integration test
  (notebook must execute and produce correct results).
- [ ] **2d. Green 2** — implement `_derive_symbolic` (the trivial
  Lagrangian), `_rhs_batch`, `_max_depth`. Unit tests and notebook pass.
- [ ] **2e. Blue (review)** — Opus review of the full
  PointSurfaceDrifter implementation.

### Phase 3 — Documentation and final pass (Opus, main agent)

Judgment calls on content and framing. After Phases 2–3 are done.

- [ ] Rewrite README.md
- [ ] Write `docs/architecture.md`
- [ ] Write `docs/point-surface-drifter.md`
- [ ] Update `docs/drifter-model.md` (import paths, add PSD note)
- [ ] Update `docs/parcels-v4-coupling.md` (drop `make_dd_kernel`,
  import paths)
- [ ] Update `docs/stokes-drift.md` (import path)
- [ ] Update `base.py` docstring
- [ ] Sync and execute all notebooks
- [ ] Run full test suite
- [ ] Update PR #15 checklist and title
