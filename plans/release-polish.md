# Release polish: deferred items from review session

Everything below was identified during the E4/E5 review session and
deferred. For release, all items should be addressed.

## Architecture

### A1. Unify velocity interfaces (I1 from review-architecture.md)

`DroguedDrifter` has two incompatible velocity callbacks: `get_uv(t, x,
y, z)` for the scalar path and `sample_uv(z)` for the batch path. Unify
around `sample_uv` as the internal protocol. `get_uv` becomes a
convenience adapter at construction time.

**Files:** `src/drogued_drifters/drifter.py`
**Scope:** medium — changes constructor, `_rhs`, `_rhs_batch`,
`get_final_drift_batch` signature. Tests need updating.

### A2. Harden pickle cache key (S1 from review-architecture.md)

Add `sys.version_info[:2]` and `pickle.HIGHEST_PROTOCOL` to the cache
key in `_cache_key()`. Prevents silent wrong results when switching
Python or pickle versions.

**File:** `src/drogued_drifters/lagrange_model.py` (`_cache_key`)
**Scope:** trivial — add two fields to the hash.

### A3. Defensive copy in `make_profile_sampler` (S2 from review-architecture.md)

Add `.copy()` to the array conversions in `make_profile_sampler` so the
closure doesn't capture mutable references.

**File:** `src/drogued_drifters/parcels_v4.py`
**Scope:** trivial — two `.copy()` calls.

### A4. Public API for direct EOM evaluation (S3 from review-architecture.md)

Export `qdd_func` (without underscore) or document that `M_func` +
`F_func` + `np.linalg.solve` is the intended external interface. Update
`__init__.py`. Add a quick usage example to backlog.

**File:** `src/drogued_drifters/__init__.py`, `docs/drifter-model.md`
**Scope:** small — one export + docs update. Example deferred to backlog.

### ~~A5. Warm-start coordinate bypass~~ — DROPPED

The u/v ↔ theta/phi roundtrip is cheap. Keeping strict abstraction
boundaries is more important than micro-optimizing coordinate conversion.

### A6. Shared test fixtures (S5 from review-architecture.md)

Add `dd` fixture and common helpers to `tests/conftest.py`. Reduce
boilerplate in test files.

**File:** `tests/conftest.py`, all test files.
**Scope:** small — mechanical.

## Code quality

### Q1. Rename `_mag` → `_sym_norm` (N1 from review-code-quality.md)

**File:** `src/drogued_drifters/lagrange_model.py`
**Scope:** trivial — rename + grep for callers.

### ~~Q2. Rename `V` → `PE` for potential energy~~ — DROPPED

`L = T - V` is standard Lagrangian mechanics notation. Pythonic naming
conventions don't apply to physics symbols in sympy code.

### Q3. Add comment for `x_pos`/`y_pos` symbol naming (N4 from review-code-quality.md)

Explain why the static substitution symbols don't match struct field names.

**File:** `src/drogued_drifters/lagrange_model.py`
**Scope:** trivial — one comment.

### Q4. Add `warnings.warn` to cache load failure (P2 from review-code-quality.md)

The bare `except Exception: pass` in `_load_or_derive` silently swallows
cache errors. Add a warning for debuggability.

**File:** `src/drogued_drifters/lagrange_model.py`
**Scope:** trivial.

### Q5. `_derive_symbolic` return description (D9 from review-code-quality.md)

Verify this was fixed by architecture agent. If not, fix `F_static`
description.

**File:** `src/drogued_drifters/lagrange_model.py`
**Scope:** trivial.

## Documentation

### D1. LICENSE file — MIT

Add MIT LICENSE file, add one-liner to README.

**Scope:** one file + one line.

### D2. Stokes drift doc (D2 from review-documentation.md)

Write `docs/stokes-drift.md` covering `compute_stokes_profile`, the
deep-water monochromatic model, and how it's used in the Baltic pipeline
(notebook 02) and the wave orbitals example (notebook 03).

**File:** `docs/stokes-drift.md`
**Scope:** small — short doc, function is simple.

### D3. AdvectionEE vs RK4 note in README (N7 from review-documentation.md)

The DD kernel (`DDAdvectEE`) does its own Euler-forward position update
internally — it computes the steady-state drift velocity from the ODE
and applies it directly. It does not use Parcels' advection schemes
(`AdvectionRK4`, `AdvectionEE`). Those schemes only apply to
point-particle comparisons (as shown in notebook 02). Add one sentence
clarifying this in the README Parcels section.

**File:** `README.md`
**Scope:** trivial — one sentence.

## Status

All items complete (commit `2114e5e`). A1 was descoped to docstring
consistency (units [m/s], depth convention [m, positive upward]) instead
of full interface unification — both get_uv and sample_uv are honest
about their different calling patterns. A5 and Q2 were dropped with
rationale above.
