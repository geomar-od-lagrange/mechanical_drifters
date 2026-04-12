# Implementation plan: Architecture A (release refactor)

Based on [revised-architecture.md](revised-architecture.md). Conservative
rewire — clean the internals, keep the public surface stable.

## Goal

Cleaner module boundaries, one velocity protocol, no adapters, no mutable
state hacks. Prepare the ground for multi-object generalization (proposal
D) in a future release.

## Preconditions

All changes are on branch `wr/go-for-real-application`. Tests must pass
after every step. Run `pixi run pytest` to verify.

---

## Step 1: Create `coords.py`

**What:** Move `_uv_to_theta`, `_uv_to_spherical`, `_spherical_to_uv`
from `lagrange_model.py` to new `src/drogued_drifters/coords.py`.

**Changes:**
- Create `src/drogued_drifters/coords.py` with the three functions
  (verbatim, keep underscore prefix).
- In `lagrange_model.py`: delete the three functions, add
  `from .coords import _uv_to_theta, _uv_to_spherical, _spherical_to_uv`
  (so existing importers of `lagrange_model` still work).
- In `drifter.py`: change import to `from .coords import ...`.

**Test impact:** No test changes needed — tests import from
`lagrange_model` and `drifter`, both of which re-export.

**Verification:** `pixi run pytest`

---

## Step 2: Create `velocity.py`

**What:** Move `make_profile_sampler` from `parcels_v4.py` to new
`src/drogued_drifters/velocity.py`.

**Changes:**
- Create `src/drogued_drifters/velocity.py` with `make_profile_sampler`
  (verbatim).
- In `parcels_v4.py`: delete `make_profile_sampler`, add
  `from .velocity import make_profile_sampler`.
- Update `__init__.py` if `make_profile_sampler` is re-exported (it
  isn't currently, so no change).

**Test impact:** Update imports in:
- `tests/test_drifter_parcels.py`: `from drogued_drifters.parcels_v4 import make_profile_sampler`
  → `from drogued_drifters.velocity import make_profile_sampler`
- `tests/test_integration_full_chain.py`: same change.

Or: keep the re-export in `parcels_v4.py` so test imports still work.
Preferred: update test imports (cleaner, avoids stale re-exports).

**Verification:** `pixi run pytest`

---

## Step 3: Rename `lagrange_model.py` → `eom.py`

**What:** Rename the module. This is pre-alpha, so just rename and
update all imports. No re-export shim.

**Changes:**
- `git mv src/drogued_drifters/lagrange_model.py src/drogued_drifters/eom.py`
- Update all imports across the codebase:

| File | Old import | New import |
|---|---|---|
| `drifter.py` | `from drogued_drifters.lagrange_model import ...` | `from .eom import ...` |
| `__init__.py` | `from .lagrange_model import ...` | `from .eom import ...` |
| `tests/conftest.py` | `from drogued_drifters.lagrange_model import ...` | `from drogued_drifters.eom import ...` |
| `tests/test_lagrange_physics.py` | `from drogued_drifters.lagrange_model import ...` | `from drogued_drifters.eom import ...` |
| `tests/test_drogued_drifter.py` | `from drogued_drifters.lagrange_model import ...` | `from drogued_drifters.eom import ...` |
| `tests/test_lagrange_symbolic_fallback.py` | `from drogued_drifters.lagrange_model import ...` | `from drogued_drifters.eom import ...` |
| `tests/test_numerical_edge_cases.py` | `from drogued_drifters.lagrange_model import ...` | `from drogued_drifters.eom import ...` |

Also grep for string references like `"lagrange_model"` in test
fixtures or monkeypatching.

**Test impact:** Import path changes only. No logic changes.

**Verification:** `pixi run pytest`

---

## Step 4: Cache `_make_qdd_func`, delete `_qdd_func`

**What:** Add `@functools.lru_cache()` to `_make_qdd_func` so repeated
calls with the same backend return the same function object. Delete the
`_qdd_func` convenience wrapper (it hardcodes numpy and creates a new
closure each call).

**Changes in `eom.py`:**
- Add `@functools.lru_cache()` decorator to `_make_qdd_func`.
- Delete the `_qdd_func` function entirely.

**Test impact:** Grep for `_qdd_func` in tests:
- `tests/test_lagrange_physics.py` and `tests/test_drogued_drifter.py`
  may import it. Replace with `_make_qdd_func("numpy")`.

**Verification:** `pixi run pytest`

---

## Step 5: Add `backend=` to public `qdd_func`

**What:** The public `qdd_func` currently hardcodes numpy. Add a
`backend="numpy"` keyword parameter.

**Changes in `eom.py`:**
```python
def qdd_func(physics: DrifterPhysics, state: EOMState, *, backend="numpy"):
    return _make_qdd_func(backend)(physics, state)
```

**Test impact:** None — existing callers don't pass `backend=`, so
the default preserves behavior.

**Verification:** `pixi run pytest`

---

## Step 6: Kill the adapter, unify velocity protocol

**What:** Delete `_adapt_get_uv`, the `get_uv` constructor parameter,
and `_default_uv`. One velocity protocol: `sample_uv(z) -> (U, V)`.

**Changes in `drifter.py`:**
- Delete `_adapt_get_uv` function.
- Delete `_default_uv` instance method.
- Move `_default_sample_uv` from static method to module-level function.
- In `DroguedDrifter.__init__`:
  - Remove `get_uv=None` parameter.
  - Remove the `get_uv is not None and sample_uv is not None` check.
  - Remove the `elif get_uv is not None` branch.
  - Remove the `else` branch that sets `self.get_uv`.
  - Simplify to:
    ```python
    self._sample_uv = sample_uv if sample_uv is not None else _default_sample_uv
    ```

**Test impact — this is the big one.** Many tests use `get_uv=`. Each
needs to be converted to `sample_uv=`. The conversion pattern:

Tests that define `def get_uv(*, t, x, y, z): ... return U, V` and pass
`DroguedDrifter(get_uv=get_uv)` need to be rewritten as:
```python
def sample_uv(z):
    z_arr = np.atleast_1d(np.asarray(z, float))
    # ... vectorized logic ...
    if np.asarray(z).ndim == 0:
        return float(U[0]), float(V[0])
    return U, V

dd = DroguedDrifter(sample_uv=sample_uv)
```

Files with `get_uv=` usage:
- `tests/test_drogued_drifter.py` (~8 call sites)
- `tests/test_integration_full_chain.py` (~6 call sites)
- `tests/test_numerical_edge_cases.py` (~5 call sites)

Also: tests that access `dd.get_uv(...)` directly (1 site in
`test_drogued_drifter.py` line 50-51) need rewriting.

Also: tests that monkeypatch `dd._sample_uv` (2 sites in
`test_numerical_edge_cases.py`) — these still work since
`_sample_uv` stays as the instance attribute.

**Verification:** `pixi run pytest`

---

## Step 7: Extract `_rhs`, `_rhs_batch`, `_z_eff` to module-level functions

**What:** Move the core ODE logic out of `DroguedDrifter` methods into
module-level functions with explicit `physics`, `qdd_func`, `sample_uv`
parameters. Class methods become one-line delegates. Eliminate the
save/restore trick in `get_final_drift_batch`.

**Changes in `drifter.py`:**

New module-level functions:
```python
def _z_eff(l, u, v):
    s = u**2 + v**2
    cos_theta = (s - 4) / (s + 4)
    return np.minimum(0.0, l * cos_theta)

def _rhs(t, y, *, physics, qdd_func, sample_uv):
    ...  # current self._rhs body with explicit params

def _rhs_batch(Y, *, physics, qdd_func, sample_uv):
    ...  # current self._rhs_batch body with explicit params
```

Class methods become delegates:
```python
class DroguedDrifter:
    def _rhs(self, t, y):
        return _rhs(t, y, physics=self.physics,
                    qdd_func=self._qdd_func, sample_uv=self._sample_uv)

    def _rhs_batch(self, Y, sample_uv=None):
        return _rhs_batch(Y, physics=self.physics,
                          qdd_func=self._qdd_func,
                          sample_uv=sample_uv or self._sample_uv)
```

`get_final_drift_batch` — eliminate save/restore:
```python
def get_final_drift_batch(self, *, sample_uv=None, t_span=(0, 120),
                          y0=None, atol=1e-3, rtol=1e-3):
    uv = sample_uv if sample_uv is not None else self._sample_uv
    return self._get_final_drift_batch_impl(t_span, y0, atol, rtol, uv)
```

And thread `uv` through `_get_final_drift_batch_impl` → `_rhs_batch`.

**Test impact:** Minimal — the class API is unchanged. Tests that
monkeypatch `dd._sample_uv` still work. Tests that call
`dd.get_final_drift_batch(sample_uv=...)` still work.

**Verification:** `pixi run pytest`

---

## Step 8: Slim `_extract_profiles`

**What:** Change `_extract_profiles(particles, fieldset, dd)` to
`_extract_profiles(particles, fieldset, drogue_depth)`. The Parcels
module no longer needs the full `DroguedDrifter` instance — just the
pole length.

**Changes in `parcels_v4.py`:**
- `_extract_profiles` signature: replace `dd` with `drogue_depth` (float).
- Replace `dd.physics.l` with `drogue_depth` inside.
- In `DDAdvectEE`: call `_extract_profiles(particles, fieldset, dd.physics.l)`.

**Test impact:** `tests/test_drifter_parcels.py` — any direct calls to
`_extract_profiles` need the new signature. Check if tests call it
directly (unlikely — they test via `DDAdvectEE` / `make_dd_kernel`).

**Verification:** `pixi run pytest`

---

## Step 9: Export `compute_stokes_profile`

**What:** Add `compute_stokes_profile` to `__init__.py`.

**Changes:**
```python
# __init__.py
from .stokes import compute_stokes_profile
```

**Test impact:** None.

**Verification:** `pixi run pytest`

---

## Step 10: New example notebook for EOM study

**What:** Create `examples/eom_study/01_eom_study.ipynb` that exercises
`DrifterPhysics`, `EOMState`, `qdd_func`, `M_func`, `F_func`, and the
drag/added-mass helpers.

**Changes:** New notebook. No source changes.

**Verification:** `pixi run papermill examples/eom_study/01_eom_study.ipynb examples/eom_study/01_eom_study.ipynb`

---

## Summary

| Step | Type | Risk | Files changed |
|---|---|---|---|
| 1. `coords.py` | Move | Low | 3 source, 0 test |
| 2. `velocity.py` | Move | Low | 3 source, 2 test |
| 3. Rename → `eom.py` | Move | Low | 9 source+test |
| 4. Cache `_make_qdd_func` | Refactor | Low | 1 source, ~2 test |
| 5. `backend=` on `qdd_func` | Enhancement | Low | 1 source, 0 test |
| 6. Kill adapter | Refactor | **Medium** | 1 source, **~19 test call sites** |
| 7. Extract to functions | Refactor | Medium | 1 source, ~0 test |
| 8. Slim `_extract_profiles` | Refactor | Low | 1 source, ~0 test |
| 9. Export stokes | Polish | Low | 1 source, 0 test |
| 10. Example notebook | Polish | Low | 1 new notebook |

Step 6 is the riskiest — ~19 test call sites need `get_uv` → `sample_uv`
conversion. All others are mechanical or low-impact.
