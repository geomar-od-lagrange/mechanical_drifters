# DW-F: Rationalize the drift-velocity API

## Decisions

1. **Rename internals:** `rhs` → `_rhs`, `default_uv` → `_default_uv`.
2. **Unify z_eff:** Rename `_z_eff_batch` → `_z_eff`, use in both `_rhs` and
   `_rhs_batch`. Drop `_uv_to_theta` usage in `_rhs` — compute cos(theta)
   directly from the stereographic identity, no angle detour.
3. **Drop early stopping everywhere.** No `converged` terminal event. Run for
   the full `t_span`, report `max_accel` at the end. User decides if converged.
4. **Report `max_accel`, not a boolean.** The user knows their tolerance.
5. **Shared `_solve` backend.** Both `get_full_solution` and `get_final_drift`
   call `_solve`. `get_full_solution` wraps in xarray. `get_final_drift` does not.
6. **Clean coordinate boundary.** Public in: spherical `(theta, phi)`. Public out:
   spherical. Warm-start `y0`: spherical. Internal: stereographic `(u, v)` only.

## API after

### Public

| Method | Returns | Use case |
|---|---|---|
| `get_full_solution(t_span, ...)` | `xr.Dataset` (full time series) | Analysis, plotting, one drifter |
| `get_final_drift(t_span, ...)` | `(xd, yd, max_accel)` | Quick scalar query, one drifter |
| `get_final_drift_batch(sample_uv, t_span, ...)` | `(xd, yd, Y_final, max_accel)` | Parcels integration, N drifters |

`Y_final` has columns `[x, y, theta, phi, xd, yd, thetad, phid]` — spherical
public coords. Theta and phi are in there; no separate unpacking needed.

`max_accel` is `max(|xdd|, |ydd|)` across all particles (batch) or scalar (single),
evaluated once at the end of integration by calling the RHS one more time.

### Internal

| Method | Role |
|---|---|
| `_solve(t_span, y0, ...)` | Raw `solve_ivp` wrapper. Internal coords. No xarray, no events. |
| `_rhs(t, y)` | Scalar ODE callback for `solve_ivp`. |
| `_rhs_batch(Y, sample_uv)` | Vectorized ODE callback for batch. |
| `_z_eff(u, v)` | Drogue depth from stereographic (u,v). Direct formula, no trig. Used by both `_rhs` and `_rhs_batch`. |
| `_default_uv(*, t, x, y, z)` | Testing fallback for `get_uv`. |

### Dropped

| What | Why |
|---|---|
| `converged` terminal event | Run for `t_span`, check after. Simpler, no doubled RHS cost. |
| Binary `converged` flag | Report `max_accel` instead. User decides threshold. |
| `_uv_to_theta` in `_rhs` | Use `_z_eff` (stereographic identity) everywhere. |
| `_eval_M_F` | Already gone (DW-C). |

## Changes

### 1. `drifter.py` — renames

- `def rhs(` → `def _rhs(`
- `self.rhs` reference in `_solve` → `self._rhs`
- `def default_uv(` → `def _default_uv(`
- `self.default_uv` reference in `__init__` → `self._default_uv`

### 2. `drifter.py` — unify `_z_eff`

- Rename `_z_eff_batch` → `_z_eff`. No signature change (already `self, u, v`).
- In `_rhs`: replace
  ```python
  theta = _uv_to_theta(u, v)
  z_d = float(min(0.0, self.physics.l * np.cos(theta)))
  ```
  with
  ```python
  z_d = float(self._z_eff(u, v))
  ```
- Drop `_uv_to_theta` import from drifter.py (verify it's not used elsewhere in file).

### 3. `drifter.py` — drop early stopping in `get_final_drift_batch`

Remove:
- `converged` function definition
- `converged.terminal = True`
- `converged.direction = -1`
- `events=converged` from `solve_ivp` call

Add after `solve_ivp` returns:
```python
# Evaluate convergence diagnostic: max drift acceleration at final state
Y_final_internal = sol.y[:, -1].reshape(N, 8)
dY_final = self._rhs_batch(Y_final_internal, sample_uv)
max_accel = float(np.max(np.abs(dY_final[:, IXD:IYD + 1])))
```

Change return to include `max_accel`:
```python
return Y_final[:, IXD], Y_final[:, IYD], Y_final, max_accel
```

(Dropping the separate `theta_final` return — it's in `Y_final[:, 2]`.)

### 4. `drifter.py` — `get_final_drift` returns `max_accel`

Currently calls `get_full_solution` and picks the last timestep. Change to call
`_solve` directly (like `get_final_drift_batch` does), then compute `max_accel`:

```python
def get_final_drift(self, *, t_span, ...):
    u0, v0, ud0, vd0 = _spherical_to_uv(theta, phi, thetad, phid)
    y0 = [x, y, u0, v0, xd, yd, ud0, vd0]
    sol = self._solve(t_span, y0)
    y_final = sol.y[:, -1]

    # Convergence diagnostic
    dy_final = self._rhs(0.0, y_final)
    max_accel = float(max(abs(dy_final[IXD]), abs(dy_final[IYD])))

    return float(y_final[IXD]), float(y_final[IYD]), max_accel
```

### 5. `drifter.py` — rename `_get_full_solution` → `_solve`

Shorter, clearer. Update callers:
- `get_full_solution` calls `self._solve(...)`
- `get_final_drift` calls `self._solve(...)`

### 6. `drifter.py` — update `get_final_drift_batch` return

Old: `(xd_final, yd_final, theta_final, Y_final)`
New: `(xd_final, yd_final, Y_final, max_accel)`

Theta/phi are in Y_final columns 2 and 3. No need to unpack separately.

### 7. Callers of `get_final_drift_batch` that unpack the return

- `make_dd_velocity_interpolator` in `drifter.py` — currently unpacks
  `xd_ms, yd_ms, theta, Y_final`. Update to `xd_ms, yd_ms, Y_final, max_accel`.
  (It doesn't use `theta`.)
- Tests — update unpacking.
- Notebooks — update unpacking.

### 8. `__init__.py`

No changes. `DroguedDrifter`, `DrifterPhysics`, `M_func`, `F_func` stay.

### 9. Tests

**Update:**
- All tests calling `get_final_drift_batch` — update return unpacking (drop `theta`,
  add `max_accel`). Extract theta from `Y_final[:, 2]` where needed.
- All tests calling `get_final_drift` — update return unpacking (add `max_accel`).
- Tests that reference `_z_eff_batch` → `_z_eff`.
- Verify no test calls `dd.rhs` or `dd.default_uv` directly.

**Add:**
- `test_max_accel_decreases_with_longer_t_span` — run with short and long t_span,
  verify max_accel is smaller for longer integration.
- `test_max_accel_zero_for_zero_currents` — no forcing, start at equilibrium,
  max_accel should be ~0.
- Test coverage check: `pixi run pytest --cov=drogued_drifters tests/` to verify
  no dead code after refactor.

### 10. Notebooks

Update any notebook that unpacks `get_final_drift_batch` return value.
Glob for `get_final_drift_batch` in `examples/**/*.ipynb`.

### 11. Plan updates

Tick off DW-F in `plans/code-review-remarks.md`.

## Order of operations

1. Renames: `rhs` → `_rhs`, `default_uv` → `_default_uv`.
2. Rename `_z_eff_batch` → `_z_eff`. Fix `_rhs` to use it. Drop `_uv_to_theta` import.
3. Rename `_get_full_solution` → `_solve`.
4. Drop early stopping from `get_final_drift_batch`. Add `max_accel` to return.
5. Rewrite `get_final_drift` to call `_solve` directly, return `max_accel`.
6. Update `make_dd_velocity_interpolator` return unpacking.
7. Update all tests.
8. Update notebooks.
9. Run full suite + coverage.
10. Commit.
