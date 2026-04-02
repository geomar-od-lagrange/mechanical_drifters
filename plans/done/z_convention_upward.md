# Migration plan: z positive UPWARD everywhere

## Motivation

The internal Lagrangian mechanics (`lagrange_model.py`) uses z positive
upward implicitly: at equilibrium the drogue is at `r_d[2] = l * cos(pi) = -l`,
and the potential energy `V = (m_d - m_hat_d) * g * r_d[2]` is minimized when
the drogue hangs below the buoy (negative z). But the external API in
`drifter.py` and `stokes.py` uses z positive **downward**. This creates a
confusing mismatch and forces every caller to reason about sign flips.

The goal is to make z positive upward everywhere: in `src/`, in tests, in
notebooks, and in the public API. Drogue depth will be a negative number
(e.g., -3 m for a 3 m pole hanging straight down).

---

## Inventory of z-down usage

### `src/drogued_drifters/drifter.py`

| Line | Code / docstring | What it does | Change needed |
|------|-----------------|--------------|---------------|
| 21 | `depth_levels: 1-D array of depth values` | `make_profile_sampler` arg | Update docstring: "sorted ascending (negative = below surface)" |
| 38 | `idx = np.searchsorted(depth_levels, z_arr).clip(1, D - 1)` | Interpolation assumes depth_levels sorted ascending | With z-up, depth_levels will be sorted ascending (e.g., `[-20, -10, -5, 0]`), so `searchsorted` still works correctly. No code change needed, only docstring. |
| 324 | `z: Depth [m], positive downward` | `default_uv` docstring | Change to: `z: Vertical position [m], positive upward (0 = surface, negative = below)` |
| 329 | `if z == 0.0: return 1.0, 1.0` | Surface detection | No change (z=0 still means surface) |
| 379 | `z_d = float(max(0.0, -self.l * np.cos(theta)))` | Scalar RHS: computes drogue depth as positive-down value | Change to: `z_d = float(min(0.0, self.l * np.cos(theta)))`. At equilibrium (theta=pi), this gives `l * (-1) = -l < 0` (below surface). The `min(0.0, ...)` clamps to non-positive (drogue can't be above water). |
| 382 | `U_d, V_d = self.get_uv(t=t, x=x_b, y=y_b, z=z_d)` | Passes z_d to callback | No change (value is now negative instead of positive) |
| 397 | `z_eff: Drogue depth [m], positive downward` | `_z_eff_batch` docstring | Change to: `z_eff: Drogue vertical position [m], positive upward (non-positive)` |
| 401 | `return np.maximum(0.0, -self.l * cos_theta)` | Batch: computes drogue depth positive-down | Change to: `return np.minimum(0.0, self.l * cos_theta)`. At equilibrium cos_theta = -1 so result = -l. Clamped to <= 0 (can't be above surface). |
| 427 | `U_b, V_b = sample_uv(np.zeros(N))` | Buoy at z=0 | No change (z=0 still means surface) |
| 428-429 | `z_eff = self._z_eff_batch(u, v)` / `U_d, V_d = sample_uv(z_eff)` | Passes z_eff to sampler | No change (z_eff will now be negative) |

### `src/drogued_drifters/stokes.py`

| Line | Code / docstring | What it does | Change needed |
|------|-----------------|--------------|---------------|
| 28 | `depth_levels: Depth levels [m], positive downward` | Arg docstring | Change to: `depth_levels: Vertical positions [m], positive upward (0 = surface, negative = below)` |
| 45 | `z = depth_levels.reshape(-1, *([1] * ndim))` | Broadcast depth for decay | No code change; z values will be negative |
| 48 | `decay = np.exp(-2 * k_b * z)` | Exponential decay | **Key change**: With z-up (z <= 0), this becomes `exp(-2k * z)` where z < 0, giving `exp(+2k|z|)` which **grows** -- wrong! Must change to: `decay = np.exp(2 * k_b * z)`. With z = -|d|, this gives `exp(-2k|d|)`, the correct decay. |

### `src/drogued_drifters/lagrange_model.py`

No changes needed. The Lagrangian already uses z-up internally via the
stereographic parameterization. The z-axis in the 3-vector
`r = l * [sin_theta*cos_phi, sin_theta*sin_phi, cos_theta]` points upward:
at theta=pi, r[2] = -l (drogue below buoy).

### `src/drogued_drifters/__init__.py`

No changes needed. Just re-exports.

---

## Test changes

### `tests/test_drogued_drifter.py`

| Line(s) | What | Change |
|---------|------|--------|
| 46 | `U_d, V_d = dd.get_uv(t=t, x=0.0, y=0.0, z=3.0)` | Change `z=3.0` to `z=-3.0` (drogue 3 m below surface) |
| 115 | `factor = np.exp(-abs(z) / 2.0)` | Change to `np.exp(z / 2.0)` -- with z-up, z<=0 so `exp(z/2)` decays downward. Or keep `abs(z)` which is convention-agnostic. **Keep `abs(z)` for clarity.** No change needed. |
| 162 | `if z == 0.0: return U_b, V_b` | No change (z=0 still means surface) |
| 329 | `if z == 0.0:` in `default_uv` test | z=0 still means surface. No change. |

### `tests/test_stokes.py`

| Line(s) | What | Change |
|---------|------|--------|
| 28 | `depth_levels = np.array([0.0, 5.0, 10.0, 15.0])` | Change to `np.array([0.0, -5.0, -10.0, -15.0])` |
| 45 | `expected_u5 = surface_u * np.exp(-2 * k_expected * 5.0)` | Change to `expected_u5 = surface_u * np.exp(2 * k_expected * (-5.0))` (same numerical value) |
| 61 | `depth_levels = np.array([0.0, 2.0, 5.0, 10.0, 20.0])` | Change to negatives |
| 78 | `depth_levels = np.array([0.0, 3.0, 6.0])` | Change to negatives |
| 108 | `depth_levels = np.array([0.0, 5.0, 10.0])` | Change to negatives |
| 163 | `depth_levels = np.array([0.0, 5.0, 10.0])` | Change to negatives |
| 183-184 | `depth_levels`, `peak_period` arrays | Change depth_levels to negatives |
| 203 | `depth_levels = np.array([0.0, 5.0, 10.0])` | Change to negatives |
| 229 | `depth_levels_list = [0.0, 5.0, 10.0]` | Change to negatives |
| 249 | `depth_levels = np.array([0.0, -5.0, 10.0])` (negative test) | This test documents "invalid" depths. With z-up, positive depths are the invalid ones. Change to `np.array([0.0, 5.0, -10.0])` and update the commentary. |
| 270 | `depth_levels = np.array([0.0, 100.0, 500.0])` | Change to negatives |

**Important**: The test assertions themselves (e.g., comparing to `exp(-2k*z)`)
need updating because the formula changes. With z-up, the analytical reference
is `exp(2kz)` where z <= 0. The numerical values stay identical.

All `depth_levels` arrays must be sorted ascending. With z-up convention,
`[-15, -10, -5, 0]` is ascending. The current `[0, 5, 10, 15]` is also
ascending. So for `searchsorted` to work, the new arrays must go from
most-negative (deepest) to zero (surface). This is a reversal of the current
order. Alternatively, keep the convention "sorted from surface downward" and
reverse: `[0, -5, -10, -15]`. But `searchsorted` requires ascending order,
so depth_levels must be `[-15, -10, -5, 0]`.

**Decision**: depth_levels should be sorted ascending (deepest first, surface
last): `[-15, -10, -5, 0]`. This is the natural ascending order for negative
numbers. Document this requirement.

### `tests/test_lagrange_physics.py`

| Line(s) | What | Change |
|---------|------|--------|
| 168, 172, 186 | `z_eff` assertions: `assert np.all(z_eff >= 0.0)`, `assert np.all(z_eff <= 3.0)` | Change to `assert np.all(z_eff <= 0.0)` and `assert np.all(z_eff >= -3.0)` |
| 186 | `np.testing.assert_allclose(z_eff, 3.0, ...)` | Change to `assert_allclose(z_eff, -3.0, ...)` |
| 210 | `assert z_large < z_small` (tilt decreases depth) | With z-up, tilt makes depth less negative (shallower), so: `assert z_large > z_small` (both negative, larger = shallower) |

### `tests/test_integration_full_chain.py`

| Line(s) | What | Change |
|---------|------|--------|
| 28 | `depth_levels = np.array([0.0, 5.0, 10.0, 15.0, 20.0])` | Change to `np.array([-20.0, -15.0, -10.0, -5.0, 0.0])` |
| 64, 107, 157, 198, 277, 310 | Similar depth_levels arrays | Change all to z-up (negative, ascending) |
| 133 | `sample_uv_weak(z): return ... if z == 0` | z=0 still means surface. No change. |
| 229 | `sample_uv_east(z): return ... if z == 0` | No change. |
| 282 | `sample_uv_scalar(z): if z == 0:` | No change. |

### `tests/test_numerical_edge_cases.py`

| Line(s) | What | Change |
|---------|------|--------|
| 71, 88, 102, 210, 219, 277, 293 | `get_uv` callbacks with `if z == 0` | z=0 still means surface. No change. |

### `tests/test_drifter_parcels.py`

| Line(s) | What | Change |
|---------|------|--------|
| 21-26 | `depth_levels = np.array([0.0, 5.0, 10.0])` | Change to negatives (ascending: `[-10.0, -5.0, 0.0]`) |
| 49 | `depth_levels = np.array([0.0, 10.0])` | Change to `[-10.0, 0.0]` |
| 63 | `depth_levels = np.array([0.0, 5.0, 10.0])` | Change to `[-10.0, -5.0, 0.0]` |
| 83 | `depth_levels = np.array([2.0, 5.0, 10.0])` | Change to `[-10.0, -5.0, -2.0]`. Note: the shallow boundary test semantics change (shallowest is now -2, not 2). Test z=0 is now above the shallowest level. |
| 100 | `depth_levels = np.array([0.0, 5.0, 10.0])` with z=100 deep test | Change to `[-10.0, -5.0, 0.0]` with z=-100 test |
| 134, 209, 227, 237 | Various depth_levels | Change to negatives, ascending |

### `tests/test_lagrange_symbolic_fallback.py`

No changes needed. Tests only exercise symbolic M/F which don't involve z.

---

## Notebook changes

### `examples/wave_orbitals/00_wave_stokes_reference_levels.ipynb`

| Cell/line ref | What | Change |
|------|------|--------|
| Line 1026 | `"Stokes drift profile evaluated at depth z (positive downward per src convention)."` | Remove "positive downward per src convention". Change to "positive upward". |
| Line 1027 | `z_up = -z` (manual conversion from z-down to z-up) | Remove the conversion. Use `z` directly since it will now be z-up. |
| Line 1028 | `return float(P_A**2 * P_omega * P_k * np.exp(2 * P_k * z_up)), 0.0` | Change to `return float(P_A**2 * P_omega * P_k * np.exp(2 * P_k * z)), 0.0` (z is already z-up now). |

### `examples/baltic_drifters/05_wave_orbital_effects.ipynb`

This notebook defines `wave_uv` and `stokes_uv` callbacks and `wave_3c_uv`
and `stokes_3c_uv` callbacks. They all use `exp(-k * z)` for orbital velocity
decay and `exp(-2 * k * z)` for Stokes drift decay, which assumes z
positive downward.

| Cell ref | Code | Change |
|----------|------|--------|
| cell `wy6mq8puxu` | `U = U_mean + A * sigma * np.exp(-k * z) * np.cos(phase)` | Change to `np.exp(k * z)` (z-up, z<=0 gives decay) |
| cell `wy6mq8puxu` | `U = U_mean + u_St_0 * np.exp(-2 * k * z)` | Change to `np.exp(2 * k * z)` |
| cell `lgepe8li95a` | `amp = c["A"] * c["sigma"] * np.exp(-c["k"] * z) * np.cos(phase)` (in `wave_3c_uv`) | Change to `np.exp(c["k"] * z)` |
| cell `lgepe8li95a` | `c["A"]**2 * c["sigma"] * c["k"] * np.exp(-2 * c["k"] * z) * c["dir_x"]` (in `stokes_3c_uv`) | Change to `np.exp(2 * c["k"] * z)` |
| cell `y0xva4yhk9` | `amp = c["A"] * c["sigma"] * np.exp(-c["k"] * z) * np.cos(ph)` (in `_uv` multi-seed) | Change to `np.exp(c["k"] * z)` |

### Other baltic_drifters notebooks

Notebooks 00-04, 06-12 do not import or use `drogued_drifters` or reference
depth conventions. No changes needed.

---

## Parcels integration (`make_dd_velocity_interpolator`)

### How Parcels handles depth

OceanParcels stores depth as **positive downward** in `field.grid.depth`.
The `make_dd_velocity_interpolator` function (line 106) reads
`depth_levels = np.asarray(field_U.grid.depth, dtype=float)` and passes
them directly to `make_profile_sampler`.

With z-up convention, we need to **negate** the Parcels depth levels before
passing them to `make_profile_sampler`:

```python
# Line 106: convert Parcels z-down to our z-up convention
depth_levels = -np.asarray(field_U.grid.depth, dtype=float)
```

However, this reverses the sort order (Parcels depths are ascending positive,
negating gives descending negative). We need to also reverse the profiles:

```python
depth_levels_raw = np.asarray(field_U.grid.depth, dtype=float)
depth_levels = -depth_levels_raw[::-1]  # negate and reverse -> ascending z-up
# ... later, reverse U_profiles and V_profiles along depth axis:
U_profiles = U_profiles[::-1]
V_profiles = V_profiles[::-1]
```

Alternatively, negate after extracting and sort:

```python
depth_levels = -np.asarray(field_U.grid.depth, dtype=float)
order = np.argsort(depth_levels)
depth_levels = depth_levels[order]
# ... profiles indexed by iz need reordering too
```

**Recommended approach**: Negate and reverse in one step after profile
extraction. This keeps the Parcels loop index (`iz`) aligned with the
original Parcels depth ordering, and we reverse once at the end:

```python
# After the extraction loop:
depth_levels = -depth_levels[::-1]
U_profiles = U_profiles[::-1]
V_profiles = V_profiles[::-1]
```

This is the cleanest approach.

---

## Public API breaking changes

The following are breaking changes for external users:

1. **`get_uv` callback signature**: The `z` argument changes from positive-down
   to positive-up. Any user-supplied `get_uv` callback must be updated.
   - Surface: z=0 (unchanged)
   - Drogue at 3 m depth: z was 3.0, now z=-3.0

2. **`compute_stokes_profile` depth_levels argument**: Values change from
   positive to non-positive, and sort order reverses (from `[0, 5, 10]` to
   `[-10, -5, 0]`).

3. **`make_profile_sampler` depth_levels argument**: Same as above.

4. **`_z_eff_batch` return values**: Change from `[0, l]` to `[-l, 0]`.
   This is a private method but may be used in tests or notebooks.

5. **`default_uv` z argument**: Documented convention changes.

---

## Migration strategy

### Can this be done incrementally?

**No.** The z convention is threaded through the entire call chain:

```
get_uv(z) <- rhs() computes z_d <- _z_eff_batch() <- stereographic coords
          <- make_profile_sampler(depth_levels) <- compute_stokes_profile(depth_levels)
          <- make_dd_velocity_interpolator (Parcels bridge)
```

If we change `_z_eff_batch` to return negative values but don't change
`stokes.py` or the profile sampler, the sampler will look up the wrong
depth. If we change `stokes.py` but not the callers, the depth_levels
arrays will be inconsistent.

**This must be an atomic change.** All source, all tests, and all notebooks
must be updated in a single commit (or a tightly coupled PR).

### Execution plan

1. **Update `src/drogued_drifters/stokes.py`**: Change decay formula and
   docstring.

2. **Update `src/drogued_drifters/drifter.py`**:
   - Change `_z_eff_batch` to return non-positive values.
   - Change `rhs` scalar path `z_d` computation.
   - Update `default_uv` docstring.
   - Update `make_profile_sampler` docstring.
   - Add Parcels depth negation in `make_dd_velocity_interpolator`.

3. **Update all tests**: Negate depth_levels arrays, reverse sort order,
   update z_eff assertions, update analytical reference formulas.

4. **Update notebooks**: Change `exp(-k*z)` to `exp(k*z)` patterns,
   remove manual z-up/z-down conversions, update comments.

5. **Run full test suite**: `pixi run pytest` to verify nothing breaks.

6. **Re-run notebooks with papermill** to verify plots and printed values
   are unchanged.

---

## Tricky spots

1. **`stokes.py` sign flip**: The formula `exp(-2*k*z)` becomes `exp(2*k*z)`.
   This looks like it could amplify rather than decay, but with z <= 0 it
   correctly gives `exp(-2*k*|z|)`. Add a code comment explaining this.

2. **`make_profile_sampler` sort order**: `searchsorted` requires ascending
   depth_levels. With z-up, depth_levels must go `[-20, -10, -5, 0]` (most
   negative first). This is the opposite of the current `[0, 5, 10, 20]`
   ordering. All callers constructing depth_levels arrays must reverse. Add
   a runtime check: `assert np.all(np.diff(depth_levels) > 0)` or at least
   document the requirement.

3. **Parcels depth convention**: OceanParcels uses z positive downward in
   `grid.depth`. The `make_dd_velocity_interpolator` must negate and reverse
   these values. This is isolated to one function but must be tested carefully.
   Consider adding a standalone test that verifies the conversion.

4. **`default_uv` dispatch on `z == 0`**: This still works because z=0
   means the surface in both conventions. But callbacks that check
   `z > 0` vs `z < 0` will need updating. In particular, `_step_sampler`
   in tests uses `if np.all(z_arr == 0)` which is fine. And callbacks
   like `_getuv_sheared` use `abs(z)` or `z == 0`, which are also fine.

5. **Notebook `00_wave_stokes_reference_levels.ipynb`**: This notebook does
   its own physics (linear wave theory) with z-up internally and then
   explicitly converts to z-down when calling the library (line 1027:
   `z_up = -z`). After migration, this conversion layer disappears, which
   is a simplification.

6. **`depth_levels` in profile sampler edge cases**: The boundary clipping
   tests (`test_make_profile_sampler_boundary_shallow` and
   `test_make_profile_sampler_boundary_deep`) need careful updates. With
   z-up, "shallow" means near 0 and "deep" means large negative. The
   test at line 83 uses `depth_levels = [2.0, 5.0, 10.0]` for "shallowest
   is z=2" -- this becomes `[-10.0, -5.0, -2.0]` where shallowest is -2.

7. **No `W` (vertical velocity) component**: The drifter model only queries
   horizontal `(U, V)` at a given z. The z value is used only for profile
   lookup, never as a dynamic variable in the ODE state. This means the
   sign change is purely about which value gets passed to `get_uv` / `sample_uv`.

---

## Validation checklist

- [ ] `pixi run pytest` passes (all tests green)
- [ ] `pixi run pytest -m slow` passes (symbolic fallback tests)
- [ ] `cd examples/wave_orbitals && pixi run papermill 00_wave_stokes_reference_levels.ipynb 00_wave_stokes_reference_levels.ipynb --execution-timeout 600`
- [ ] `cd examples/baltic_drifters && pixi run papermill 05_wave_orbital_effects.ipynb 05_wave_orbital_effects.ipynb`
- [ ] Numerical results in notebooks are unchanged (same drift velocities, same plots)
- [ ] No remaining occurrences of "positive downward" in src/ or examples/
- [ ] `grep -r "positive down" src/ examples/` returns nothing
