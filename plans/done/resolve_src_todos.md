# Resolve src/ TODO comments

Ten TODO comments across four files, plus z-convention test stragglers from the previous migration.

## 1. Mechanical fixes (safe to do now)

### 1a. Drop `compute_M` / `compute_F` aliases — `__init__.py:4-7`

> `# TODO: Drop these. We break stuff.`

No notebook or external code imports `compute_M` or `compute_F`. Delete lines 4-7 of `__init__.py` and update the docstring reference in `drifter.py:351`.

### 1b. Change `theta` default to `np.pi` — `drifter.py:629`, `drifter.py:696`, `drifter.py:491`

> `# TODO: As we regularized now, this is not necessary anymore`

The stereographic parameterization is smooth at theta=pi (origin u=v=0). Three locations:

- `get_full_solution` default `theta=0.999 * np.pi` -> `theta=np.pi`
- `get_final_drift` default `theta=0.999 * np.pi` -> `theta=np.pi`
- `get_final_drift_batch` default `theta0=0.999 * np.pi` -> `theta0=np.pi`

Remove the TODO comments.

### 1c. Remove `LagrangeParams` TODO — `lagrange_model.py:11`

> `# TODO: Are the LagrangeParams really used throughout?`

**Answer: yes.** It defines the canonical parameter order for lambdification (line 170) and is used to pack args in `M_func`/`F_func` (lines 352, 423). `drifter.py` doesn't use it — it passes a plain dict via `**p` to `M_func`/`F_func`, which is fine and doesn't need changing. Remove the TODO.

### 1d. Replace depth-order TODO — `drifter.py:13`

> `# TODO: Sure about the ascending order? Is this what parcels gives for the grids?`

**Answer: yes.** `make_profile_sampler` needs ascending depth for `np.searchsorted`. The caller `make_dd_velocity_interpolator` (line 159) explicitly negates and reverses Parcels' z-down array, so the result is always ascending z-up (most negative first, 0 last, e.g. `[-20, -10, 0]`).

Replace the TODO with: `# depth_levels must be sorted ascending (z-up, e.g. [-20, -10, 0]); the Parcels bridge handles the conversion.`

### 1e. Replace Parcels SHA TODO — `drifter.py:54`

> `# TODO: Give parcels v4 Git SHA. We're working w/ an alpha version.`

The SHA is already pinned in `pyproject.toml` (`parcels @ git+...@17241585...`). Replace the TODO with: `# Parcels v4 alpha — pinned to specific commit in pyproject.toml.`

### 1f. Keep warm-state TODO — `drifter.py:174-177`

> `# TODO: warm_state cache validation only checks particle count.`

Keep as-is. Known limitation; needs deeper thought about particle deletion/domain exit scenarios.

### 1g. Default values documentation — `drifter.py:304`

> `# TODO: Shuold the default values come from the helper functions above?`

Keep the numeric defaults. Replace the TODO with a comment block documenting how each default was computed from the helper functions (rho, geometry, coefficients). Example:

```python
# Default values for Callies et al. drifter geometry, computed from:
#   drogue_horizontal_added_mass(rho=1025, w_d=..., h_d=...)  -> m_tilde_d=101.0
#   buoy_horizontal_added_mass(rho=1025, d_b=..., h_b=...)    -> m_tilde_b=1.9
#   drogue_horizontal_drag_coeff(rho=1025, w_d=..., h_d=...)  -> k_d=154.0
#   buoy_horizontal_drag_coeff(rho=1025, d_b=..., h_b=...)    -> k_b=12.0
```

## 2. Rename and document horizontal drag/added-mass helpers — `drifter.py:197`

> `# TODO: The param functions should specify that for the drogue and the buoy, we're talking about horizontal drag and horizontal inertia`

Two changes:

1. **Rename** the four helper functions to include `_horizontal`:
   - `drogue_added_mass` -> `drogue_horizontal_added_mass`
   - `buoy_added_mass` -> `buoy_horizontal_added_mass`
   - `drogue_drag_coeff` -> `drogue_horizontal_drag_coeff`
   - `buoy_drag_coeff` -> `buoy_horizontal_drag_coeff`

2. **Update docstrings** to state plainly that these are horizontal components.

Update all call sites (tests, notebooks, the default-values comment from 1g). Remove the TODO.

## 3. Wire `compute_stokes_profile` into notebooks — `stokes.py:4`

> `# TODO: Let's discuss again to make sure this is really used anywhere. I'd like to use it in the existing example notebooks where it makes sense.`

`compute_stokes_profile` is well-tested (15+ call sites in tests) but not used in any notebook. Two notebooks compute Stokes profiles inline and should use the centralized function instead:

### 3a. `examples/baltic_drifters/02_derive_effective_currents.ipynb` — primary target

Lines ~2041-2093 contain a full hand-rolled Stokes profile loop over three wave partitions (wind waves, SW1, SW2). This reimplements exactly what `compute_stokes_profile` does. Refactor to:

1. Derive surface Stokes (u, v) from each partition's Hs, period, and direction.
2. Call `compute_stokes_profile(surface_u, surface_v, peak_period, depth_levels)` per partition.
3. Sum the results — matching the multi-partition pattern in the `compute_stokes_profile` docstring.

Note: the notebook currently uses z-down depths. After refactoring, depth_levels passed to `compute_stokes_profile` must be z-up (negative, ascending).

### 3b. `examples/idealized_flow/03_drogued_drifter_in_wave_orbitals.ipynb` — secondary target

The `stokes_uv` callback (~line 1025) and `u_St` helper (~line 1129) evaluate Stokes profiles inline. These can call `compute_stokes_profile` for consistency. The earlier symbolic derivations in the notebook are pedagogical and should stay as-is.

Remove the TODO from `stokes.py:4`.

## 4. Z-convention test stragglers

Leftovers from the `z_convention_upward` migration:

### 4a. Dead `depth_levels` in `test_integration_full_chain.py`

Lines ~130 and ~226 assign `depth_levels = np.array([0.0, 10.0])` but never use it — the tests use `get_uv` callbacks instead of `sample_uv`. Delete the dead assignments.

### 4b. Degenerate interval test in `test_drifter_parcels.py`

Line ~146 uses `depth_levels = np.array([0.0, 0.0, 5.0])`. This is passed to `make_profile_sampler` and should be z-up: `np.array([-5.0, 0.0, 0.0])` (degenerate interval at surface).

## Execution order

1. **Batch 1 (mechanical):** 1a-1e, 1g, 4a, 4b — drop aliases, fix theta, remove/replace TODOs, fix test stragglers.
2. **Batch 2 (rename + doc):** 2 — rename four helper functions, update docstrings and all call sites.
3. **Batch 3 (notebook wiring):** 3a, 3b — refactor inline Stokes into `compute_stokes_profile` calls.
4. **Run tests** after each batch.
5. **Re-run affected notebooks with papermill** after batch 3.
