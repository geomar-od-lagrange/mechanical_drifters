# API Review: `drogued_drifters` package

## Executive Summary

The package is well-structured for research code at its scale. The core physics is clean and the test suite is solid. However, there are several meaningful issues: redundant API surface, a leaky coordinate system abstraction, inconsistency between the two integration paths, and Parcels integration code that is copy-pasted across notebooks rather than encapsulated in the library.

---

## 1. Redundant / Overlapping API Surface

### 1a. `get_final_drift` is a no-op wrapper around `get_full_solution`

`drifter.py:836–875` — `get_final_drift` accepts identical arguments to `get_full_solution` and delegates entirely to it. The docstring claims it "returns the buoy drift velocity at the end" and callers use `ds.xd.isel(time=-1)`, but the method returns the entire trajectory Dataset — identical to `get_full_solution`. Tests (`test_drogued_drifter.py:52–55`) call `get_final_drift` and then index `ds.xd.isel(time=-1)`, which is the same pattern used after `get_full_solution` in notebook `05_wave_orbital_effects`.

**Proposed fix:** Remove `get_final_drift` and update tests to use `get_full_solution`, or give it a genuinely different return value (e.g., return `(xd_final, yd_final)` as floats).

---

### 1b. Dead legacy kwargs in `lagrange_model.py`

`lagrange_model.py:152–258` — Both `M_func` and `F_func` accept `t=None, x=None, y=None` as "unused legacy kwargs". These appear in no call site and do nothing. The module is already not exported from `__init__.py`.

**Proposed fix:** Remove `t`, `x`, `y` from `M_func` and `F_func` signatures.

---

### 1c. Stokes drift computation copy-pasted across five notebooks

The Stokes drift profile loop (sum over wave partitions, deep-water dispersion `k = sigma²/g`, decay `exp(-2kz)`) is duplicated verbatim in:
- `09a_simulation.ipynb` (cell `by00acpc9kp`)
- `08_drifter_in_effective_currents.ipynb`
- `11a_reseeded_simulation.ipynb`
- `10_along_track_validation.ipynb`
- `12_parameter_sensitivity.ipynb`

A known limitation (deep-water dispersion overestimates `k` in shallow Baltic, noted in `04_stokes_analysis`) would need to be fixed in five places.

**Proposed fix:** Add `compute_stokes_profile(ds_wav, depth_levels)` to the package, exported from `__init__.py`.

---

### 1d. `_DEG2M` constant duplicated in library and notebooks

`drifter.py:84` defines `_DEG2M = 1852.0 * 60.0`. The same constant appears in notebooks `08` and `11a`. Minor, but indicates the notebooks are reimplementing conversion logic that belongs in the library.

---

## 2. Unclear Abstractions

### 2a. Two incompatible `y0` conventions for `get_final_drift_batch`

`drifter.py:624–636` — The method accepts `y0` in "public" format `(x, y, theta, phi, xd, yd, thetad, phid)`, converts internally to stereographic `(u, v)`, then returns `Y_final` back in public format. The `_state_vars` class attribute (`drifter.py:454`) lists `["x", "y", "u", "v", "xd", "yd", "ud", "vd"]` — the *internal* naming — and is never used anywhere, misleading readers about the `Y_final` column layout.

**Proposed fix:** Remove unused `_state_vars`. Add a docstring note to `get_final_drift_batch` explicitly documenting the `Y_final` column layout.

---

### 2b. `get_full_solution` reimplements `_spherical_to_uv` / `_uv_to_spherical` inline

`drifter.py:747–820` — The initial condition conversion from `(theta, phi, thetad, phid)` to `(u, v, ud, vd)` is done by manually building and inverting a 2×2 Jacobian, rather than calling the existing `_spherical_to_uv` / `_uv_to_spherical` helpers. Two slightly different code paths do the same coordinate transform, with subtle differences in edge-case handling.

**Proposed fix:** Replace both inline Jacobian computations in `get_full_solution` with calls to `_spherical_to_uv` and `_uv_to_spherical`.

---

### 2c. `make_dd_velocity_interpolator` imports unused Parcels private symbol

`drifter.py:78–80` imports `XLinear` and `_get_corner_data_Agrid` from `parcels.interpolators._xinterpolators`. `XLinear` is imported but never used. The `_get_corner_data_Agrid` call couples the library to a private Parcels API that may change without warning.

Additionally, notebooks `08` and `11a` do not use `make_dd_velocity_interpolator` at all — they implement their own inline `DroguedDrifterKernel` using `fieldset.UV[...]` directly. Two integration workflows exist and neither is clearly preferred.

**Proposed fix:** Remove the dead `XLinear` import. Add a docstring warning about the private API dependency. Document both integration patterns with tradeoff notes.

---

### 2d. `default_uv` uses opposing unit vectors as "defaults"

`drifter.py:375–389` — Returns `U_b=1.0, V_b=1.0` (surface) and `U_d=-1.0, V_d=-1.0` (drogue). Opposing unit vectors are a stress-test, not a physically meaningful default. Users instantiating `DroguedDrifter()` without `get_uv` get a nonsensical scenario with no indication.

**Proposed fix:** Rename to `_test_uv_opposing` (private), or replace defaults with zero-current `0.0` and document that a non-trivial `get_uv` is required for meaningful results.

---

### 2e. `warm_state` stale-cache risk on particle deletion

`drifter.py:45–166` — The `warm_state` dict is mutated each call. Cache invalidation (`if warm_state.get("n") == N`) silently uses stale state if particle count changes mid-simulation (e.g., after OOB deletes). Not documented as a known limitation.

---

## 3. API vs. Notebook Inconsistencies

### 3a. `fieldset.add_constant("drogue_depth", ...)` is dead in all notebooks

Notebooks `01`, `08`, `09a`, and `11a` call `fieldset.add_constant("drogue_depth", DROGUE_DEPTH)`, implying the library reads this. But `make_dd_velocity_interpolator` and inline kernels both ignore it — they use `DroguedDrifter.l` to derive drogue position dynamically during ODE integration. These `add_constant` calls are dead code.

---

### 3b. `__init__.py` exposes only `DroguedDrifter`; user-facing helpers are not exported

`__init__.py` exports `DroguedDrifter`, `compute_F`, `compute_M`. But `make_profile_sampler`, `make_dd_velocity_interpolator`, `buoy_added_mass`, `buoy_drag_coeff`, `drogue_added_mass`, `drogue_drag_coeff` must be imported from `drogued_drifters.drifter` directly. There is no single public namespace.

**Proposed fix:** Add all user-facing symbols to `__init__.py`.

---

### 3c. `AdvectionEE` vs `AdvectionRK4` asymmetry in `09a`

`09a_simulation.ipynb` uses `AdvectionEE` for the drogued drifter run but `AdvectionRK4` for point particle baselines. The asymmetry is not justified in the narrative. Readers comparing trajectories may attribute differences to the drifter model rather than the lower-order integrator.

---

## 4. Other Code Quality Issues

### 4a. `_rhs_batch` mass matrix assembly is verbose (16 explicit assignments)

`drifter.py:507–523` — A 16-line construction fills each matrix element individually. Correct, but harder to audit than a symmetric-matrix fill helper would be.

---

### 4b. Convergence event in `get_final_drift_batch` doubles RHS evaluations

`drifter.py:654–659` — The `converged` event function calls `self._rhs_batch(Y, sample_uv)` redundantly at each ODE step. For `RK45` (the default solver), this roughly doubles the number of RHS evaluations. For large N or expensive `sample_uv`, this is a significant hidden cost.

**Proposed fix:** Cache the last RHS evaluation inside `rhs_flat` and reuse it in the event function.

---

### 4c. Dead assignment in `_uv_to_spherical`

`drifter.py:276–279` — `thetad` is assigned on line 276 and immediately overwritten on line 279. The first assignment is dead code, likely an editing artifact.

---

### 4d. `parcels` pinned to `git+main`

`pyproject.toml` — `parcels @ git+https://github.com/OceanParcels/parcels.git@main`. Pinning to an untagged branch means the library can silently break on any upstream commit, especially given the private API imports in `drifter.py`. The `.pixi.lock` mitigates daily churn but not open sharing scenarios.

---

## 5. Additional Issues (from deeper review)

### 5a. README state vector is stale

`README.md` describes the state vector as `[x, y, theta, phi, xd, yd, thetad, phid]`. The internal integration now uses `[x, y, u, v, xd, yd, ud, vd]` (stereographic coordinates). The public API still returns (theta, phi) in the xarray Dataset, so user-facing examples remain valid, but the README's claim about the state vector is wrong.

---

### 5b. Static two-depth sampler dispatch is fragile

`drifter.py:608–614` — The backward-compat `sample_uv` wrapper (used when `U_b, V_b, U_d, V_d` are passed directly) dispatches buoy vs. drogue velocity using `np.all(z_arr == 0)`. Any z value that is non-zero — including `z=1e-30` from floating-point noise — routes to drogue velocities. This works in practice (the ODE solver only calls `sample_uv(zeros)` and `sample_uv(z_eff > 0)`) but the logic is fragile.

---

### 5d. `make_dd_velocity_interpolator` had wrong unit conversion (fixed)

`drifter.py` — `_get_corner_data_Agrid` returns raw field values in m/s (same as stored in the netCDF). The interpolator was treating these as deg/s and multiplying by `deg2m_lon ≈ 64,000`, feeding the DD model velocities ~64,000× too large → overflow → OOB particle deletion. Fixed: removed the erroneous input multiplication; the output conversion (m/s → deg/s) is still correct and necessary.

This was masked in notebook 01 (idealized flow) because the test fieldset used Cartesian `mesh="flat"`, not `mesh="spherical"`, so the spherical conversion branch was never hit.

---

### 5c. Convergence tolerance is not documented in physical terms

`get_final_drift_batch` stops when `max(|xdd|, |ydd|) < conv_tol=1e-4` m/s². With `t_span=(0, 120)` s, this corresponds to a velocity convergence of roughly 0.012 m/s — adequate but not tight. The `atol=rtol=1e-3` solver tolerances are also loose. Worth documenting explicitly given the expected drift velocity magnitude (0.1–1.0 m/s).
