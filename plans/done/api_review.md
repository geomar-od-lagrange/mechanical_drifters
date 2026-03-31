# API Review: `drogued_drifters` package

## Executive Summary

The package is well-structured for research code at its scale. The core physics is clean and the test suite is solid. However, there are several meaningful issues: redundant API surface, a leaky coordinate system abstraction, inconsistency between the two integration paths, and Parcels integration code that is copy-pasted across notebooks rather than encapsulated in the library.

---

## 1. Redundant / Overlapping API Surface

### 1a. `get_final_drift` refactored to return only final drift values

`drifter.py:836–875` — Previously, `get_final_drift` accepted identical arguments to `get_full_solution` and delegated entirely to it, returning the full trajectory Dataset. This required callers to index the result with `ds.xd.isel(time=-1)` to extract the final drift.

**Decision:** Changed return value of `get_final_drift` to return only `(xd_final, yd_final)` as floats, making the method's purpose explicit and eliminating the need for post-processing indexing.

---

### 1b. Dead legacy kwargs removed from `lagrange_model.py`

`lagrange_model.py:152–258` — Both `M_func` and `F_func` previously accepted unused legacy kwargs `t=None, x=None, y=None` that appeared in no call site and served no purpose.

**Decision:** Removed `t`, `x`, `y` from `M_func` and `F_func` signatures. Backward compatibility is not a concern for this internal-use module.

---

### 1c. Stokes drift computation extracted to library helper

The Stokes drift profile loop (sum over wave partitions, deep-water dispersion `k = sigma²/g`, decay `exp(-2kz)`) was duplicated verbatim across five notebooks:
- `09a_simulation.ipynb` (cell `by00acpc9kp`)
- `08_drifter_in_effective_currents.ipynb`
- `11a_reseeded_simulation.ipynb`
- `10_along_track_validation.ipynb`
- `12_parameter_sensitivity.ipynb`

**Decision:** Added `compute_stokes_profile(ds_wav, depth_levels)` helper function to `src/drogued_drifters/stokes.py`. This consolidates the computation logic, making future maintenance (e.g., addressing deep-water dispersion overestimation in shallow Baltic) a single-point fix.

---

### 1d. `_DEG2M` constant duplication — deferred

`drifter.py:84` defines `_DEG2M = 1852.0 * 60.0`, which also appears in notebooks `08` and `11a`.

**Decision:** Out of scope for this review. Low impact and easily addressable once notebook refactoring is prioritized separately.

---

## 2. Unclear Abstractions

### 2a. State vector conventions clarified in `get_final_drift_batch`

`drifter.py:624–636` — The method accepts `y0` in "public" format `(x, y, theta, phi, xd, yd, thetad, phid)`, converts internally to stereographic `(u, v)`, then returns `Y_final` back in public format. Previously, an unused `_state_vars` class attribute listed internal naming, creating confusion.

**Decision:** Removed the unused `_state_vars` class attribute. Enhanced `get_final_drift_batch` docstring to explicitly document the `Y_final` column layout in public format.

---

### 2b. Coordinate transforms consolidated via helper functions

`drifter.py:747–820` — Previously, `get_full_solution` reimplemented initial condition conversion from `(theta, phi, thetad, phid)` to `(u, v, ud, vd)` by manually building and inverting a 2×2 Jacobian, rather than calling existing helpers.

**Decision:** Moved `_spherical_to_uv` and `_uv_to_spherical` helper functions to `lagrange_model.py`. Replaced inline Jacobian computations in `get_full_solution` with calls to these helpers, ensuring a single coordinate-transform code path.

---

### 2c. Private Parcels API dependency documented

`drifter.py:78–80` imports from `parcels.interpolators._xinterpolators`. The library couples to private Parcels internals (`_get_corner_data_Agrid`) that may change without warning. Additionally, `XLinear` was imported but unused.

**Decision:** Removed the dead `XLinear` import. Kept the private API call to `_get_corner_data_Agrid` (required for the interpolator to function) and added a clear docstring warning about the dependency and its fragility. Notebook integration patterns (08, 11a) deferred to separate refactoring.

---

### 2d. Unrealistic default velocities retained as test scenario

`drifter.py:375–389` — The `default_uv` method returns opposing unit vectors: `U_b=1.0, V_b=1.0` (surface) and `U_d=-1.0, V_d=-1.0` (drogue). This is a stress-test scenario, not a physically realistic default.

**Decision:** Retained this behavior as a valid (if unrealistic) test case. Users should understand that instantiating `DroguedDrifter()` without a meaningful `get_uv` callback produces nonsensical results and is not intended for production use.

---

### 2e. `warm_state` cache limitation noted for future work

`drifter.py:45–166` — The `warm_state` dict is mutated each call with cache invalidation based on particle count. If particle count changes mid-simulation (e.g., after OOB deletes), stale cached state could be reused, though this is masked in practice.

**Decision:** Added a TODO note documenting this cache fragility as a known limitation. Deferred a more robust refactoring to future work, as the current pattern is functional and particle deletion events are infrequent in typical workflows.

---

## 3. API vs. Notebook Inconsistencies

### 3a. Dead `drogue_depth` constant removed from notebooks

Notebooks `01`, `08`, `09a`, and `11a` previously called `fieldset.add_constant("drogue_depth", DROGUE_DEPTH)`, but the library never reads it — drogue position is derived dynamically from `DroguedDrifter.l` during ODE integration.

**Decision:** Removed all `add_constant("drogue_depth", ...)` calls from notebooks. The library ignores this constant, and its removal simplifies the fieldset setup without functional impact.

---

### 3b. Public API exports — deferred

`__init__.py` currently exports `DroguedDrifter`, `compute_F`, and `compute_M`, but user-facing helpers like `make_profile_sampler`, `make_dd_velocity_interpolator`, and coefficient functions must be imported directly from `drogued_drifters.drifter`.

**Decision:** Out of scope for this review. API completeness and export strategy are deferred until the notebook refactoring reveals which symbols are actively used and should be public.

---

### 3c. Integration method unified to Euler-Euler in `09a`

`09a_simulation.ipynb` previously used `AdvectionEE` for the drogued drifter but `AdvectionRK4` for point particle baselines, conflating model differences with integrator choice.

**Decision:** Standardized all integrations to `AdvectionEE` (Euler-Euler) everywhere in the notebook. This ensures trajectory differences reflect physics of the drifter model rather than numerical integrator choice.

---

## 4. Other Code Quality Issues

### 4a. RHS and mass matrix implementation — deferred for refactoring discussion

`drifter.py:507–523` — The `_rhs_batch` method constructs the mass matrix with 16 explicit assignments. While correct, the code is verbose and difficult to audit.

**Decision:** Deferred to a separate discussion about whether `_rhs_batch` should use generated code for `M` and `F` (as `M_func` and `F_func` do in `lagrange_model.py`). This refactoring touches core numerical machinery and requires careful planning. 

---

### 4b. RHS implementation strategy — deferred with 4a

`drifter.py:654–659` — The `converged` event function in `get_final_drift_batch` calls `self._rhs_batch(Y, sample_uv)` at each ODE step, approximately doubling RHS evaluations for the `RK45` default solver.

**Decision:** Deferred along with 4a. This optimization (caching and reuse of RHS evaluations) is part of the larger RHS/mass-matrix refactoring discussion.

---

### 4c. Dead assignment in `_uv_to_spherical`

`drifter.py:276–279` — `thetad` is assigned on line 276 and immediately overwritten on line 279. The first assignment is dead code, likely an editing artifact.

---

### 4d. Parcels dependency pinned to specific commit

`pyproject.toml` — Previously pinned `parcels` to `git+main`, an untagged branch. This risked silent breakage on any upstream commit, particularly given the private API imports in `drifter.py`.

**Decision:** Updated dependency to pin `parcels` to a specific git SHA. This provides deterministic builds and ensures reproducibility across environments.

---

## 5. Additional Issues (from deeper review)

### 5a. README state vector is stale

`README.md` describes the state vector as `[x, y, theta, phi, xd, yd, thetad, phid]`. The internal integration now uses `[x, y, u, v, xd, yd, ud, vd]` (stereographic coordinates). The public API still returns (theta, phi) in the xarray Dataset, so user-facing examples remain valid, but the README's claim about the state vector is wrong.

---

### 5b. Two-depth sampler removed; profile sampling enforced

`drifter.py:608–614` — Previously provided a backward-compat `sample_uv` wrapper that dispatched velocities based on depth using `np.all(z_arr == 0)`. This was fragile (floating-point noise could trigger wrong branch) and conflated two sampling models.

**Decision:** Removed the two-depth sampler entirely. Users now must provide velocity profiles as proper functions or interpolators. If a user needs only surface and drogue velocities, they should implement a step function accordingly. This clarifies the API: the library samples profiles, not discrete depths.

---

### 5d. `make_dd_velocity_interpolator` had wrong unit conversion (fixed)

`drifter.py` — `_get_corner_data_Agrid` returns raw field values in m/s (same as stored in the netCDF). The interpolator was treating these as deg/s and multiplying by `deg2m_lon ≈ 64,000`, feeding the DD model velocities ~64,000× too large → overflow → OOB particle deletion. Fixed: removed the erroneous input multiplication; the output conversion (m/s → deg/s) is still correct and necessary.

This was masked in notebook 01 (idealized flow) because the test fieldset used Cartesian `mesh="flat"`, not `mesh="spherical"`, so the spherical conversion branch was never hit.

---

### 5c. Convergence tolerance documented in physical terms

`get_final_drift_batch` previously used convergence criterion `max(|xdd|, |ydd|) < conv_tol=1e-4` m/s² with loose ODE solver tolerances (`atol=rtol=1e-3`), but the physical meaning was not documented.

**Decision:** Added explicit documentation: with `t_span=(0, 120)` s, the default tolerance corresponds to velocity convergence of ~0.012 m/s, adequate but deliberately loose relative to typical drift magnitudes (0.1–1.0 m/s). This makes the tolerance choice explicit and adjustable by users.