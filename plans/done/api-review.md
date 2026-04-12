# API review: `drogued_drifters` package

Full inventory of the public and internal API surface in `src/drogued_drifters/`.

## Public API (`__init__.py`)

Re-exported names:

```python
DroguedDrifter, DrifterPhysics, EOMState, M_func, F_func, qdd_func
```

`stokes.compute_stokes_profile` is **not** re-exported.

FB: Should export stokes public api

---

## `lagrange_model.py` — Symbolic EOM derivation and numeric evaluation

### Data structures (NamedTuples)

| Name | Fields | Role |
|---|---|---|
| `DrifterPhysics` | `m_b, m_d, m_hat_d, m_tilde_d, m_tilde_b, l, g, k_b, k_d` (9 fields) | Frozen physical constants, set once per drifter instance. Not used directly in examples — only constructed internally by `DroguedDrifter.__init__` |
| `EOMState` | `u_stereo, v_stereo, xd, yd, ud_stereo, vd_stereo, U_b, V_b, U_d, V_d` (10 fields) | Per-timestep state + forcing; scalar or `(N,)` arrays. Not used directly in examples — only constructed internally in `_rhs` / `_rhs_batch` |

### Public functions

- **`qdd_func(physics, state)`** — Evaluate generalized accelerations `qdd = M⁻¹F`. Returns `(4,)` scalar or `(N, 4)` batch. Not used in examples.

FB: delete qdd func then or rewire to be used in parcels_v4 or examples. 

- **`M_func(physics, state)`** — Evaluate 4×4 mass matrix. Returns `(4, 4)` or `(N, 4, 4)`. Not used in examples.
- **`F_func(physics, state)`** — Evaluate 4-element force vector. Returns `(4,)` or `(N, 4)`. Not used in examples.

FB: Let's add an example?

### Internal functions

- **`_derive_symbolic()`** — (LRU-cached) Full Lagrangian derivation in stereographic coordinates. Returns `(M_static, F_static, args)`.
- **`_load_or_derive()`** — Pickle-cached wrapper around `_derive_symbolic()` + `M⁻¹F` solve. Keyed on source hash + sympy version + Python version.
- **`_get_eom_callables()`** — (LRU-cached) Lambdifies M, F, qdd with CSE; returns `(qdd_raw, M_raw, F_raw, args, pack_eom_args)`.
- **`_make_qdd_func(backend)`** — Factory for qdd evaluator. Supports `"numpy"` and `"numba"` backends. Returns `qdd_func(physics, state)`.

FB: Let's dive in how this is solved exactly. Does _make_qdd_func create the func that is exported publicly?

- **`_qdd_func(physics, state)`** — Convenience wrapper using numpy backend.

FB: Convenience wrapper for something with two backends which we treat as almost symmetric feels imbalanced. Delete.

- **`_build_packer(raw_func)`** — Inspects lambda signature, returns `pack_eom_args(physics, state)` that maps NamedTuple fields to positional args by name.

FB: Let's dive in here as well.

- **`_sym_norm(vec)`** — Symbolic `sqrt(v·v)` helper.
- **`_cache_key()`** — SHA256 hash for pickle cache invalidation.

### Coordinate transforms

- **`_uv_to_theta(u, v)`** — Stereographic → polar angle theta only.
- **`_uv_to_spherical(u, v, ud, vd)`** — Stereographic → `(theta, phi, thetad, phid)`. Handles `r=0` (equilibrium) safely.

FB: Why split in theta only? Because we use theta only sometimes and want to avoid overhead of also getting phi?

- **`_spherical_to_uv(theta, phi, thetad, phid)`** — Inverse of above.

### Module-level constants

- **`_CACHE_PATH`** — `Path(...)/data/eom_cache.pkl`.

---

## `drifter.py` — ODE integrator and `DroguedDrifter` class

### Drag / added-mass helpers (module-level, public)

All keyword-only, return a single float:

- **`drogue_horizontal_added_mass(*, rho, w_d, h_d, C_perp_d=pi/4)`** → `m_tilde_d`. Not used in examples.
- **`buoy_horizontal_added_mass(*, rho, d_b, h_b, C_perp_b=1.0)`** → `m_tilde_b`. Not used in examples.
- **`drogue_horizontal_drag_coeff(*, rho, w_d, h_d, C_D_d=1.2)`** → `k_d`. Not used in examples.
- **`buoy_horizontal_drag_coeff(*, rho, d_b, h_b, C_D_b=1.0)`** → `k_b`. Not used in examples.

FB: Let's use in examples. Make a separate one which demonstrates the public functions from lagrange model and these parameters?

### Velocity adapter (internal)

- **`_adapt_get_uv(get_uv)`** — Wraps a scalar `get_uv(*, t, x, y, z)` callback into the batch `sample_uv(z)` protocol (called with `t=0, x=0, y=0`).

FB: This is broken. I'd say we think about how we can broadcast get_uv to non-scalar t,z,y,x. 

FB: Also make sure we're always using c-ordering t, z, y, x for args etc. Esp for positional args this is error prone otherwise.

### State vector layout (module-level constants)

```python
IX, IY, IU, IV, IXD, IYD, IUD, IVD = range(8)
```

### `DroguedDrifter` class

**Constructor** — all keyword-only, Callies et al. defaults:

```python
DroguedDrifter(
    *, m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
    l=3.0, k_b=12.0, k_d=154.0, g=9.81,
    get_uv=None, sample_uv=None, backend="numpy",
)
```

Two velocity interfaces (`get_uv` xor `sample_uv`, mutually exclusive). Backend selects numpy or numba for the qdd evaluator.

FB: Let's get rid of the adapter entirely.

FB: Let's even think about if we can get rid of the drifter class entirely. In the end it's only a complicated wrapper for a few physics properties and awkwardly wired methods. In any case, we need to find a much cleaner and leaner solution here. No adapters etc. We can also just have ..._numpy() and ..._numba() versions of functions. The only place where a factory could be in order is in wiring up the parcels kernel, I think.

**Attributes:**
- `physics` — `DrifterPhysics` instance
- `backend` — `"numpy"` or `"numba"`
- `_qdd_func` — bound qdd evaluator
- `_sample_uv` — active velocity sampler

#### Public methods

| Method | Returns | Notes |
|---|---|---|
| `get_full_solution(*, t_span, x, y, theta, phi, xd, yd, thetad, phid, t_eval, atol, rtol)` | `xr.Dataset` with `(time,)` variables: `x, y, theta, phi, xd, yd, thetad, phid` | Single-particle, full time series. Used in `03_drogued_drifter_in_wave_orbitals` for visualizing pole dynamics |
| `get_final_drift(*, t_span, x, y, theta, phi, xd, yd, thetad, phid)` | `(xd_final, yd_final, max_accel)` | Single-particle steady state. Not used in examples |
| `get_final_drift_batch(*, sample_uv, t_span, y0, atol, rtol)` | `(xd_final, yd_final, Y_final, max_accel)` — `(N,)`, `(N,)`, `(N, 8)`, scalar | Batch N-particle steady state. `Y_final` is in public `(theta, phi)` format for warm-starting. Used in `01_synthetic_flow_profiles` for direct drift computation; called indirectly via `DDAdvectEE` in all Parcels notebooks |

#### Private methods

| Method | Signature | Role |
|---|---|---|
| `_rhs(t, y)` | `(t, y[8])` → `dy[8]` | Scalar ODE RHS for `solve_ivp` |
| `_rhs_batch(Y)` | `(N, 8)` → `(N, 8)` | Vectorized RHS for batch solver |
| `_z_eff(u, v)` | stereo coords → depth `(N,)` | Effective drogue depth, clamped ≤ 0 |
| `_get_final_drift_batch_impl(...)` | | Core batch implementation (save/restore separated) |
| `_solve(t_span, y0, ...)` | | Raw `solve_ivp` wrapper |
| `_default_sample_uv(z)` | static | Placeholder: ±1 m/s step function |
| `_default_uv(*, t, x, y, z)` | | Scalar equivalent of above |

---

## `parcels_v4.py` — Parcels v4 coupling layer

### Public interface

- **`make_dd_kernel(dd)`** — Factory: captures a `DroguedDrifter` in a closure, returns a `(particles, fieldset)` kernel for `pset.execute()`. Parcels v4 requires plain `def` functions (no `functools.partial`). Used in `02_sheared_jet_parcels`, `04_run_simulations`, `06_run_short_simulations` for Parcels integration.
- **`make_profile_sampler(depth_levels, U_profiles, V_profiles)`** — Builds a fast `sample_uv(z)` linear interpolator from pre-sampled `(D, N)` velocity profiles. Depth levels must be sorted ascending (z-up). Not used directly in examples — only used internally by `_extract_profiles`.

FB: Think about the make profile sampler together with my architectural feedback on the drifter class. This is too entangled. Remember, we can internally break anything we want.

- **`DDAdvectEE(particles, fieldset, *, dd)`** — Advection logic: extract profiles → solve steady-state drift → Euler-forward position update. Not a Parcels kernel itself (extra `dd` kwarg). Not used directly in examples — only used internally by `make_dd_kernel`.

### Internal functions

- **`_extract_profiles(particles, fieldset, dd)`** — Samples fieldset at depth levels via `fieldset.UV.eval()`. Handles spherical → m/s conversion. Returns `sample_uv` callable.
- **`_position_update(particles, xd_ms, yd_ms, fieldset)`** — Euler-forward displacement: m/s → degrees on spherical grids, direct on flat. Mutates `particles.dlon`/`dlat`.

### Module-level constants

- **`_DEG2M = 1852.0 * 60.0`** — degrees-to-meters conversion factor.

---

## `stokes.py` — Stokes drift profiles (not re-exported)

- **`compute_stokes_profile(surface_u, surface_v, peak_period, depth_levels, g=None)`** — Deep-water exponential Stokes drift profile using `k = ω²/g` dispersion and `exp(2kz)` decay. Returns `(stokes_u, stokes_v)` with shape `(D, ...)`. Depth levels must be sorted ascending. Overestimates decay in shallow water. Used in `02_derive_effective_currents` for adding wave-driven currents to depth profiles, and in `03_drogued_drifter_in_wave_orbitals` for building a Stokes-drift velocity sampler.
- **`_G = 9.81`** — default gravitational acceleration.

FB: General. Let's think about this from the examples PoV first. In the examples, find all uses of the API in src/ and flag those that are awkward. (Keep in mind, we'll add an example for the pure lagrange model physics.) Then, let's decide how to address the architecture challenges from above. What do we need from the outside perspective? And how do we structure this internally? Let's treat numpy ans numba as almost on par from a user perspective. Let's still make sure the parcels-related logic is clear and easy to grasp as we plan to communicated this to the parcels team. But really try to come up with a simpler internal structure.