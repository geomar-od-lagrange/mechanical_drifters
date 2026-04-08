# Code review remarks — April 2026

Remarks from WR's review of `src/` on branch `wr/go-for-real-application`.
31 TODOs across `drifter.py`, `lagrange_model.py`, `stokes.py`.

---

## Themes

1. **Parcels coupling is too tight.** The profile sampler and interpolator hard-code A-grid internals. Swapping grid type means rewriting.
2. **Parameter passing is messy.** `_params()` dict, `LagrangeParams` NamedTuple, and positional arg ordering coexist without a clear contract.
3. **State vector index ordering is a footgun.** `Y[:, 2]` meaning `v` is implicit; nothing enforces or documents the mapping.
4. **Documentation gaps.** Several non-obvious mechanisms (CSE lambdify, `converged.terminal`, stereographic derivatives, `_sub`/`_flat` naming) are unexplained.
5. **API surface needs rationalization.** Multiple ways to get drift velocity with inconsistent signatures and return types.
6. **Minor naming/hygiene.** `_sub`, `_flat`, `conv_tol`, `_load_or_derive`, `u`/`v` ambiguity with velocity, missing formatter run.

---

## Immediate fixes

Small, unambiguous changes. No design decisions needed.

### IF-1: Docstring mentions FieldSet but function doesn't use it
- **File:** `drifter.py`, `make_profile_sampler` docstring
- **Remark:** `# TODO: Mentions FieldSet but doesn't use any?`
- **Fix:** Rewrite the docstring to describe what the function actually does (build a z-interpolator from pre-sampled arrays). Remove the FieldSet reference.

### IF-2: Default-param comment -> docstring
- **File:** `drifter.py`, `DroguedDrifter` class body
- **Remark:** `# TODO: turn into a docstring?`
- **Fix:** Move the Callies geometry comment block into the class or `__init__` docstring.

### IF-3: `conv_tol` -> more descriptive name
- **File:** `drifter.py`, `get_final_drift_batch`
- **Remark:** `# TODO: rename to more explaining name`
- **Fix:** Rename to `drift_accel_tol`. The convergence check is on `max(|xdd|, |ydd|)` — specifically the *drift* acceleration, not total. Update docstring accordingly.

### IF-4: Drop `theta0` parameter
- **File:** `drifter.py`, `get_final_drift_batch`
- **Remark:** `# TODO: It doesn't make any sense to pass a (meaningless w/o phi) theta0 alone.`
- **Fix:** Remove `theta0`. When `y0 is None`, start at equilibrium `(u_stereo, v_stereo) = (0, 0)` which is `theta = pi, phi = 0`. If a caller needs a non-equilibrium start, they pass `y0`.

### IF-5: Explain `_flat` naming
- **File:** `drifter.py`, `get_final_drift_batch`
- **Remark:** `# TODO: explain rational behind _flat naming`
- **Fix:** Add comment: `# _flat: (N, 8) state raveled to (8*N,) vector for solve_ivp's 1-D interface`.

### IF-6: Explain `converged.terminal` / `.direction`
- **File:** `drifter.py`, `get_final_drift_batch`
- **Remark:** `# TODO: Explain! This likely uses some solve_ivp details.`
- **Fix:** Add comment explaining that `solve_ivp` calls events at each step; `.terminal = True` stops integration when the event function crosses zero; `.direction = -1` means trigger only on downward zero-crossings (acceleration dropping below threshold).

### IF-7: Run black
- **File:** `lagrange_model.py`
- **Remark:** `# TODO: run black`
- **Fix:** `pixi run black src/`

### IF-8: Rename `_sub` suffix
- **File:** `lagrange_model.py`, `_derive_symbolic` return values
- **Remark:** `# TODO: Why call these _sub?`
- **Fix:** Rename `M_sub` / `F_sub` to `M_static` / `F_static` (they are the post-substitution forms with static symbols replacing dynamic ones).

### IF-9: Rename `_load_or_derive`
- **File:** `lagrange_model.py`
- **Remark:** `# TODO: use more meaningful function name here.`
- **Fix:** Rename to `_get_eom_callables`. **[Confirmed]**

### IF-10: Explain `_uv_to_spherical` derivative
- **File:** `lagrange_model.py`, `_uv_to_spherical`
- **Remark:** `# TODO: explain in comment!`
- **Fix:** Expand the comment to explain the chain rule: theta = f(r), phi = atan2(v, u), so thetad = d(theta)/dr * (u*ud + v*vd)/r, etc. Explain `safe` guards against division by zero at r=0.

### IF-11: Explain singularity in `_derive_symbolic` docstring
- **File:** `lagrange_model.py`, `_derive_symbolic`
- **Remark:** `# TODO: Here make sure to mention and briefly explain the nature of the singularity before referencing it.`
- **Fix:** Add a sentence before "This avoids the phi singularity" explaining: at theta=pi the azimuthal angle phi is undefined (any rotation around the vertical axis is the same physical configuration), which makes the (theta, phi) EOM singular there. The stereographic projection maps this point to the origin (u, v) = (0, 0) where everything is smooth.

### IF-12: Cite Liu et al. and Breivik
- **File:** `stokes.py`
- **Remark:** `# TODO: Cite Liu et al!?`
- **Fix:** Add Liu et al. reference. Also look up and include Breivik DOI for the exponential Stokes profile.

### IF-13: Define "shallow" with computed numbers
- **File:** `stokes.py`
- **Remark:** `# TODO: Give definition of shallow.`
- **Fix:** Add: shallow water means depth < half the deep-water wavelength (h < g*T_p^2 / (8*pi)). Compute actual depth thresholds with Python for both long swell (~10 s period) and short wind waves (~3-5 s period) and include both numbers.

### IF-14: Drop or hide `g` parameter
- **File:** `stokes.py`
- **Remark:** `# TODO: Parameter g really necessary?`
- **Fix:** Remove from the signature; use `g = 9.81` as a module-level constant. Nobody calls this with a different g.

---

## Deeper but clearly defined work

Design decisions are needed, but the scope is bounded.

### DW-A: Unify parameter passing

**Remarks:**
- `# TODO: Consolidate with named-tuple approach in lagrange model?`
- `# TODO: This feels like there should be a more straightforward way to pass around parameters.`
- `# TODO: Make sure we use the NamedTuple.` (on `M_func`)

**Problem:** Three mechanisms coexist for the same 15-19 parameters: `_params()` returns a dict, `LagrangeParams` is a NamedTuple, and `M_func`/`F_func` take `**kwargs`. The dict gets splatted into kwargs which get positionally aligned with sympy args. Each handoff is a place where ordering bugs can hide.

**Direction:** Settle on `LagrangeParams` (renamed `_EOMArgs`, see D-II) as the single source of truth. `_params()` should return an instance. `M_func` / `F_func` should accept it (or unpack one), not raw kwargs. This makes the contract explicit and type-checkable.

**Open question:** Where does the theta/phi -> u_stereo/v_stereo conversion happen? `_EOMArgs` uses stereographic internally. The conversion boundary needs to be clean — either the caller converts before constructing the tuple, or there's a factory method that takes spherical coords and converts.

**Depends on:** IF-8 (rename `_sub`), and resolving DW-B (state vector layout) since `LagrangeParams` currently includes state variables alongside physical parameters.

### DW-B: State vector layout as a first-class definition

**Remarks:**
- `# TODO: Where is the order of these determined? What if we change it there. Would we notice?`
- `# TODO: Again: Order of args. Just assuming [3] is v is a huge footgun.`

**Problem:** The 8-element state vector `[x, y, u, v, xd, yd, ud, vd]` is defined only implicitly by array indexing scattered across `rhs`, `_rhs_batch`, `get_final_drift_batch`, and the conversion functions. If the order ever changes in one place, the rest silently breaks.

**Direction:** Define indices once:
```python
IX, IY, IU, IV, IXD, IYD, IUD, IVD = range(8)
```
Use symbolic names everywhere: `Y[:, IU]` instead of `Y[:, 2]`. Consider whether this should be an IntEnum or just module-level constants.

**Additionally:** Write tests that enforce consistency. E.g. a test that constructs a known state, runs it through the public->internal->public round trip, and checks that each component ends up in the right slot. This catches ordering bugs even without named indices.

### DW-C: Document and simplify the CSE/lambdify pipeline

**Remarks:**
- `# TODO: This solution needs more explanation. How's the CSE logic lambdified exactly?`
- `# TODO: Awkward comment. Explain what's done. Not where we read it.`
- `# TODO: This feels really hacky. Is there no other way? Can we make sympy lambdifies accept kwargs?`
- `# TODO: the --- split feels awkward. Is there a proper way to do this?`
- `# TODO: Brief comment explaining the symbols. Do we need the static symbols?`

**Problem:** `_apply_cse_and_lambdify` generates Python source via string formatting and `exec`, loads a `---`-delimited `.srepr` cache file, and maps between dynamic/static symbols. The pipeline works but is opaque and fragile.

**Direction (incremental, don't rewrite from scratch):**
1. Document the pipeline in a top-of-module comment: derive -> substitute static symbols -> CSE -> codegen -> exec -> wrap.
2. Replace `---`-delimited srepr with a proper format (e.g. a dict serialized via `pickle` or a `.py` file with the generated functions directly).
3. Test whether `sp.lambdify` with `cse=True` (added in sympy 1.12) can replace the manual CSE + exec. If so, the entire `_apply_cse_and_lambdify` function collapses to a few lines.
4. Check if the static-symbol substitution is still necessary with modern sympy's lambdify. **Context:** At `38d7c31` (the pre-stereographic version on this branch), `_derive_and_lambdify` lambdifies dynamic symbols directly with no static substitution and it works. The substitution was introduced later alongside the stereographic reparameterization — check whether it was actually needed or just cargo-culted in.

### DW-D: Inline `_eval_M_F` / simplify parameter flow

**Remarks:**
- `# TODO: This feels like there should be a more straightforward way to pass around parameters. Maybe we just inline the _eval_M_F part into rhs and use LagrangeParameters namedtuple there?`

**Problem:** `_eval_M_F` unpacks currents, calls `_params()`, splats the dict into `M_func`/`F_func`. It's a thin wrapper that adds indirection without value.

**Direction:** Once DW-A is done, `rhs` can build the params tuple directly and call `M_func`/`F_func`. `_eval_M_F` becomes unnecessary.

**Depends on:** DW-A.

### DW-E: `z_eff` clamp — keep it, don't warn at runtime

**Remark:** `# TODO: Clamp necessary? It doesn't make sense to have zeff > 0. But just clamping here without raising anything is worse than letting unphysical configs emerge.`

**Problem:** `np.minimum(0.0, self.l * cos_theta)` silently clamps the drogue to z <= 0. If the pole tilts past horizontal (theta < pi/2), the drogue would be above the surface, which is unphysical.

**Resolution:** Keep the clamp as a safety net. No per-call warnings — this is a hot path. If the drogue somehow swings above the surface, the uv callback will likely error on positive z anyway. With extreme initial conditions (e.g. theta=pi but huge lateral speed relative to water) the drogue could theoretically flip up — we accept this as outside the physical operating regime.

Add a comment documenting this decision.

### DW-F: Rationalize the drift-velocity API

**Remark:** `# TODO: There's breaks in signature and return types here and there's likely a lot of room for cleanup.`

**Problem:** Multiple ways to get drift velocity:
- `rhs(t, y)` — scalar, internal coords, returns derivatives
- `_rhs_batch(Y, sample_uv)` — batched, internal coords
- `get_final_drift_batch(sample_uv, ...)` — batched, public coords in/out, returns `(xd, yd, theta, Y)`
- `get_full_solution(t_span, ...)` — scalar, spherical in, xarray out
- `get_final_drift_velocity(t_span, ...)` — scalar, spherical in, `(xd, yd)` out
- `_get_full_solution(t_span, y0, ...)` — raw solve_ivp wrapper

Some of these are internal, some public. The scalar and batch paths have different conventions and don't share code.

**Direction:**
1. Mark internal methods clearly (`_rhs`, `_rhs_batch`, `_get_full_solution`).
2. Decide on public surface: likely `get_full_solution` (for analysis/plotting) and `get_final_drift_batch` (for Parcels integration). `get_final_drift_velocity` is a convenience wrapper — keep or drop.
3. Unify input convention: public methods take spherical `(theta, phi)`, convert internally. No mixing.
4. Unify return convention: public methods return xarray or plain arrays in spherical coords. Internal methods use `(u_stereo, v_stereo)`.

---

## Starting points for discussion — resolved

These were open questions. WR's feedback gives clear direction on each.

### D-I: Parcels integration architecture

**Remarks:**
- `# TODO: Put the parcels-related functions into a parcels_v4.py`
- `# TODO: Let's make sure we understand how FieldSet leverages Xarray and grids.`
- `# TODO: Again, can we just let Parcels retrieve the profile in a grid agnostic way for us?`

**Decision:** Go the slow (grid-agnostic) way. Also tackle the warm_state cache validation issue (existing TODO in `make_dd_velocity_interpolator`: cache only checks particle count, so stale state can be silently reused when particles are deleted OOB) as part of this refactor.

1. **Isolate now:** Move all Parcels-related code into `parcels_v4.py`.
2. **Grid-agnostic sampling:** Call Parcels' existing single-point interpolation D times (once per depth level) instead of reaching into `_get_corner_data_Agrid`. Make the z-level lookups parameters. This is slower but works with any grid type. If necessary, build something that is at least vertical-grid agnostic by iterating fieldset.UV calls at each z level.
3. **Upstream engagement:** Work this out cleanly. The implementation will be shown to the Parcels devs to inform future design decisions about a profile-sampling API.

### D-II: LagrangeParams naming and visibility

**Decision:** Rename to `_EOMArgs` (or similar private name). No users will touch this — the user-facing surface is Parcels and the `DroguedDrifter` public methods.

Split into physical constants (fixed per drifter) and state+forcing (changes every timestep) if it helps clarity.

### D-III: `u`/`v` naming ambiguity

**Decision:** Use `u_stereo`/`v_stereo` in Python variable names, `u_st`/`v_st` in sympy symbols (both dynamic and static forms). Note: `u_s`/`v_s` already exist in `lagrange_model.py` as the static replacement symbols (`sp.symbols("u_s v_s", real=True)`), so those names are taken. Use `u_st`/`v_st` for sympy to avoid the collision — the `_st` suffix then does double duty (static + stereographic), which is fine since the static symbols *are* the stereographic coordinates.
