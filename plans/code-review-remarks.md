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

## Immediate fixes — DONE

All completed and tested (94 tests pass).

- [x] **IF-1:** Rewrote `make_profile_sampler` docstring, removed FieldSet reference.
- [x] **IF-2:** Moved Callies geometry defaults into class docstring.
- [x] **IF-3:** Renamed `conv_tol` to `drift_accel_tol`.
- [x] **IF-4:** Removed `theta0`; defaults to equilibrium `(u_stereo, v_stereo) = (0, 0)`.
- [x] **IF-5:** Added `_flat` naming explanation comment.
- [x] **IF-6:** Added `converged.terminal` / `.direction` explanation.
- [x] **IF-7:** Ran black.
- [x] **IF-8:** Renamed `M_sub`/`F_sub` to `M_static`/`F_static`.
- [x] **IF-9:** Renamed `_load_or_derive` to `_get_eom_callables`.
- [x] **IF-10:** Expanded `_uv_to_spherical` chain rule comment.
- [x] **IF-11:** Explained phi singularity in `_derive_symbolic` docstring.
- [x] **IF-12:** Added Liu et al. (2021) JAMES and Breivik et al. (2016) Ocean Modelling citations.
- [x] **IF-13:** Defined "shallow" with computed thresholds (39 m for 10 s swell, 6 m for 4 s wind waves).
- [x] **IF-14:** Removed `g` param; module-level `_G = 9.81`.

---

## Deeper but clearly defined work

Design decisions are needed, but the scope is bounded.

### DW-A: Unify parameter passing — DONE

**Remarks:**
- `# TODO: Consolidate with named-tuple approach in lagrange model?`
- `# TODO: This feels like there should be a more straightforward way to pass around parameters.`
- `# TODO: Make sure we use the NamedTuple.` (on `M_func`)

**Problem:** Three mechanisms coexist for the same 15-19 parameters: `_params()` returns a dict, `LagrangeParams` is a NamedTuple, and `M_func`/`F_func` take `**kwargs`. The dict gets splatted into kwargs which get positionally aligned with sympy args. Each handoff is a place where ordering bugs can hide.

**Direction:** Settle on `LagrangeParams` (renamed `_EOMArgs`, see D-II) as the single source of truth. `_params()` should return an instance. `M_func` / `F_func` should accept it (or unpack one), not raw kwargs. This makes the contract explicit and type-checkable.

**Open question:** Where does the theta/phi -> u_stereo/v_stereo conversion happen? `_EOMArgs` uses stereographic internally. The conversion boundary needs to be clean — either the caller converts before constructing the tuple, or there's a factory method that takes spherical coords and converts.

**Depends on:** IF-8 (rename `_sub`), and resolving DW-B (state vector layout) since `LagrangeParams` currently includes state variables alongside physical parameters.

### DW-B: State vector layout as a first-class definition — DONE

Defined `IX, IY, IU, IV, IXD, IYD, IUD, IVD = range(8)` as module-level constants in `drifter.py`. All magic-number indexing replaced with named constants. Round-trip test (`test_state_vector_round_trip`) added.

### DW-C: Document and simplify the CSE/lambdify pipeline — DONE

Replaced the entire codegen pipeline (`_apply_cse_and_lambdify`, 70 lines of string formatting + `exec`, `NumPyPrinter` import, `---`-delimited `.srepr` cache) with `sp.lambdify(..., modules='numpy', cse=True)` + hash-keyed pickle cache (`_load_or_derive`). Also lambdifies `qdd = M^{-1}F` directly for a 2.4x speedup on the hot path. The `_qdd_func` internal function replaces `np.linalg.solve` in both `rhs` and `_rhs_batch`. CLI (`cli.py`) and `click` dependency removed. Broadcasting contract tests added.

### DW-D: Inline `_eval_M_F` / simplify parameter flow — DONE

`_eval_M_F` deleted. `rhs` and `_rhs_batch` now build `EOMState` directly and call `_qdd_func`. No more `M_func`/`F_func` + `np.linalg.solve` on the hot path.

### DW-E: `z_eff` clamp — DONE

Kept the clamp as a safety net. Added a comment documenting the decision: no per-call warnings on this hot path; extreme initial conditions that flip the drogue above the surface are outside the physical operating regime.

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
   `rhs` should be `_rhs` — it's a solve_ivp callback, not user API.
   `default_uv` should be `_default_uv` — testing convenience only.
2. Decide on public surface: likely `get_full_solution` (for analysis/plotting) and `get_final_drift_batch` (for Parcels integration). `get_final_drift` is a convenience wrapper — keep or drop.
3. Unify input convention: public methods take spherical `(theta, phi)`, convert internally. No mixing.
4. Unify return convention: public methods return xarray or plain arrays in spherical coords. Internal methods use `(u_stereo, v_stereo)`.
5. After DW-C, `_rhs` and `_rhs_batch` share a two-line core (`EOMState(...)` + `_qdd_func(...)`). The surrounding code (unpack, sample currents, assemble output) differs inherently between scalar and batch paths. Keep them separate — forced consolidation adds more boilerplate than it removes.
6. **Check test coverage** before and after. Renaming public → private methods will break any external tests or notebooks that call them. Run coverage to identify untested paths and dead code that can be dropped rather than renamed.

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
