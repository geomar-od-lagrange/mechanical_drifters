# Code quality review

Overall quality is high. Naming is consistent, docstrings are thorough
and follow Google style, the code is Pythonic, and the test suite is
well-structured. The findings below are mostly low-severity polish items.

## Naming

### N1. `_mag` helper is too generic (low, trivial)

`lagrange_model.py` line 83: `_mag(vec)` computes the symbolic Euclidean
norm. Rename to `_sym_norm` or `_euclidean_norm` for clarity.

### N2. `_dummy_args` magic number 19 (medium, trivial)

`parcels_v4.py` line 186: `range(19)` is the total field count of
`DrifterPhysics` + `EOMState`. Use `len(DrifterPhysics._fields) +
len(EOMState._fields)` instead of a magic number. If someone adds a
field, this silently breaks.

### N3. Inconsistent `V` for potential energy vs velocity (low, n/a)

`lagrange_model.py` line 160: local variable `V` shadows the
conventional use of `V` for velocity (e.g. `V_b`, `V_d` in the same
function). Not a bug since scoping is correct, but it adds cognitive
load. Consider `PE` or `U_pot` for potential energy.

### N4. Inconsistent variable naming: `x_pos`/`y_pos` static symbols (low, n/a)

`lagrange_model.py` line 196: the static substitutions `x_static`,
`y_static` map to symbol names `x_pos`, `y_pos`, which never appear in
`DrifterPhysics` or `EOMState`. This is harmless because those symbols
only appear in `M`/`F` expressions that are lambdified, but the naming
disconnect with the struct fields (`x`, `y` do not exist in either
struct) is confusing. A comment explaining this would help.

## Docstrings

### D1. `_build_packer` missing Args/Returns sections (low, trivial)

`lagrange_model.py` line 52: the docstring is a prose description. Add
formal `Args:` and `Returns:` sections for consistency with the rest of
the module.

### D2. `_rhs_batch` docstring says "returns ... `dY/dt`" but doesn't mention NaN guard (low, trivial)

`drifter.py` line 231: the NaN guard at line 275 silently replaces
non-finite accelerations with zero. This is a behavioral detail worth
documenting in the docstring.

### D3. `_cache_key` missing docstring Args/Returns (low, trivial)

`lagrange_model.py` line 247: one-liner docstring is fine for private
helpers, but adding `Returns: str` would match the style of `_load_or_derive`.

### D4. `EOMState` field comments use `float` but fields hold arrays too (low, trivial)

`lagrange_model.py` lines 40-49: type annotations say `float` but the
docstring says "Fields hold scalars (in rhs) or (N,) arrays (in
_rhs_batch)." The type hints are misleading. Consider
`float | np.ndarray` or remove the annotations and rely on the docstring.

### D5. Stale reference to deleted function in `parcels_v4.py` module docstring (high, trivial)

Module docstring references `make_dd_velocity_interpolator` which no
longer exists in the codebase. Remove or rewrite the reference.

### D6. `DDAdvectEE` docstring is misleading (high, small)

`parcels_v4.py`: (1) Calls itself a "Parcels kernel" but has a
non-standard signature `(particles, fieldset, *, dd)` — it's a helper
function wrapped by `make_dd_kernel`, not a kernel itself. (2) Does not
document that it mutates `particles.dlon`/`particles.dlat` in place.
(3) Usage example references `DeleteOOB` which is not defined or imported
in this module.

### D7. `make_dd_kernel` missing side-effect documentation (high, trivial)

`parcels_v4.py`: When `backend="numba"`, the function mutates
`dd._qdd_func` as a side effect. This is not mentioned in the docstring.
Also describes the numba backend as "faster batch evaluation" — it
actually JIT-compiles the raw scalar/array callable, not a batch-specific
version.

### D8. `_default_uv` docstring inaccurate (medium, trivial)

`drifter.py` line 163: says "Returns sheared currents" but the function
is a hardcoded two-value step function returning `(1.0, 1.0)` at `z==0`
and `(-1.0, -1.0)` below. Not a shear profile.

### D9. `_derive_symbolic` return description imprecise (medium, trivial)

`lagrange_model.py`: describes `F_static` as a "4×1 force vector" but
it is actually the RHS from `sp.linear_eq_to_matrix` — the negated
residual of `M·q̈ = F`, not the generalized force Q itself.

### D10. `_z_eff` missing Args section entirely (medium, trivial)

`drifter.py` line 211: parameters `u` and `v` (stereographic
coordinates) are completely undocumented — no `Args:` section at all.

## Pythonic idioms

### P1. LBYL pattern in `_default_uv` (low, trivial)

`drifter.py` line 175: `if z == 0.0` is look-before-you-leap and will
fail for array `z`. This is only used for scalar single-particle
testing, so it works, but the inconsistency with the batch path (which
uses `np.all(z == 0)`) is a code smell. Not blocking since the scalar
and batch paths use different callbacks.

### P2. `bare except` in `_load_or_derive` (low, trivial)

`lagrange_model.py` line 267: `except Exception: pass` when loading the
pickle is intentionally broad (corrupt pickle, stale data, etc.). This
is acceptable for a cache loader, but adding a logging statement or
`warnings.warn` would aid debugging when the cache silently fails.

### P3. Lazy import of `xarray` in `get_full_solution` (low, n/a)

`drifter.py` line 476: `import xarray as xr` is inside the method body.
This is fine for optional-dependency isolation. No change needed, just
noting the pattern is intentional.

## DRY violations

### R1. `_DEFAULT_PHYSICS` duplicated across test files (medium, small)

The exact same `DrifterPhysics(m_b=1.0, m_d=2.7, ...)` definition
appears in:
- `tests/test_numerical_edge_cases.py` line 22
- `tests/test_lagrange_physics.py` line 22
- `tests/test_drogued_drifter.py` line 306

Extract to a shared `tests/conftest.py` fixture or constant.

### R2. `haversine_km` duplicated in example notebooks (low, n/a)

`examples/baltic_drifters/05_validation_plots.py` line 103 and
`examples/baltic_drifters/07_short_horizon_skill.py` line 65 define the
same function. For notebooks this is acceptable (self-contained cells),
so no action needed.

### R3. `DeleteOOB` kernel duplicated in examples (low, n/a)

Defined identically in `04_run_simulations.py`, `06_run_short_simulations.py`,
and `02_sheared_jet_parcels.py`. This is a Parcels convention (kernels
must be plain functions), so duplication is expected. If it grows, a
shared utility module would help, but not urgent.

### R4. M upper-triangle assembly duplicated (low, small)

The pattern of unpacking 10 upper-triangle elements and assembling a
symmetric 4x4 matrix appears in `M_func` (line 380) and in
`test_lagrange_symbolic_fallback.py` (line 142). The test copy is
acceptable (testing the same logic independently), so this is not
actionable.

### R5. `_make_flat_fieldset` / `_make_spherical_fieldset` in test_drifter_parcels (low, n/a)

Two helper functions that are very similar in structure but differ in
coordinate names and mesh type. Refactoring into a single parametric
helper would reduce code, but the current form is clear and test-local.
Leave as-is.

## Dead code

### X1. Unused import: `StatusCode` in `02_sheared_jet_parcels.py` (low, trivial)

Line 40 imports `StatusCode` which is only used inside `DeleteOOB`, but
also imports `Variable` and `FieldOutOfBoundError` which are never used
in any code cell.

### X2. `_step_sampler` defined but duplicates `make_profile_sampler` logic (low, n/a)

`tests/test_drogued_drifter.py` line 20: `_step_sampler` is a
test-specific step-function sampler, distinct from `make_profile_sampler`.
Not actually dead code, just noting it exists alongside the similar
`_make_const_uv` helper (line 162). Both are used.

## Code smells

### S1. `TODO` comment in production code (medium, small)

`parcels_v4.py` line 175: `# TODO: This logic should live deeper down.
We should treat the numpy and numba _qdd eval as two first-class
implementations.` This documents a known asymmetry between the numpy and
numba branches of `make_dd_kernel`. The if/elif has structurally
different logic (the numba branch mutates `dd._qdd_func`). Track this in
[backlog.md](BACKLOG.md) and remove the TODO from source.

### S2. `numba` branch mutates `dd._qdd_func` as side effect (medium, n/a)

`parcels_v4.py` line 196: `dd._qdd_func = _qdd_func_numba` mutates the
instance from inside a factory function. The CLAUDE.md convention says
this is acceptable for internal APIs, and the TODO already flags it.
Just noting it as a smell for the architecture reviewer.

### S3. `_rhs_batch` silently zeros NaN accelerations (low, n/a)

`drifter.py` line 276: `qdd[bad] = 0.0` suppresses non-finite results
without warning. This is defensible on a hot path, and the docstring
comment explains the rationale. See D2 above for documentation.

### S4. `test_drogued_drifter_instantiation` has no assertion (low, trivial)

`tests/test_drogued_drifter.py` line 37: the test body is just
`dd = DroguedDrifter()` with no assertion. While this tests that
construction does not raise, adding `assert dd.physics.l == 3.0` or
similar would make intent explicit.

### S5. Long lines in test files (low, trivial)

Several test files have lines exceeding 120 characters, particularly
`EOMState(...)` constructors in `test_lagrange_physics.py` (lines
99-101, 117-122, etc.) and `test_drogued_drifter.py` (lines 398-401).
These are borderline -- the constructors are readable as-is -- but
breaking them across lines would improve readability.

### S6. `Variable` and `FieldOutOfBoundError` unused imports (low, trivial)

`examples/idealized_flow/02_sheared_jet_parcels.py` line 39-40:
`Variable` and `FieldOutOfBoundError` are imported but never used.

## Batch fixes

These can be done mechanically by a Sonnet/Haiku agent in one pass:

1. **Remove unused imports** in example notebooks: `Variable`,
   `FieldOutOfBoundError` in `02_sheared_jet_parcels.py`. Scan all
   example `.py` files for unused imports.

2. **Extract `_DEFAULT_PHYSICS`** into `tests/conftest.py` and import in
   `test_numerical_edge_cases.py`, `test_lagrange_physics.py`, and
   `test_drogued_drifter.py`.

3. **Replace magic `19` with expression** in `parcels_v4.py` line 186:
   `len(DrifterPhysics._fields) + len(EOMState._fields)`.

4. **Add `assert` to `test_drogued_drifter_instantiation`**: e.g.
   `assert dd.physics.l == 3.0`.

5. **Move TODO from `parcels_v4.py` line 175** to `plans/BACKLOG.md` and
   remove the inline comment.

6. **Add `Args:`/`Returns:` to `_build_packer` docstring**.

7. **Break long `EOMState(...)` lines** in test files to stay under 120 chars.

8. **Fix stale docstrings (D5–D10)**: Remove deleted function reference
   from `parcels_v4.py` module docstring (D5). Rewrite `DDAdvectEE`
   docstring to clarify it's a helper, document in-place mutation, fix
   usage example (D6). Document `make_dd_kernel` numba side effect (D7).
   Fix `_default_uv` description (D8). Fix `_derive_symbolic` return
   description for `F_static` (D9). Add `Args:` section to `_z_eff` (D10).

## Resolved by architecture changes

The following items were resolved as side effects of implementing
C1+I2+I3+I4 from `review-architecture.md`:

- **D5** (stale module docstring referencing deleted function): Rewritten
  in `parcels_v4.py` module docstring.
- **D6** (DDAdvectEE docstring misleading): Rewritten to clarify it is a
  helper function, documents in-place mutation of `particles.dlon`/`dlat`,
  and removes the stale `DeleteOOB` usage example.
- **D7** (make_dd_kernel missing side-effect documentation): No longer
  applicable -- `make_dd_kernel` no longer has a backend parameter or
  side effects. Backend selection moved to `DroguedDrifter.__init__`.
- **S1** (TODO comment in production code): Removed. The TODO at
  `parcels_v4.py:175` is obsolete now that backend selection lives in
  `lagrange_model._make_qdd_func` and `DroguedDrifter.__init__`.
- **S2** (numba branch mutates `dd._qdd_func`): No longer applicable.
  `_qdd_func` is set once at construction time and never mutated.
- **N2** (magic number 19 in `parcels_v4.py`): Removed along with the
  entire numba branch in `make_dd_kernel`. The JIT warmup with computed
  field count now lives in `lagrange_model._make_qdd_func`.
- **R1** (duplicated `_DEFAULT_PHYSICS` across test files): Extracted to
  `tests/conftest.py` as `DEFAULT_PHYSICS` and imported in all three files.
