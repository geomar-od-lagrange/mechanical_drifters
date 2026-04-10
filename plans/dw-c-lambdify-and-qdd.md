# DW-C: Replace codegen pipeline with lambdify(cse=True) + direct qdd

## Motivation

Three wins in one:

1. **Drop the codegen pipeline.** `_apply_cse_and_lambdify` (70 lines of string
   formatting, `NumPyPrinter`, `exec`) is replaced by `sp.lambdify(..., cse=True)`.
   Sympy 1.12+ handles CSE internally and generates numpy-broadcasting code.

2. **Lambdify qdd = M^{-1}F directly.** Instead of lambdifying M and F separately
   and calling `np.linalg.solve` at runtime, we solve symbolically (instant for 4x4)
   and lambdify the result. The hot path becomes one function call.

3. **Replace .srepr cache with hash-keyed pickle.** Drop the custom `---`-delimited
   `.srepr` format, `_save_eom_cache`, and all parsing code. Use a pickle file keyed
   by a hash of `_derive_symbolic`'s source code + sympy version. Stale cache
   auto-invalidates.

## Measured numbers

**Symbolic derivation timing:**

| Step | Time |
|---|---|
| `_derive_symbolic` (uncached) | 135s |
| `M.LUsolve(F)` | <0.1s |
| `lambdify(cse=True)` for qdd | 0.1s |

**Cache load path (pickle + re-lambdify):**

| Step | Time | Size |
|---|---|---|
| Pickle load | <10ms | 5 KB |
| Re-lambdify from pickle | 70ms | — |
| **Total** | **~70ms** | — |

(Compare: current `.srepr` is 57 KB.)

**Runtime performance (scalar, the solve_ivp hot loop):**

| Approach | Time/call |
|---|---|
| Current (M_func + F_func + np.linalg.solve) | 12 us |
| Direct qdd_func | 5 us |
| **Speedup** | **2.4x** |

**Batch (N=1000):**

| Approach | Time/call |
|---|---|
| Current | 453 us |
| Direct qdd_func | 214 us |
| **Speedup** | **2.1x** |

Numerical agreement: max diff < 7e-15 (machine epsilon).

## Architecture after

```
_load_or_derive()           → M, F, qdd_exprs, args        (pickle cache, hash-keyed)
       |                      135s derive on miss, 70ms on hit
       v
_get_eom_callables()        → qdd_raw, M_raw, F_raw,       (lambdify + _build_packer)
       |                      pack_eom_args                  (all lru_cached)
       v
_qdd_func(physics, state)   → (4,) or (N,4) array          (hot-path entry point)
M_func(physics, state)      → (4,4) or (N,4,4) array       (public API, off hot path)
F_func(physics, state)      → (4,) or (N,4) array          (public API, off hot path)
       v
rhs() / _rhs_batch()        → qdd = _qdd_func(physics, state)  (one call, no solve)
```

## Caching strategy

**What:** Pickle `(M_static, F_static, qdd_exprs, args)` — symbolic expressions
only, not lambdified callables (those can't be pickled).

**Where:** `data/eom_cache.pkl` next to the package (same location as current `.srepr`).
Committed to git (like the current `.srepr`), so fresh clones don't trigger 135s
derivation on first import.

**Invalidation:** Hash of `_derive_symbolic`'s source code + `sympy.__version__`
(via `inspect.getsource`). Stored inside the pickle. On load, recompute hash and
compare. Mismatch → re-derive (135s), re-save. Match → load + lambdify (70ms).

Note: formatting changes or comment edits in `_derive_symbolic` will invalidate
the cache (false positive). This is acceptable — it's a one-time 135s cost after
editing the derivation, and the alternative (not invalidating) risks silent bugs.

**Atomic writes:** Write to a temp file, then `os.replace` (atomic on POSIX).
Prevents corrupt pickle if two processes import simultaneously.

**Write failures:** Catch `OSError` on write and skip silently. For read-only installs,
the cache simply won't be written — derive every time. Slow but functional.

**Second safety net:** `_build_packer` inspects the lambdified function's signature
and raises `KeyError` if any parameter doesn't map to a struct field. A stale cache
with wrong symbol names fails loudly at first call.

```python
def _cache_key():
    source = inspect.getsource(_derive_symbolic)
    return hashlib.sha256((source + sp.__version__).encode()).hexdigest()[:16]

def _load_or_derive():
    cache_path = Path(__file__).parent / "data" / "eom_cache.pkl"
    if cache_path.exists():
        try:
            cached = pickle.loads(cache_path.read_bytes())
            if cached.get("key") == _cache_key():
                return cached["M"], cached["F"], cached["qdd"], cached["args"]
        except Exception:
            pass  # corrupt or incompatible pickle — re-derive
    # Miss or stale — re-derive (slow, ~2 min)
    import warnings
    warnings.warn(
        "EOM cache miss — running symbolic derivation (~2 min). "
        "This happens once after code or sympy version changes.",
        stacklevel=2,
    )
    M, F, args = _derive_symbolic()
    qdd = tuple(M.LUsolve(F)[i] for i in range(4))
    data = {"key": _cache_key(), "M": M, "F": F, "qdd": qdd, "args": args}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(data))
        os.replace(tmp, cache_path)
    except OSError:
        pass  # read-only install — skip cache write
    return M, F, qdd, args
```

## Lambdify output shapes

`sp.lambdify(args, tuple_of_exprs, modules='numpy', cse=True)` returns a
**tuple of values**, not an array. The wrappers must convert:

- **`_qdd_func`:** raw returns `(val0, val1, val2, val3)` where each is scalar
  or `(N,)`. Wrapper does `np.array(qdd_raw(...))` for scalar (→ (4,)) or
  `np.column_stack(qdd_raw(...))` for batch (→ (N,4)).

- **`M_func`:** raw returns tuple of 10 upper-triangle elements. Wrapper
  assembles into (4,4) or (N,4,4) — same logic as current code.

- **`F_func`:** raw returns tuple of 4 elements. Wrapper does same as `_qdd_func`.

Note: `sp.lambdify(args, sp.Matrix(...), ...)` returns an ndarray directly, but
we lambdify tuples of scalar expressions (for CSE compatibility), so we get tuples.

## NaN/inf handling

**Mass matrix M is positive definite** by construction (it's the kinetic energy
matrix of a mechanical system — T > 0 for any nonzero velocity). The symbolic
inverse `M.LUsolve(F)` is well-defined. In practice, extreme parameter values
(e.g. near-zero masses) or numerical overflow can produce non-finite qdd.

- **`_rhs_batch`:** guard qdd output with `~np.isfinite(qdd).all(axis=1)`,
  replace bad rows with zeros. Same end result as current M/F guard.
- **`rhs` (scalar):** no explicit guard. `solve_ivp` handles non-finite RHS
  values gracefully (reduces step size or reports failure). This matches the
  current behavior — `np.linalg.solve` on a well-conditioned M doesn't fail
  either; only extreme numerics produce inf/nan, which solve_ivp catches.

## What stays

- **`DrifterPhysics`, `EOMState`, `_build_packer`** — unchanged from DW-A.
- **`_derive_symbolic`** — unchanged. Returns M_static, F_static, args.
- **`M_func`, `F_func`** — remain as public API for tests and inspection (mass matrix
  positive definiteness, symmetry checks, the example notebook). But they are no longer
  on the hot path. Their implementation simplifies: just `lambdify(cse=True)` + reshape.
- **`_uv_to_theta`, `_uv_to_spherical`, `_spherical_to_uv`** — unchanged.

## What goes

- **`_apply_cse_and_lambdify`** (70 lines) — replaced by `sp.lambdify(..., cse=True)`.
- **`_save_eom_cache`** — replaced by `_load_or_derive` with pickle.
- **`cli.py`** — delete. The `save-eom-cache` CLI command is no longer needed
  (cache is auto-generated). Remove the `[project.scripts]` entry from `pyproject.toml`
  and the `click` dependency.
- **`.srepr` file and all parsing code** — replaced by pickle.
- **`NumPyPrinter` import** — no longer needed.
- **`_eval_M_F`** in `drifter.py` — replaced by direct `_qdd_func` call (completes DW-D).
- **`np.linalg.solve` in `rhs` and `_rhs_batch`** — replaced by direct `_qdd_func`.
- **Matrix assembly in `_rhs_batch`** — no (N,4,4) construction, no batch solve.

## Changes

### 1. `lagrange_model.py`

**Add:**
- `import pickle, hashlib, os`
- `_cache_key()` — hash of `_derive_symbolic` source + `sp.__version__`.
- `_load_or_derive()` — pickle-based cache with hash validation, atomic writes,
  OSError handling. Returns `(M_static, F_static, qdd_exprs, args)`.
- `_qdd_func(physics, state)` — internal wrapper (not exported). Calls
  `qdd_raw(*pack(physics, state))`, converts tuple to (4,) or (N,4) array.

**Simplify:**
- `_get_eom_callables()` — calls `_load_or_derive()`, then:
  - `sp.lambdify(args, qdd_exprs, modules='numpy', cse=True)` for qdd
  - `sp.lambdify(args, m_exprs, modules='numpy', cse=True)` for M (public API)
  - `sp.lambdify(args, f_exprs, modules='numpy', cse=True)` for F (public API)
  - `_build_packer` on any of the above (same signature)
  Returns `(qdd_raw, M_raw, F_raw, args, pack_eom_args)`. All cached by `lru_cache`.
- `M_func`, `F_func` — unchanged signatures `(physics, state)`. Internal implementation
  switches from exec'd code to `lambdify(cse=True)`. Reshape logic stays for M (tuple
  of 10 → (4,4) or (N,4,4)). Simplifies for F (tuple of 4 → (4,) or (N,4)).

**Delete:**
- `_apply_cse_and_lambdify` (entire function, 70 lines).
- `_save_eom_cache` (entire function).
- `_SREPR_PATH` and all `.srepr` loading/parsing code.
- `NumPyPrinter` import.

### 2. `drifter.py`

**Simplify `rhs`:**
```python
def rhs(self, t, y):
    # unpack state, sample currents (unchanged)
    ...
    state = EOMState(u, v, xd, yd, ud, vd, U_b, V_b, U_d, V_d)
    qdd = _qdd_func(self.physics, state)  # returns (4,)
    return np.array([xd, yd, ud, vd, *qdd])
```

**Simplify `_rhs_batch`:**
```python
def _rhs_batch(self, Y, sample_uv):
    # unpack, sample currents (unchanged)
    ...
    state = EOMState(u, v, xd, yd, ud, vd, U_b, V_b, U_d, V_d)
    qdd = _qdd_func(self.physics, state)  # returns (N, 4)

    # Guard NaN/inf
    bad = ~np.isfinite(qdd).all(axis=1)
    if np.any(bad):
        qdd[bad] = 0.0

    dY = np.empty_like(Y)
    dY[:, IX] = xd
    dY[:, IY] = yd
    dY[:, IU] = ud
    dY[:, IV] = vd
    dY[:, IXD:] = qdd
    return dY
```

**Delete:**
- `_eval_M_F` (completes DW-D).
- Imports of `M_func`, `F_func` from drifter.py (no longer needed internally).
- Import `_qdd_func` from `lagrange_model` instead.

### 3. `__init__.py`

Keep `M_func`, `F_func`, `DrifterPhysics`. Do NOT export `_qdd_func` (it takes
`EOMState` which uses internal stereographic coordinates).

### 4. `cli.py` and `pyproject.toml`

- Delete `src/drogued_drifters/cli.py`.
- Remove `save-eom-cache = "drogued_drifters.cli:save_eom_cache"` from
  `[project.scripts]` in `pyproject.toml`.
- Remove `click` from `dependencies` in `pyproject.toml` (if no other code uses it).

### 5. File cleanup

- Delete `src/drogued_drifters/data/symbolic_eom.srepr`.
- Commit the new `src/drogued_drifters/data/eom_cache.pkl` (generated by running
  any test or importing the package).

### 6. Tests

**Files that need updating:**
- `tests/test_lagrange_physics.py` — `test_packer_covers_all_struct_fields` and
  `test_packer_arg_order_matches_lambda` call `_get_eom_callables()` and unpack
  its return value. Update for new return signature.
- `tests/test_lagrange_symbolic_fallback.py` — cache fallback tests, references to
  `_apply_cse_and_lambdify`. Rewrite for pickle-based caching.
- `tests/test_drogued_drifter.py` — tests calling `_eval_M_F`. Update to test
  through `rhs` or `_qdd_func`.
- `tests/test_numerical_edge_cases.py` — may reference old internals.

**Existing tests preserved:**
- Tests that call `M_func`/`F_func` directly: unchanged (public API preserved).

**New tests:**
- `test_qdd_func_matches_M_F_solve` — verifies `_qdd_func` agrees with
  `M_func + F_func + np.linalg.solve` at several test points.

**Broadcasting contract tests** (guards against sympy codegen changes):

We depend on an implicit contract: `sp.lambdify(..., modules='numpy', cse=True)`
produces code that numpy-broadcasts correctly. If sympy's code printer changes
(e.g. adds `float()` casts, uses non-broadcasting functions, changes CSE variable
handling), our batch path breaks silently. These tests pin that contract:

- `test_lambdify_scalar_input` — scalar args in, scalar results out. Baseline.
- `test_lambdify_batch_input` — (N,) arrays in, each output element is (N,) array.
  Verifies broadcasting works at all.
- `test_lambdify_batch_matches_scalar_loop` — for N test points, compare batch
  result against scalar-per-point loop. Catches broadcasting bugs that produce
  wrong values (not just wrong shapes).
- `test_lambdify_mixed_scalar_array_broadcast` — physics args are scalars,
  state args are (N,) arrays. This is the actual calling pattern: DrifterPhysics
  is scalar, EOMState holds batch arrays. Verifies scalar-array mixed broadcasting.
- `test_lambdify_cse_preserves_broadcasting` — compare `lambdify(cse=True)` vs
  `lambdify(cse=False)` on batch input. If CSE breaks broadcasting, this catches it.
- `test_lambdify_batch_N1` — single-element (N=1) arrays. Common source of shape
  bugs from numpy dimension squeezing.

These tests run on the actual model expressions (`_qdd_func`, `M_func`, `F_func`),
not on toy examples. They are the canary for sympy/numpy version upgrades.

### 7. Plan updates

Tick off DW-C and DW-D in `plans/code-review-remarks.md`.

## Order of operations

1. Add `_cache_key`, `_load_or_derive`, `_qdd_func` to `lagrange_model.py`.
2. Replace `_apply_cse_and_lambdify` and `.srepr` code with `lambdify(cse=True)` + pickle.
3. Update `_get_eom_callables` return signature.
4. Update `drifter.py`: simplify `rhs` and `_rhs_batch`, delete `_eval_M_F`.
5. Do NOT update `__init__.py` exports (keep M_func, F_func, DrifterPhysics).
6. Delete `cli.py`, update `pyproject.toml`.
7. Delete `.srepr` file, generate and commit `.pkl`.
8. Update all test files.
9. Run full suite.
10. Tick off DW-C and DW-D.
