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
   by a hash of `_derive_symbolic`'s source code. Stale cache auto-invalidates.

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
_get_eom_callables()        → qdd_func, pack_eom_args      (lambdify + _build_packer)
       |                      + M_func_raw, F_func_raw      (for public API / tests)
       v
rhs() / _rhs_batch()        → qdd = qdd_func(*pack(...))   (one call, no solve)
```

## Caching strategy

**What:** Pickle `(M_static, F_static, qdd_exprs, args)` — symbolic expressions
only, not lambdified callables (those can't be pickled).

**Where:** `data/eom_cache.pkl` next to the package (same location as current `.srepr`).

**Invalidation:** Hash of `_derive_symbolic`'s source code (via `inspect.getsource`).
Stored inside the pickle alongside the expressions. On load, recompute hash and compare.
Mismatch → re-derive (135s), re-save. Match → load + lambdify (70ms).

**Second safety net:** `_build_packer` inspects the lambdified function's signature
and raises `KeyError` if any parameter doesn't map to a struct field. A stale cache
with wrong symbol names fails loudly at first call.

```python
def _cache_key():
    return hashlib.sha256(inspect.getsource(_derive_symbolic).encode()).hexdigest()[:16]

def _load_or_derive():
    cache_path = Path(__file__).parent / "data" / "eom_cache.pkl"
    if cache_path.exists():
        cached = pickle.loads(cache_path.read_bytes())
        if cached.get("key") == _cache_key():
            return cached["M"], cached["F"], cached["qdd"], cached["args"]
    # Miss or stale — re-derive
    M, F, args = _derive_symbolic()
    qdd = tuple(M.LUsolve(F)[i] for i in range(4))
    data = {"key": _cache_key(), "M": M, "F": F, "qdd": qdd, "args": args}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(data))
    return M, F, qdd, args
```

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
- **`.srepr` file and all parsing code** — replaced by pickle.
- **`NumPyPrinter` import** — no longer needed.
- **`_eval_M_F`** in `drifter.py` — replaced by direct `qdd_func` call (completes DW-D).
- **`np.linalg.solve` in `rhs` and `_rhs_batch`** — replaced by direct `qdd_func`.
- **Matrix assembly in `_rhs_batch`** — no (N,4,4) construction, no batch solve.
- **NaN/inf guard** in `_rhs_batch` — moves to guard qdd output directly.

## Changes

### 1. `lagrange_model.py`

**Add:**
- `_cache_key()` — hash of `_derive_symbolic` source.
- `_load_or_derive()` — pickle-based cache with hash validation. Returns
  `(M_static, F_static, qdd_exprs, args)`.
- `qdd_func(physics, state)` — public wrapper. Returns (4,) or (N,4) array.
  The new hot-path entry point.

**Simplify:**
- `_get_eom_callables()` — calls `_load_or_derive()`, then:
  - `sp.lambdify(args, qdd_exprs, modules='numpy', cse=True)` for qdd
  - `sp.lambdify(args, m_exprs, modules='numpy', cse=True)` for M (public API)
  - `sp.lambdify(args, f_exprs, modules='numpy', cse=True)` for F (public API)
  - `_build_packer` on any of the above (same signature)
  All cached by `lru_cache`.
- `M_func`, `F_func` — unchanged signatures `(physics, state)`. Internal implementation
  switches from exec'd code to `lambdify(cse=True)`. Still reshape outputs.

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
    qdd = qdd_func(self.physics, state)
    return np.array([xd, yd, ud, vd, *qdd])
```

**Simplify `_rhs_batch`:**
```python
def _rhs_batch(self, Y, sample_uv):
    # unpack, sample currents (unchanged)
    ...
    state = EOMState(u, v, xd, yd, ud, vd, U_b, V_b, U_d, V_d)
    qdd = qdd_func(self.physics, state)  # returns (N, 4)

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

### 3. `__init__.py`

Add `qdd_func` to exports. Keep `M_func`, `F_func`, `DrifterPhysics`.

### 4. File cleanup

- Delete `src/drogued_drifters/data/symbolic_eom.srepr`.
- The new `eom_cache.pkl` is generated on first run (not checked into git).
  Add `src/drogued_drifters/data/eom_cache.pkl` to `.gitignore`.

### 5. Tests

- Tests that call `M_func`/`F_func` directly: unchanged (public API preserved).
- Tests that call `_eval_M_F`: update to use `qdd_func` or test through `rhs`.
- Tests that reference `_apply_cse_and_lambdify`: update to use new internals.
- Add: `test_qdd_func_matches_M_F_solve` — verifies qdd_func agrees with
  M_func + F_func + np.linalg.solve at several test points.
- Update cache fallback tests to use pickle-based caching.

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
  `lambdify(cse=False)` on batch input. If CSE breaks broadcasting (e.g. by
  introducing intermediate scalars), this catches it.

These tests run on the actual model expressions (qdd_func, M_func, F_func), not
on toy examples. They are the canary for sympy/numpy version upgrades.

### 6. Plan updates

Tick off DW-C and DW-D in `plans/code-review-remarks.md`.

## Order of operations

1. Add `_cache_key`, `_load_or_derive`, `qdd_func` to `lagrange_model.py`.
2. Replace `_apply_cse_and_lambdify` and `.srepr` code with `lambdify(cse=True)` + pickle.
3. Update `drifter.py`: simplify `rhs` and `_rhs_batch`, delete `_eval_M_F`.
4. Update `__init__.py` exports.
5. Delete `.srepr` file, add `.pkl` to `.gitignore`.
6. Update tests.
7. Run full suite.
8. Tick off DW-C and DW-D.
