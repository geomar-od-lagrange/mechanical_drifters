# Implementation Plan: Replace Code Generation with Serialize-Once / Lambdify-at-Runtime

## Overview

Replace the current code-generation architecture (`cli.py` → `_generated_eom.py`) with a serialize-once/lambdify-at-runtime approach:

1. **Run symbolic derivation once** → serialize M_sub, F_sub via `sp.srepr()` → commit as `.srepr` text file
2. **At import time**: recover expressions via `sp.sympify()` → apply `sp.cse()` → build+exec Python function string (Approach B) → cache lambdified functions
3. **Public interface** (`M_func`/`F_func`) wraps raw lambdified callables, handles scalar vs batch input, returns shaped arrays
4. **Drop manual assembly** in `drifter.py`: call `M_func`/`F_func`, receive shaped results

Benefits:
- No more static generated code in git (cleaner repo)
- CSE optimization applied at runtime with full control
- Same performance as pre-generated code (Approach B exec pattern proven in `test_cse_lambdify.py`)
- Easier to test and modify without re-running code generation CLI
- Single source of truth: symbolic derivation cached in `.srepr`

---

## 1. Data File: `src/drogued_drifters/data/symbolic_eom.srepr`

### Format

Plain text file with three sections, separated by `---`:

```
<M_sub_srepr>
---
<F_sub_srepr>
---
<arg_names_csv>
```

Where:
- **M_sub_srepr**: Output of `sp.srepr(M_sub)` — complete S-expression of the 4×4 mass matrix
- **F_sub_srepr**: Output of `sp.srepr(F_sub)` — complete S-expression of the 4×1 force vector
- **arg_names_csv**: Comma-separated list of argument names in order: `u_s,v_s,xd_s,yd_s,ud_s,vd_s,m_b,m_d,m_hat_d,m_tilde_d,m_tilde_b,l,g,k_b,k_d,U_b,V_b,U_d,V_d`

### Why `.srepr`?

- **Roundtrip-safe**: `sp.sympify(srepr_string)` perfectly recovers the expression
- **Cheap**: No serialization overhead compared to pickle or custom formats
- **Readable**: Pure Python S-expression syntax
- **Git-friendly**: Text file, diffs work, no binary data

### Generation

A new CLI command or fallback in `_load_or_derive()` will create this file by running `_derive_symbolic()` once and writing:

```python
def _save_srepr_file(path):
    """Write M_sub, F_sub, args to path in srepr format."""
    M_sub, F_sub, args = _derive_symbolic()
    arg_names = ','.join(str(s) for s in args)
    content = f"{sp.srepr(M_sub)}\n---\n{sp.srepr(F_sub)}\n---\n{arg_names}\n"
    Path(path).write_text(content)
```

---

## 2. New Function in `lagrange_model.py`: `_load_or_derive()`

### Purpose

Central cached function that:
1. Tries to load `.srepr` file from `data/symbolic_eom.srepr`
2. Falls back to `_derive_symbolic()` if missing
3. Applies CSE + Approach B exec to produce raw lambdified callables `_raw_M_func` and `_raw_F_func`
4. Caches result with `@functools.lru_cache()`

### Implementation Sketch

```python
import functools
from pathlib import Path

_SREPR_PATH = Path(__file__).resolve().parent / "data" / "symbolic_eom.srepr"

@functools.lru_cache()
def _load_or_derive():
    """Load or derive M_sub, F_sub, args; apply CSE; return raw lambdified callables.
    
    Returns:
        (_raw_M_func, _raw_F_func, arg_symbols)
        where _raw_M_func/F_func are lambdified callables that accept positional args
        and return results (potentially with batch dimension last).
    """
    # Try to load .srepr file
    if _SREPR_PATH.exists():
        content = _SREPR_PATH.read_text().strip()
        parts = content.split("---")
        if len(parts) == 3:
            M_srepr, F_srepr, arg_names_csv = parts
            M_sub = sp.sympify(M_srepr)
            F_sub = sp.sympify(F_srepr)
            arg_names = arg_names_csv.split(",")
            arg_symbols = tuple(sp.Symbol(name, real=True) if '(' not in name 
                               else sp.sympify(name) for name in arg_names)
        else:
            raise ValueError(f"Invalid .srepr format in {_SREPR_PATH}")
    else:
        # Fallback: derive symbolically
        M_sub, F_sub, arg_symbols = _derive_symbolic()
    
    # Apply CSE and build raw lambdified functions via Approach B
    return _apply_cse_and_lambdify(M_sub, F_sub, arg_symbols)


def _apply_cse_and_lambdify(M_sub, F_sub, args):
    """Apply CSE to M and F; build raw lambdified functions via exec.
    
    This is Approach B from test_cse_lambdify.py:
    Generate Python function string, exec it, return the functions.
    
    Args:
        M_sub: 4x4 sympy matrix
        F_sub: 4x1 sympy matrix
        args: tuple of sympy symbols in lambdify order
    
    Returns:
        (_raw_M_func, _raw_F_func, args)
        where _raw_M_func(*args) returns M elements as tuple
        and _raw_F_func(*args) returns F elements as tuple
        (both vectorized over numpy arrays; batch dim goes last).
    """
    # Extract elements for CSE
    m_exprs = []
    m_labels = []
    for i in range(4):
        for j in range(i, 4):  # Upper triangle only (symmetric)
            m_exprs.append(M_sub[i, j])
            m_labels.append((i, j))
    
    f_exprs = [F_sub[i] for i in range(4)]
    
    all_exprs = m_exprs + f_exprs
    
    # Apply CSE
    replacements, reduced = sp.cse(all_exprs, optimizations="basic")
    
    # Generate Python code using NumPyPrinter
    from sympy.printing.numpy import NumPyPrinter
    printer = NumPyPrinter()
    arg_names = [str(s) for s in args]
    
    # Build compute_M source
    lines_M = [f"def _raw_M({', '.join(arg_names)}):"]
    for sym, expr in replacements:
        lines_M.append(f"    {sym} = {printer.doprint(expr)}")
    for expr, (i, j) in zip(reduced[:len(m_exprs)], m_labels):
        lines_M.append(f"    M_{i}{j} = {printer.doprint(expr)}")
    ret_names_M = ", ".join(f"M_{i}{j}" for i, j in m_labels)
    lines_M.append(f"    return {ret_names_M}")
    
    # Build compute_F source
    lines_F = [f"def _raw_F({', '.join(arg_names)}):"]
    for sym, expr in replacements:
        lines_F.append(f"    {sym} = {printer.doprint(expr)}")
    for idx, expr in enumerate(reduced[len(m_exprs):]):
        lines_F.append(f"    F_{idx} = {printer.doprint(expr)}")
    ret_names_F = ", ".join(f"F_{idx}" for idx in range(len(f_exprs)))
    lines_F.append(f"    return {ret_names_F}")
    
    # Combine source
    source = "\n".join(lines_M) + "\n\n" + "\n".join(lines_F)
    source = source.replace("numpy.", "np.")
    
    # Exec to create functions
    local_ns = {"np": np}
    exec(source, local_ns)
    
    _raw_M = local_ns["_raw_M"]
    _raw_F = local_ns["_raw_F"]
    
    return _raw_M, _raw_F, args
```

### Key Points

- **CSE applied at import time**, not pre-generated, for full control
- **Exec pattern proven** in `tmp_sp_lambdify_broadcast/test_cse_lambdify.py` (Approach B)
- **Same performance** as pre-generated code (benchmark shows CSE exec ≈ compiled code)
- **Cached**: LRU cache ensures this runs only once per Python session
- **Fallback**: If `.srepr` file missing, derive symbolically (safe for dev/testing)

---

## 3. Updated `M_func` / `F_func` in `lagrange_model.py`

### Purpose

These become the **sole public interface**. They:
1. Call `_load_or_derive()` to get raw lambdified callables
2. Detect scalar vs batch input (check `ndim` of first dynamic arg)
3. Call raw callable
4. **Reshape output** to match input shape:
   - Scalar input → (4,4) or (4,) output
   - Batch input (N,) → (N,4,4) or (N,4) output

### Implementation Sketch

```python
def M_func(
    *,
    u, v, xd, yd, ud, vd,
    m_b, m_d, m_hat_d, m_tilde_d, m_tilde_b,
    l, g, k_b, k_d, U_b, V_b, U_d, V_d,
):
    """Numerically evaluate the mass matrix M in stereographic coordinates.
    
    Wraps the raw lambdified callable from _load_or_derive().
    Detects scalar vs batch input and returns shaped output.
    
    Args:
        u, v: Stereographic coordinates (scalar or (N,) array).
        xd, yd, ud, vd: Velocities (scalar or (N,) array).
        m_b, m_d, ..., V_d: Physical parameters (scalar or (N,) array).
    
    Returns:
        4x4 mass matrix:
        - Scalar input: (4,4) array
        - Batch input: (N,4,4) array
    """
    _raw_M, _, arg_symbols = _load_or_derive()
    params = LagrangeParams(
        u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd,
        m_b=m_b, m_d=m_d, m_hat_d=m_hat_d,
        m_tilde_d=m_tilde_d, m_tilde_b=m_tilde_b,
        l=l, g=g, k_b=k_b, k_d=k_d,
        U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
    )
    
    # Detect batch size from first dynamic argument
    u_arr = np.asarray(u)
    batch_ndim = u_arr.ndim
    
    # Call raw function with positional args
    M_elems = _raw_M(*params)
    
    if batch_ndim == 0:
        # Scalar: assemble (4,4)
        M00, M01, M02, M03, M11, M12, M13, M22, M23, M33 = M_elems
        M = np.array([
            [M00, M01, M02, M03],
            [M01, M11, M12, M13],
            [M02, M12, M22, M23],
            [M03, M13, M23, M33],
        ], dtype=float)
    else:
        # Batch: assemble (N, 4, 4)
        N = u_arr.shape[0]
        M = np.zeros((N, 4, 4))
        labels = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        for k, (i, j) in enumerate(labels):
            M[:, i, j] = M[:, j, i] = np.broadcast_to(M_elems[k], N)
    
    return M


def F_func(
    *,
    u, v, xd, yd, ud, vd,
    m_b, m_d, m_hat_d, m_tilde_d, m_tilde_b,
    l, g, k_b, k_d, U_b, V_b, U_d, V_d,
):
    """Numerically evaluate the force vector F in stereographic coordinates.
    
    Wraps the raw lambdified callable from _load_or_derive().
    Detects scalar vs batch input and returns shaped output.
    
    Args:
        Same as M_func.
    
    Returns:
        4-element force vector:
        - Scalar input: (4,) array
        - Batch input: (N,4) array
    """
    _, _raw_F, arg_symbols = _load_or_derive()
    params = LagrangeParams(
        u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd,
        m_b=m_b, m_d=m_d, m_hat_d=m_hat_d,
        m_tilde_d=m_tilde_d, m_tilde_b=m_tilde_b,
        l=l, g=g, k_b=k_b, k_d=k_d,
        U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
    )
    
    # Detect batch size
    u_arr = np.asarray(u)
    batch_ndim = u_arr.ndim
    
    # Call raw function
    F_elems = _raw_F(*params)
    
    if batch_ndim == 0:
        # Scalar: (4,)
        F = np.array(F_elems, dtype=float)
    else:
        # Batch: (N, 4)
        N = u_arr.shape[0]
        F = np.column_stack([np.broadcast_to(f, N) for f in F_elems])
    
    return F
```

### Key Changes from Current

- **No longer calls `_derive_and_lambdify()`** which did plain lambdify (slow)
- **Calls `_load_or_derive()`** which does CSE + exec (fast)
- **Shapes output** to match input: scalar → (4,4)/(4,), batch → (N,4,4)/(N,4)
- **Still accepts all kwargs** but internally calls raw function with positional args

---

## 4. Simplified `drifter.py`: `_eval_M_F` and `_rhs_batch`

### Changes

#### `_eval_M_F` (scalar path)

```python
def _eval_M_F(self, t, x, y, u, v, xd, yd, ud, vd, currents):
    """Evaluate mass matrix and force vector numerically (scalar)."""
    U_b, V_b, U_d, V_d = currents
    p = self._params()
    
    # Call M_func/F_func directly — they handle shaping
    M = M_func(
        u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd,
        **p, U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
    )  # Returns (4,4)
    
    F = F_func(
        u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd,
        **p, U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
    )  # Returns (4,)
    
    return M, F
```

**Why simpler?** No more unpacking tuple of 10 elements and manually assembling symmetric matrix. `M_func` and `F_func` handle it.

#### `_rhs_batch` (vectorized path)

```python
def _rhs_batch(self, Y, sample_uv):
    """Vectorized RHS for N particles."""
    N = Y.shape[0]
    u = Y[:, 2]
    v = Y[:, 3]
    xd = Y[:, 4]
    yd = Y[:, 5]
    ud = Y[:, 6]
    vd = Y[:, 7]
    
    # Sample velocity at buoy and drogue
    U_b, V_b = sample_uv(np.zeros(N))
    z_eff = self._z_eff_batch(u, v)
    U_d, V_d = sample_uv(z_eff)
    
    p = self._params()
    
    # Call M_func/F_func with batch arrays
    M = M_func(
        u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd,
        **p, U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
    )  # Returns (N, 4, 4)
    
    F = F_func(
        u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd,
        **p, U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
    )  # Returns (N, 4)
    
    # Handle NaN/inf (overflow in expressions)
    bad = ~np.isfinite(M).all(axis=(1, 2)) | ~np.isfinite(F).all(axis=1)
    if np.any(bad):
        M[bad] = np.eye(4)
        F[bad] = 0.0
    
    # Batched solve
    qdd = np.linalg.solve(M, F[:, :, np.newaxis])[:, :, 0]
    
    # Assemble dY
    dY = np.empty_like(Y)
    dY[:, 0] = xd
    dY[:, 1] = yd
    dY[:, 2] = ud
    dY[:, 3] = vd
    dY[:, 4:] = qdd
    
    return dY
```

**Why simpler?** 
- No more manually assembling (N, 4, 4) from upper-triangle tuple
- No more using `np.triu_indices` and loop
- Direct call to `M_func`/`F_func` with arrays, get shaped result back

---

## 5. `cli.py` Changes

### Option A: Keep CLI but simplify (recommended)

Replace the `generate-eom` command with a `save-eom-cache` command:

```python
@click.command()
def save_eom_cache():
    """Save the symbolic EOM to .srepr cache file.
    
    This pre-computes the symbolic derivation once and writes it to
    src/drogued_drifters/data/symbolic_eom.srepr for faster imports.
    """
    cache_path = Path(__file__).resolve().parent / "data" / "symbolic_eom.srepr"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    click.echo("Running sympy derivation (this may take a minute)...")
    M_sub, F_sub, args = _derive_symbolic()
    
    arg_names = ','.join(str(s) for s in args)
    content = f"{sp.srepr(M_sub)}\n---\n{sp.srepr(F_sub)}\n---\n{arg_names}\n"
    
    cache_path.write_text(content)
    click.echo(f"Wrote {cache_path}")
```

Update `pyproject.toml`:

```toml
[project.scripts]
save-eom-cache = "drogued_drifters.cli:save_eom_cache"
```

### Option B: Remove CLI entirely (alternative)

- Delete `cli.py` and the script entry from `pyproject.toml`
- Rely on `_load_or_derive()` fallback: if `.srepr` missing, derive on first import (slow but safe)
- For production, create `.srepr` offline and commit it

We recommend **Option A** for explicit control and clear messaging.

---

## 6. `_generated_eom.py` Deletion

This file is no longer needed. Delete:
```
src/drogued_drifters/_generated_eom.py
```

The raw lambdified functions are now generated on-the-fly in `_apply_cse_and_lambdify()` and cached.

---

## 7. `__init__.py` Update

Current behavior:
```python
try:
    from ._generated_eom import compute_F, compute_M
except ImportError:
    def _not_generated(*args, **kwargs):
        raise ImportError("The generated EOM module is missing...")
    compute_M = _not_generated
    compute_F = _not_generated
```

**New behavior** — remove the try/except, replace with:

```python
from .drifter import DroguedDrifter
from .lagrange_model import M_func, F_func

# For backward compatibility, alias to the new API
compute_M = M_func
compute_F = F_func
```

Or simply:

```python
from .drifter import DroguedDrifter
```

And let users import `M_func`/`F_func` directly from `lagrange_model` if needed.

---

## 8. Test Updates in `test_drogued_drifter.py`

### Tests That Need Changes

| Test | Change | Reason |
|------|--------|--------|
| `test_MF_callable` | Update to test `M_func`, `F_func` instead of `compute_M`, `compute_F` | New API |
| `test_MF_evaluates` | No change needed (calls `_eval_M_F` internally) | Works with new `M_func`/`F_func` |
| `test_generated_vs_lambdified` | **Update**: Call `M_func`/`F_func` directly, compare to... (see below) | New approach |
| `test_generated_vectorized` | **Update**: Call `M_func`/`F_func` with arrays directly | New approach |
| `test_generated_eom_freshness` | **Delete** | No more `.srepr` hash check needed |

### New/Refactored Tests

#### Verify `.srepr` → sympify → CSE roundtrip

```python
def test_srepr_sympify_roundtrip():
    """Verify that .srepr file loads correctly and produces same results as direct derivation."""
    from drogued_drifters.lagrange_model import _load_or_derive, _derive_symbolic
    
    # Direct derivation
    M_direct, F_direct, args_direct = _derive_symbolic()
    
    # Via _load_or_derive (uses .srepr if available)
    _raw_M, _raw_F, args_loaded = _load_or_derive()
    
    # Evaluate at a test point
    test_params = (0.1, 0.05, 0.0, 0.0, 0.0, 0.0,
                   1.0, 2.7, 1.0, 101.0, 1.9,
                   3.0, 9.81, 12.0, 154.0,
                   0.5, -0.3, 0.2, 0.1)
    
    M_raw_elems = _raw_M(*test_params)
    F_raw_elems = _raw_F(*test_params)
    
    # Verify against direct lambdify
    M_direct_lbd = sp.lambdify(args_direct, M_direct, modules="numpy")
    F_direct_lbd = sp.lambdify(args_direct, F_direct, modules="numpy")
    
    M_direct_result = M_direct_lbd(*test_params)
    F_direct_result = F_direct_lbd(*test_params)
    
    # Assemble raw results into matrices for comparison
    M00, M01, M02, M03, M11, M12, M13, M22, M23, M33 = M_raw_elems
    M_from_raw = np.array([
        [M00, M01, M02, M03],
        [M01, M11, M12, M13],
        [M02, M12, M22, M23],
        [M03, M13, M23, M33],
    ])
    F_from_raw = np.array(F_raw_elems)
    
    np.testing.assert_allclose(M_from_raw, M_direct_result, atol=1e-12)
    np.testing.assert_allclose(F_from_raw, F_direct_result, atol=1e-12)
```

#### Verify M_func / F_func shape handling

```python
def test_M_F_func_shapes():
    """Verify M_func and F_func return correct shapes."""
    from drogued_drifters.lagrange_model import M_func, F_func
    
    # Scalar input
    M_scalar = M_func(
        u=0.1, v=0.05, xd=0.0, yd=0.0, ud=0.0, vd=0.0,
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
    )
    assert M_scalar.shape == (4, 4), f"Expected (4,4), got {M_scalar.shape}"
    
    F_scalar = F_func(
        u=0.1, v=0.05, xd=0.0, yd=0.0, ud=0.0, vd=0.0,
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=0.5, V_b=-0.3, U_d=0.2, V_d=0.1,
    )
    assert F_scalar.shape == (4,), f"Expected (4,), got {F_scalar.shape}"
    
    # Batch input (N=5)
    N = 5
    u_batch = np.full(N, 0.1)
    v_batch = np.full(N, 0.05)
    
    M_batch = M_func(
        u=u_batch, v=v_batch, xd=np.zeros(N), yd=np.zeros(N),
        ud=np.zeros(N), vd=np.zeros(N),
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=np.full(N, 0.5), V_b=np.full(N, -0.3),
        U_d=np.full(N, 0.2), V_d=np.full(N, 0.1),
    )
    assert M_batch.shape == (N, 4, 4), f"Expected (N,4,4), got {M_batch.shape}"
    
    F_batch = F_func(
        u=u_batch, v=v_batch, xd=np.zeros(N), yd=np.zeros(N),
        ud=np.zeros(N), vd=np.zeros(N),
        m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
        l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        U_b=np.full(N, 0.5), V_b=np.full(N, -0.3),
        U_d=np.full(N, 0.2), V_d=np.full(N, 0.1),
    )
    assert F_batch.shape == (N, 4), f"Expected (N,4), got {F_batch.shape}"
```

### Tests to Delete/Replace

- `test_generated_eom_freshness`: Deletes (`_generated_eom.py` no longer exists)
- `test_generated_vs_lambdified`: Refactor to compare M_func/F_func (wrapped) vs direct lambdify for regression testing
- `test_generated_vectorized`: Refactor to test M_func/F_func with arrays

---

## 9. `pyproject.toml` Update

```toml
[project.scripts]
save-eom-cache = "drogued_drifters.cli:save_eom_cache"
```

(Replace `generate-eom` with `save-eom-cache` if keeping Option A CLI.)

If going with **Option B** (no CLI), remove the `[project.scripts]` section entirely.

---

## 10. Implementation Order (Keep Tests Passing)

### Phase 1: Core new infrastructure (no tests broken yet)

1. **Create directory**: `mkdir -p src/drogued_drifters/data`

2. **Add `_load_or_derive()` and `_apply_cse_and_lambdify()`** to `lagrange_model.py`
   - Keep existing `_derive_symbolic()`, `_derive_and_lambdify()` intact
   - New functions are independent additions
   - Tests still passing (nothing broken yet)

3. **Create `.srepr` file**:
   - Run `_derive_symbolic()` once
   - Write to `src/drogued_drifters/data/symbolic_eom.srepr`
   - Commit to git

### Phase 2: Update public interface (some tests will need updates)

4. **Update `M_func` and `F_func`** in `lagrange_model.py`:
   - Change from calling `_derive_and_lambdify()` to `_load_or_derive()`
   - Add shape detection and reshaping logic
   - These functions still work with old tests (same kwargs interface)

5. **Update `__init__.py`**:
   - Keep backward-compat aliases if needed (`compute_M = M_func`, etc.)
   - Or just export `M_func`, `F_func`, `DroguedDrifter`

### Phase 3: Simplify drifter code

6. **Update `drifter.py`**:
   - Simplify `_eval_M_F`: call `M_func`/`F_func` directly
   - Simplify `_rhs_batch`: call `M_func`/`F_func` directly
   - Remove manual assembly of symmetric matrix

### Phase 4: Update tests and CLI

7. **Update tests** in `test_drogued_drifter.py`:
   - Change `compute_M`/`compute_F` imports to `M_func`/`F_func`
   - Delete `test_generated_eom_freshness`
   - Add new roundtrip/shape tests

8. **Update or delete `cli.py`**:
   - Option A: Replace `generate-eom` → `save-eom-cache`
   - Option B: Delete entire `cli.py`

9. **Update `pyproject.toml`**:
   - Update script entry point or remove entirely

### Phase 5: Cleanup

10. **Delete `_generated_eom.py`** once all code references it removed

11. **Run full test suite**:
    ```bash
    pixi run pytest tests/ -v
    ```

12. **Test CLI** (if keeping):
    ```bash
    pixi run save-eom-cache  # Should complete in ~1 min
    ```

---

## Implementation Checklist

### New Files & Directories

- [ ] Create `src/drogued_drifters/data/` directory
- [ ] Create `src/drogued_drifters/data/symbolic_eom.srepr` (generated)

### Modified Files

- [ ] `lagrange_model.py`: Add `_load_or_derive()`, `_apply_cse_and_lambdify()`; update `M_func`/`F_func`
- [ ] `drifter.py`: Simplify `_eval_M_F()`, `_rhs_batch()`
- [ ] `__init__.py`: Update imports
- [ ] `cli.py`: Replace or delete
- [ ] `pyproject.toml`: Update script entry
- [ ] `test_drogued_drifter.py`: Update imports, delete/refactor tests

### Deleted Files

- [ ] `src/drogued_drifters/_generated_eom.py`

### Verification Steps

- [ ] All tests pass: `pixi run pytest`
- [ ] CLI works (if kept): `pixi run save-eom-cache`
- [ ] `.srepr` file is readable and roundtrips correctly
- [ ] Batch and scalar paths both work in `_rhs_batch` and `rhs`
- [ ] Performance is not degraded (CSE + exec should match pre-generated)

---

## Approach B Code Sketch (Proven Pattern)

The key innovation is generating a Python function string that mirrors the structure of `_generated_eom.py`, then executing it to create functions. This is proven to work in `test_cse_lambdify.py` Approach B.

### Why Approach B?

- **Approach A** (sequential lambdify chain of CSE temps): Each temporary is a separate lambdified function; overhead from calling multiple functions.
- **Approach B** (exec Python string): Single function with inlined CSE temporaries; **no function call overhead**, same performance as static generated code.
- **Baseline** (plain lambdify, no CSE): Slow; no subexpression optimization.

**Benchmark result from test_cse_lambdify.py**:
- Baseline: ~0.5 ms per call
- Approach B: ~0.08 ms per call (6x faster)
- Generated code: ~0.08 ms per call (same as B)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `.srepr` file too large | It's text; typically <1 MB for our 4x4 + 4x1 system. Acceptable to commit. |
| First import slow if `.srepr` missing | Fallback to `_derive_symbolic()` which takes ~1 min on first run. Acceptable for dev; production runs `save-eom-cache` once. |
| exec() security | Only executed with our own generated code (NumPyPrinter output is trusted). No user input. |
| Sympy version compatibility | `sp.srepr()` and `sp.sympify()` are stable APIs. If sympy version changes, `.srepr` file can be regenerated. |
| Lost CSE optimization | CSE is applied at import time, so every Python session gets full optimization. No performance loss vs pre-generated. |

---

## Benefits Summary

1. **No code generation step required** → simpler dev workflow
2. **Cleaner git history** → `.srepr` is a data file, not generated code
3. **Full CSE optimization** → applied at import time with Approach B exec
4. **Same performance** → Approach B proven to match compiled code
5. **Easier to test** → symbolic derivation is cached and testable
6. **Fallback path** → if `.srepr` missing, derive automatically (safe for development)

---

## Timeline Estimate

- **Phase 1** (core infra): 30 min
- **Phase 2** (update API): 20 min
- **Phase 3** (simplify drifter): 15 min
- **Phase 4** (tests + CLI): 30 min
- **Phase 5** (cleanup): 10 min
- **Testing & verification**: 30 min

**Total: ~2.5 hours**

---

## References

- **Proven Approach B pattern**: `tmp_sp_lambdify_broadcast/test_cse_lambdify.py` (lines 160–241)
- **Current generated code structure**: `src/drogued_drifters/_generated_eom.py`
- **Current CLI**: `src/drogued_drifters/cli.py`
- **Current usage in drifter**: `src/drogued_drifters/drifter.py` lines 341–462
