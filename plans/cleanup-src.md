# Source Code Cleanup Plan

## Executive Summary

The codebase has completed a refactoring from code-generation-based EOM computation (`generate-eom` → `_generated_eom.py`) to a serialize-once/lambdify-at-runtime approach using `.srepr` caching (documented in `plans/srepr_cse_implementation.md`). However, the old architecture is still partially in place.

**Status:** Implementation is ~95% complete. Core architecture has been updated (`lagrange_model.py` has `_load_or_derive()`, `M_func`/`F_func` use it), but cleanup tasks remain.

**No backward compatibility concerns** — the old `compute_M`/`compute_F` aliases in `__init__.py` are already internal redirects to `M_func`/`F_func`.

---

## File-by-File Analysis and Cleanup Actions

| File | Status | Action | Notes |
|------|--------|--------|-------|
| `__init__.py` | Good | Keep as-is | Already exports `M_func`/`F_func` via aliases. No old code left. |
| `lagrange_model.py` | Good | Minor cleanup only | Core refactor done: `_load_or_derive()`, `_apply_cse_and_lambdify()` implemented. `M_func`/`F_func` updated. Old `_derive_and_lambdify()` is now **dead code** — delete. |
| `drifter.py` | Good | Keep; already simplified | `_eval_M_F()` and `_rhs_batch()` correctly call `M_func`/`F_func` with shaping. |
| `cli.py` | Partial | Keep and update docstrings | Both `generate_eom` and `save_eom_cache` commands exist. Docstring says `generate_eom` is "deprecated" but function is still implemented. Either delete `generate_eom` or keep both for explicit transition period. Recommend: **Keep both** with clear deprecation notice. |
| `_generated_eom.py` | Obsolete | **Delete** | No code imports from it. The `.srepr` file is the canonical source; `_load_or_derive()` recomputes when needed. File is not used anywhere. |
| `stokes.py` | Clean | Keep as-is | Self-contained utility; no dependencies on old architecture. |
| `test_drogued_drifter.py` | Good | Update only docstrings | Tests already use `M_func`/`F_func` correctly. Two helper functions `_make_kwargs()` and `_positional_from_kwargs()` are dead code — remove. |
| `data/symbolic_eom.srepr` | Required | Ensure present | Must exist in repository. If missing, fallback derivation runs on first import (~1 min). Should be committed to git. |

---

## Detailed Cleanup Actions

### 1. **`lagrange_model.py`** — Delete Dead Code

**Location:** Lines 174–183

**Function to delete:** `_derive_and_lambdify()`

```python
@functools.lru_cache()
def _derive_and_lambdify():
    """Derive and lambdify M and F (cached).
    
    This runs the full sympy derivation once on first use.
    """
    M_sub, F_sub, args = _derive_symbolic()
    M_lbd = sp.lambdify(args, M_sub, modules="numpy")
    F_lbd = sp.lambdify(args, F_sub, modules="numpy")
    return M_lbd, F_lbd
```

**Why delete:**
- Not called anywhere in the codebase (verified by grep).
- Replaced by `_load_or_derive()` which applies CSE and is faster.
- Keeping it only adds maintenance burden.

---

### 2. **`_generated_eom.py`** — Delete Entire File

**Path:** `src/drogued_drifters/_generated_eom.py`

**Why delete:**
- File is never imported anywhere (verified by grep; only temp test files reference it).
- The `.srepr` file is the canonical serialized symbolic form.
- `_load_or_derive()` reconstructs and lambdifies on-the-fly.
- Deleting removes ~130 lines of generated code that clutters the repo.

**No code breakage:** `_generated_eom.py` is not in the public API; `compute_M`/`compute_F` are aliases in `__init__.py` that point to `M_func`/`F_func`.

---

### 3. **`cli.py`** — Clarify Status and Optionally Simplify

**Current state:**
- `generate_eom(check)` function exists but docstring says "deprecated".
- `save_eom_cache()` function exists.
- `pyproject.toml` registers both as CLI commands.

**Recommendation: Keep both, but clarify**

Update docstrings in `cli.py` (lines 1–8):

```python
"""CLI for drogued_drifters code generation and cache management.

Usage::

    pixi run save-eom-cache        # RECOMMENDED: save symbolic EOM to .srepr cache
    pixi run generate-eom          # DEPRECATED: generate _generated_eom.py (no longer used)
    pixi run generate-eom --check  # DEPRECATED: verify freshness
"""
```

**Why keep both:**
- Allows explicit deprecation period (existing code can still call `generate_eom`).
- `save_eom_cache` is the forward path.
- No harm in keeping the old command; it's harmless if not used.

**Alternative (aggressive):** Delete `generate_eom()` entirely if confident no external users depend on it.

---

### 4. **`test_drogued_drifter.py`** — Remove Dead Helper Functions

**Helper 1: `_make_kwargs()` (lines 294–301)**
```python
def _make_kwargs(u, v, xd, yd, ud, vd, U_b, V_b, U_d, V_d):
    """Build the kwargs dict for both lambdified and generated functions."""
    return dict(...)
```

**Helper 2: `_positional_from_kwargs()` (lines 304–311)**
```python
def _positional_from_kwargs(kw):
    """Convert kwargs to positional args for the generated functions."""
    return (...)
```

**Why delete:**
- `_positional_from_kwargs()` is defined but never called (verified by grep).
- `_make_kwargs()` is used in `test_generated_vs_lambdified()` and `test_generated_vectorized()`, but those test functions could build kwargs inline.
- Removing reduces cognitive load; the inlined dicts are clearer anyway.

**Action:** Inline the `_make_kwargs()` call in the two tests that use it and delete both helpers.

---

### 5. **`__init__.py`** — No Changes Needed

Current state:
```python
from .drifter import DroguedDrifter
from .lagrange_model import M_func, F_func

# For backward compatibility, alias to the new API
compute_M = M_func
compute_F = F_func
```

**Status:** Already correct. No action needed. These aliases ensure old code calling `compute_M` still works.

---

### 6. **`pyproject.toml`** — Optional Cleanup

**Current:**
```toml
[project.scripts]
generate-eom = "drogued_drifters.cli:generate_eom"
save-eom-cache = "drogued_drifters.cli:save_eom_cache"
```

**Option A (recommended):** Keep both, but add comment:

```toml
[project.scripts]
save-eom-cache = "drogued_drifters.cli:save_eom_cache"  # Primary: saves symbolic EOM cache
generate-eom = "drogued_drifters.cli:generate_eom"      # Deprecated: kept for backward compat
```

**Option B (aggressive):** Delete `generate-eom` entry once confident no external dependents.

---

## Summary Table: Deletions and Changes

| Item | Action | Files Affected | Scope |
|------|--------|----------------|-------|
| `_derive_and_lambdify()` | Delete | `lagrange_model.py:174–183` | 10 lines |
| `_generated_eom.py` | Delete | `src/drogued_drifters/_generated_eom.py` | Entire file (~130 lines) |
| `_make_kwargs()` | Delete | `test_drogued_drifter.py:294–301` | 8 lines; inline in 2 tests |
| `_positional_from_kwargs()` | Delete | `test_drogued_drifter.py:304–311` | 8 lines; dead code |
| CLI docstrings | Update | `cli.py:1–8` | 3 lines; clarify status |

---

## Implementation Order

1. **Delete `_derived_and_lambdify()`** from `lagrange_model.py`
   - Verify no imports: `grep -r "_derive_and_lambdify" src tests`
   - Delete the function and its docstring.

2. **Update `test_drogued_drifter.py`**
   - Remove `_make_kwargs()` and `_positional_from_kwargs()` helper functions.
   - Inline the `_make_kwargs()` dict into the two tests that use it:
     - `test_generated_vs_lambdified()` (line 376)
     - `test_generated_vectorized()` (line 402)
   - Run tests to confirm they pass.

3. **Delete `_generated_eom.py`**
   - Verify no imports: `grep -r "_generated_eom" src tests`
   - Delete the file.
   - Run tests to confirm they pass.

4. **Update CLI docstring** in `cli.py` (optional but recommended)
   - Clarify that `generate_eom` is deprecated and kept for backward compat.
   - Recommend `save_eom_cache` as the primary cache generation command.

5. **Update `pyproject.toml`** (optional)
   - Add inline comments indicating deprecation status, or
   - Delete `generate-eom` entry if no external dependents.

6. **Verify `.srepr` file exists**
   - Path: `src/drogued_drifters/data/symbolic_eom.srepr`
   - If missing, run `pixi run save-eom-cache` to generate it.
   - Commit to git.

7. **Run full test suite**
   ```bash
   pixi run pytest tests/ -xvs
   ```
   - Confirm all tests pass after deletions.

---

## Verification Checklist

After cleanup, verify:

- [ ] No code imports from `_generated_eom.py`
- [ ] No code calls `_derive_and_lambdify()`
- [ ] No code calls `_make_kwargs()` or `_positional_from_kwargs()`
- [ ] `src/drogued_drifters/data/symbolic_eom.srepr` exists
- [ ] All pytest tests pass: `pixi run pytest tests/ -v`
- [ ] `drogued_drifters` package imports without errors: `python -c "import drogued_drifters; print('OK')"`
- [ ] `DroguedDrifter` can be instantiated: `python -c "from drogued_drifters import DroguedDrifter; d = DroguedDrifter(); print('OK')"`

---

## Impact Summary

**Lines of code removed:** ~170 lines (function + entire file + test helpers)
**Lines of code changed:** ~10 lines (docstrings, inline dicts)
**Files deleted:** 1 (`_generated_eom.py`)
**Files modified:** 3 (`lagrange_model.py`, `test_drogued_drifter.py`, `cli.py`)
**Risk level:** Very low (dead code only; no public API changes)
**Testing:** All existing tests should pass without changes

---

## Notes

- The refactoring to `.srepr` + `_load_or_derive()` is already complete and working. This cleanup removes the leftover dead code.
- No external package depends on `_generated_eom.py` (it's an internal module).
- The `compute_M`/`compute_F` backward-compat aliases in `__init__.py` remain untouched and will continue to work via `M_func`/`F_func`.
- If `.srepr` file is missing from the repository, `_load_or_derive()` has a fallback to re-derive symbolically (slow but safe). The file should be committed for production use.
