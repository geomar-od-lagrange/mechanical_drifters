# Test Coverage Gaps & Improvement Plan

## Current State (63% statement coverage)

| Module | Statement Coverage | Conceptual Assessment |
|---|---|---|
| `drifter.py` | 63% | Scalar/batch paths well-tested; **Parcels interpolator (lines 51–181) entirely untested** |
| `lagrange_model.py` | 70% | CSE lambdification and cache round-trip tested; **symbolic derivation (`_derive_symbolic`) never exercised** |
| `stokes.py` | **0%** | Completely untested |
| `cli.py` | **0%** | CLI wrapper untested (underlying function tested) |

### What IS well-tested

- Scalar and batch RHS evaluation (`rhs()`, `_eval_M_F()`, `_rhs_batch()`)
- `DroguedDrifter` instantiation with default and custom parameters
- Steady-state equilibrium physics (zero currents → zero drift)
- CSE-based lambdification (`_apply_cse_and_lambdify()`)
- Spherical ↔ stereographic coordinate conversions (round-trip)
- Cache round-trip (save/load .srepr files)

---

## Priority Gaps

### 1. stokes.py (0% → target 100%)

Test exponential depth decay against analytical solution for a single wave partition. Test multi-partition summation. Edge cases: zero peak period (undefined k), very long period (k→0, no decay), very large depth. Vectorization over (y, x) spatial dimensions. Surface boundary: z=0 returns surface drift unchanged.

### 2. Parcels interpolator (`make_dd_velocity_interpolator`, `make_profile_sampler`)

This is the critical bridge to real oceanographic simulations and has zero coverage (drifter.py lines 51–181). Test with synthetic (t, y, x, z) gridded data. Verify spherical mesh conversion (deg/s ↔ m/s). Test warm-starting across timesteps. Test particle deletion/reordering edge cases. Test `make_profile_sampler` boundary behavior near grid limits (`searchsorted` clipping).

### 3. Symbolic derivation fallback (`lagrange_model._derive_symbolic`)

Tests always use the `.srepr` cache, never exercise the ~130-line live derivation path (lines 38, 64–171). Force cache miss and exercise on-the-fly symbolic derivation. Compare cached vs freshly-derived M/F for numerical agreement.

**This test is slow (~30–60 s for sympy derivation + CSE) and must be marked accordingly** — see slow test infrastructure below.

### 4. Physics validation

- Mass matrix M positive-definiteness across the parameter space (physical constraint: kinetic energy > 0)
- Drag forces scale as |v|·v (quadratic drag, not linear)
- Pole tilt correctly maps to drogue depth via `_z_eff_batch()`

### 5. Numerical edge cases

- NaN/inf recovery in `_rhs_batch()` (lines 442–445): inject overflow, verify recovery replaces M with identity and F with zero
- Extreme pole tilts: near-vertical (θ ≈ 0) and near-horizontal (θ ≈ π/2)
- Zero drogue velocity: stability in coordinate conversions

### 6. Integration test

Full chain: synthetic gridded data → `compute_stokes_profile` → `DroguedDrifter` → final drift velocity. Currently no test spans multiple modules.

---

## Slow Test Infrastructure

The symbolic derivation test (gap #3) takes ~30–60 s because sympy must derive the Euler-Lagrange equations, apply CSE, and lambdify from scratch. This is too slow for the default test loop but important for CI and cache validation.

**Approach**: Use `@pytest.mark.slow` and configure pytest to skip slow tests by default:

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]
addopts = "-m 'not slow'"
```

Usage:
- Fast iteration: `pixi run pytest` (skips slow tests)
- Slow tests only: `pixi run pytest -m slow`
- Everything: `pixi run pytest -m ""`

This lets developers iterate quickly while CI runs the full suite.

---

## Implementation Order

1. **stokes.py tests** — quick win, zero coverage → 100%, no infrastructure needed
2. **Slow test marker** — add `@pytest.mark.slow` infrastructure to pyproject.toml
3. **Symbolic derivation test** — mark as slow, force cache miss, compare against cached
4. **Physics validation tests** — positive-definiteness, drag scaling
5. **Parcels interpolator tests** — requires synthetic FieldSet setup, more involved
6. **Numerical edge cases** — overflow recovery, extreme angles
7. **Integration test** — full chain, depends on stokes + drifter tests existing
