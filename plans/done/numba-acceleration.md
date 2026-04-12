# Numba acceleration for the ODE hot path

## Motivation

Profiling the Baltic drifter simulation (notebook 04, 6 particles,
288 h, dt=300 s) shows:

| Component | Time | % | Calls |
|---|---|---|---|
| `_lambdifygenerated` (qdd) | 155 s | 54% | 1.4M |
| `sample_uv` (profile interp) | 26 s | 9% | 2.9M |
| `fieldset.UV.eval` (Parcels) | 30 s | 10% | 17K |
| ODE stepper + other overhead | ~76 s | 27% | — |

The lambdified function dominates because it's called 1.4M times on
tiny arrays (N=6).  At this size, numpy dispatch overhead per call
(~100 µs) far exceeds the actual arithmetic.  A quick benchmark shows
numba `@njit` brings this to ~4 µs/call — **25x speedup**, turning
155 s into ~6 s.

The second target is `sample_uv` (26 s, 2.9M calls).  It does
`searchsorted` + fancy indexing on (D, N) arrays — also
numba-friendly.

## Constraint: numba must be fully optional

This code is intended to be shown to Parcels developers.  Numba is
substantial baggage and adds cognitive load.  The design must ensure:

1. **No numba import at module level.**  The core physics
   (`drifter.py`, `lagrange_model.py`, `parcels_v4.py`) must work
   without numba installed.
2. **No numba in the public API.**  Users who don't care about
   performance should never see numba-related code or errors.
3. **Opt-in activation.**  The numba-accelerated path is selected
   explicitly, not auto-detected.

## Design options

### Option A: separate `_numba.py` accelerator module

A new `src/drogued_drifters/_numba.py` module that:

- Imports numba (fails fast if not installed)
- Wraps `_lambdifygenerated` with `@njit(cache=True)`
- Provides a numba-compiled `sample_uv` equivalent
- Provides a numba-compiled `_rhs_batch` that fuses qdd + sample_uv +
  state unpacking into one compiled function
- Exports a `patch(dd)` or `accelerate(dd)` function that
  monkey-patches `dd._rhs_batch` (or returns an accelerated copy)

Usage:

```python
from drogued_drifters._numba import accelerate
dd = accelerate(DroguedDrifter())
```

Pro: zero impact on existing code.  Con: monkey-patching is fragile,
and the accelerated `_rhs_batch` must track changes in the original.

### Option B: strategy parameter on DroguedDrifter

```python
dd = DroguedDrifter(backend="numba")  # default: "numpy"
```

`__init__` imports `_numba` lazily and swaps `_rhs_batch`.  The rest
of the class is unchanged.

Pro: clean API.  Con: the class now knows about backends, even if
only as a one-line lazy import.

### Option C: compiled qdd only, no fused RHS

Only accelerate the lambdified function.  In `_get_eom_callables`,
optionally wrap `qdd_raw` with `@njit(cache=True)`:

```python
def _get_eom_callables(use_numba=False):
    ...
    if use_numba:
        from numba import njit
        qdd_raw = njit(cache=True)(qdd_raw)
    ...
```

Pro: minimal change, targets the 54% bottleneck.  Con: doesn't
accelerate `sample_uv` (9%) or `pack_eom_args` (1.5%), and
`use_numba` propagates through the call chain.

### Recommendation

Start with **Option C** for validation (targets 54% with minimal
risk), then move to **Option A** if fusing the full RHS is worth the
additional 10% gain.  Option B is premature — add the constructor
parameter only if numba becomes a permanent feature rather than an
optimization experiment.

## What to accelerate (priority order)

### 1. `_lambdifygenerated` (qdd_raw) — 54% of runtime

The sympy-generated function is pure scalar arithmetic + `sqrt`.  No
numpy-specific calls.  `@njit` compiles it directly.

Benchmark (N=6):
- numpy lambdify: 105 µs/call
- numba njit: 4 µs/call
- compilation: 2.4 s (one-time, cacheable)

### 2. `sample_uv` — 9% of runtime

`searchsorted` + fancy indexing on (D, N) arrays.  Numba supports
both.  Could be compiled standalone or fused into a compiled
`_rhs_batch`.

### 3. `pack_eom_args` — 1.5% of runtime

Pure Python tuple construction from struct fields.  Would disappear
if qdd arguments are passed directly in a fused `_rhs_batch`.

### 4. Fused `_rhs_batch` — eliminates intermediate arrays

The full RHS (unpack state → sample_uv → build EOMState →
pack_eom_args → qdd_raw → assemble dY) could be one `@njit`
function.  This eliminates all intermediate (N,) array allocations
and Python dispatch between steps.  Expected to capture most of the
remaining 27% overhead.

Trade-off: a fused RHS is harder to maintain and must be kept in sync
with the numpy version.  Only worthwhile if the per-component
approach leaves significant performance on the table.

## Validation approach

1. **Monkey-patch in profiling notebook** (04_prof) to measure real
   speedup without touching src.
2. **Correctness**: assert `allclose` between numpy and numba paths
   on the full Baltic simulation trajectories.
3. **Decide** based on measured speedup whether numba is worth the
   dependency.

## Related: warm-start + fixed-step ODE solver

Profiling warm-start with the current `solve_ivp` (RK45 adaptive,
`t_span=(0, 120)`) showed **no speedup** — 433 RHS evals/call vs 416
cold.  The adaptive solver wastes effort stepping through 120 s of
near-equilibrium state, with step rejections from small perturbations
when the velocity field changes between timesteps.

The fix is to combine warm-starting with:

1. **Short t_span** (e.g. 5–10 s instead of 120 s) — the DD only
   needs to track the slowly-drifting equilibrium, not re-converge
   from scratch.
2. **Fixed-step ODE scheme** (Euler or RK4, dt_ode ~ 1 s, ~10 steps)
   — completely predictable cost per call, no adaptive overhead or
   step rejections.  Natural fit: the steady state is a stable
   attractor, the perturbation between timesteps is small, and the
   first cold-start call (which does need full convergence) can use
   the current `solve_ivp` with `t_span=(0, 120)`.

Expected combined effect (warm-start + fixed step + numba):
- RHS evals: 416 → ~10 per call (40x fewer)
- Per-eval cost: 105 µs → 4 µs (25x numba speedup)
- Total qdd time: 155 s → ~0.06 s (~2500x)
- Bottleneck shifts to Parcels `UV.eval` (30 s) and `sample_uv` (26 s)

This is independent of the numba decision — warm-start + fixed step
alone would bring the 155 s lambdify cost to ~0.4 s.

## Risks

| Risk | Mitigation |
|---|---|
| numba compilation adds startup latency | `cache=True` persists compiled code to disk; 2.4 s first run only |
| sympy-generated code changes break numba cache | Cache key already includes source hash; numba cache auto-invalidates |
| numba version compatibility | Pin in optional dependency group; test in CI |
| Cognitive load for Parcels developers | Keep numba in a separate module, not visible in the main code path |
| N=6 benchmark may not generalize | Profile with larger N (100, 1000) to check scaling |
