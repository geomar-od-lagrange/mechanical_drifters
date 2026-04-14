# Package architecture

`mechanical_drifters` provides Lagrangian mechanics models for ocean
drifters. Each model defines a Lagrangian (kinetic minus potential
energy) and non-conservative forces (quadratic drag), derives equations
of motion symbolically with SymPy, and integrates them numerically with
SciPy. The package also generates Parcels kernels for spatially
distributed particle tracking.

## Models

Two models are included:

- **DroguedDrifter** -- a surface buoy connected by a rigid pole to a
  subsurface drogue. Four generalized coordinates (buoy position +
  pole direction in stereographic coordinates). The dominant use case.
  See [drifter-model.md](drifter-model.md).

- **PointSurfaceDrifter** -- a point particle at the ocean surface with
  quadratic drag. Two generalized coordinates (x, y). A baseline
  comparison model whose steady-state drift equals the surface current
  exactly. See [point-surface-drifter.md](point-surface-drifter.md).

## Base class: `LagrangianMechanicsModel`

All models inherit from `LagrangianMechanicsModel` in `base.py`.
Subclasses provide:

| What | Type | Purpose |
|------|------|---------|
| `Physics` | NamedTuple class | Physical constants (mass, drag, geometry) |
| `State` | NamedTuple class | Per-timestep state variables + forcing |
| `n_q` | int | Number of generalized coordinates |
| `state_names` | tuple of str | Names for state vector components (used by `to_xarray`) |
| `_derive_symbolic()` | method | SymPy derivation returning (M, F, args) |
| `_rhs_batch(Y, sample_uv)` | method | Vectorized ODE right-hand side |
| `drift_velocity(Y)` | method | Extract drift velocity from state array |

The base class provides:

- **`integrate(sample_uv)`** -- stacks N particles into a single ODE
  system, integrates with `solve_ivp`, returns `(t, Y, max_acceleration)`.
  `t_eval=None` returns only the final state (T=1).
- **`to_xarray(t, Y)`** -- wraps integrate output into an xr.Dataset
  using `state_names`.
- **`state_size`** -- `2 * n_q`.
- **`_cache_path`** -- auto-derived from the class name.

## EOM pipeline

The path from physics to executable functions:

1. **Symbolic derivation** (`_derive_symbolic`): each model defines its
   Lagrangian, computes the Euler-Lagrange equations, and extracts the
   mass matrix M and force vector F such that M qdd = F.

2. **Caching**: the symbolic result is pickled to
   `data/eom_cache_{snake_case_name}.pkl`. The cache key is a hash of
   the derivation source code, the SymPy version, and the Python
   version. Cache misses trigger a re-derivation (~2 min for
   DroguedDrifter, negligible for PointSurfaceDrifter).

3. **Lambdification**: `sp.lambdify` with `cse=True` converts the
   symbolic expressions to NumPy callables. qdd is lambdified as a
   tuple of scalars (hot path, numba-compatible). M and F are
   lambdified as full Matrix expressions (for exploration).

4. **Packer**: `_build_packer` inspects the lambda parameter names and
   maps them to fields in the Physics and State NamedTuples. This runs
   once and returns a closure that assembles positional arguments from
   `(physics, state)`.

5. **qdd evaluator**: `_make_qdd_func` combines the lambdified function
   with the packer, handles scalar vs batch input, and optionally
   JIT-compiles with numba (`backend="numba"`).

All of this is keyed by class name. Two DroguedDrifter instances share
the same compiled EOM callables. A PointSurfaceDrifter gets its own.

## Module layout

```
src/mechanical_drifters/
  __init__.py           # empty
  base.py               # LagrangianMechanicsModel
  eom.py                # caching, lambdification, qdd evaluator
  parcels.py            # Parcels kernel factory + profile sampler
  stokes.py             # Stokes drift profile computation
  models/
    __init__.py
    drogued_drifter.py   # DroguedDrifter, DroguedDrifterPhysics, DroguedDrifterState, coord helpers
    point_surface_drifter.py  # PointSurfaceDrifter, PointSurfacePhysics, PointSurfaceState
  data/
    eom_cache_*.pkl      # cached symbolic derivations
```

## Adding a new model

1. Create `models/my_model.py` with a `MyPhysics` NamedTuple, a
   `MyState` NamedTuple, and a `MyModel(LagrangianMechanicsModel)`
   class.

2. Set the class attributes: `Physics`, `State`, `n_q`, `state_names`.

3. Implement: `_derive_symbolic`, `_rhs_batch`, `drift_velocity`.
   Provide a default physics via a class attribute or custom `__init__`.

4. The symbol names in the `args` tuple returned by `_derive_symbolic`
   must exactly match `Physics._fields + State._fields`.

The model inherits `integrate` and `to_xarray` for free.
The EOM cache is auto-derived from the class name. See
`PointSurfaceDrifter` for a minimal example.
