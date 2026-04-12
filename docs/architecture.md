# Package architecture

`mechanical_drifters` provides Lagrangian mechanics models for ocean
drifters. Each model defines a Lagrangian (kinetic minus potential
energy) and non-conservative forces (quadratic drag), derives equations
of motion symbolically with SymPy, and integrates them numerically with
SciPy. The package also generates Parcels kernels for spatially
distributed particle tracking.

## Models

Two models are included:

- **DroguedDrifter** — a surface buoy connected by a rigid pole to a
  subsurface drogue. Four generalized coordinates (buoy position +
  pole direction in stereographic coordinates). The dominant use case.
  See [drifter-model.md](drifter-model.md).

- **PointSurfaceDrifter** — a point particle at the ocean surface with
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
| `_drift_velocity_indices` | tuple of int | Which state vector entries are the drift velocity output |
| `default_physics()` | method | Default Physics instance |
| `_derive_symbolic()` | method | SymPy derivation returning (M, F, args) |
| `_rhs_batch(Y, sample_uv)` | method | Vectorized ODE right-hand side |
| `_max_depth(physics)` | method | Maximum depth to sample from a fieldset |

The base class provides:

- **`steady_state_batch(sample_uv)`** — stacks N particles into a
  single ODE system, integrates to steady state with `solve_ivp`,
  returns drift velocities.
- **`make_kernel()`** — creates a Parcels-compatible kernel function.
- **`state_size`** — `2 * n_q`.
- **`_cache_path`** — auto-derived from the class name.

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
   symbolic expressions to NumPy callables. Common subexpression
   elimination reduces redundant computation.

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
  __init__.py           # public exports
  base.py               # LagrangianMechanicsModel
  eom.py                # caching, lambdification, evaluation
  parcels.py            # Parcels kernel factory
  velocity.py           # make_profile_sampler
  coords.py             # stereographic ↔ spherical coordinate transforms
  stokes.py             # Stokes drift profile computation
  models/
    __init__.py
    drogued_drifter.py   # DroguedDrifter, DrifterPhysics, EOMState
    point_surface_drifter.py  # PointSurfaceDrifter, PointSurfacePhysics, PointSurfaceState
  data/
    eom_cache_*.pkl      # cached symbolic derivations
```

## Adding a new model

1. Create `models/my_model.py` with a `MyPhysics` NamedTuple, a
   `MyState` NamedTuple, and a `MyModel(LagrangianMechanicsModel)`
   class.

2. Set the four class attributes: `Physics`, `State`, `n_q`,
   `_drift_velocity_indices`.

3. Implement four methods: `default_physics`, `_derive_symbolic`,
   `_rhs_batch`, `_max_depth`.

4. The symbol names in the `args` tuple returned by `_derive_symbolic`
   must exactly match `Physics._fields + State._fields`.

5. Export from `__init__.py`.

The model inherits `steady_state_batch` and `make_kernel` for free.
The EOM cache is auto-derived from the class name. See
`PointSurfaceDrifter` for a minimal example.
