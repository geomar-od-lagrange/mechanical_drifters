# Drifter model: physics and API

The `DroguedDrifter` class models a surface drifter buoy connected by a rigid
pole to a subsurface drogue. The buoy sits at the sea surface (z = 0) and
experiences drag from surface currents. The drogue hangs below at depth, feeling
drag from the current at that depth. Under vertical shear, the pole tilts, the
drogue rises from its equilibrium depth, and the buoy drifts at a velocity that
is a compromise between the surface current and the deeper current. The model
computes this compromise from first principles.

## Physical setup

The system has three rigid bodies:

- **Buoy** (mass `m_b`): a cylinder floating at the surface. Drag coefficient
  `k_b` sets the quadratic drag force from the surface current.
- **Pole** (length `l`): a rigid, massless rod connecting buoy and drogue. The
  default length is 3 m, matching the Callies et al. (2017) drifter design.
- **Drogue** (mass `m_d`): a cross-shaped subsurface drag element. Drag
  coefficient `k_d` sets the quadratic drag force from the current at drogue
  depth.

The drogue also has two hydrodynamic corrections:

- **Added mass** (`m_tilde_d` for drogue, `m_tilde_b` for buoy): the
  surrounding water that accelerates with each body. Computed from the geometry
  using `drogue_horizontal_added_mass()` and `buoy_horizontal_added_mass()`.
- **Buoyancy correction** (`m_hat_d`): the mass of water displaced by the
  drogue, which reduces its effective gravitational weight.

Drag forces are quadratic: `F = -k |v_rel| v_rel` where `v_rel` is the velocity
of the body relative to the local ocean current. Only horizontal drag is
modeled -- vertical drag on the pole is neglected, which is valid when the pole
tilt is small (the typical regime for ocean drifters).

## Lagrangian mechanics derivation

The equations of motion are derived symbolically using SymPy's mechanics module,
not hand-derived. The full chain runs from the Lagrangian to executable NumPy
functions:

1. **Define generalized coordinates**: buoy position `(x, y)` and pole
   direction, parameterized by stereographic coordinates `(u, v)` (see below).
2. **Compute kinetic energy** `T` from buoy and drogue velocities (including
   added masses) and **potential energy** `V` from drogue weight minus buoyancy.
3. **Form the Lagrangian** `L = T - V`.
4. **Compute generalized forces** `Q` from quadratic drag on buoy and drogue.
5. **Apply the Euler-Lagrange equations**: `d/dt(dL/dqd) - dL/dq = Q`.
6. **Extract the mass matrix** `M` and force vector `F` such that `M * qdd = F`.
7. **Solve symbolically** for `qdd = M^{-1} F` (the generalized accelerations).
8. **Lambdify** the resulting expressions to NumPy callables using `sp.lambdify`
   with common subexpression elimination (`cse=True`).

The symbolic derivation is expensive (~2 min) but cached to a pickle file
(`data/eom_cache.pkl`). The cache is invalidated automatically when the
derivation source code or the SymPy version changes.

## Stereographic coordinates

The pole direction is a point on the unit sphere, naturally described by
spherical angles `(theta, phi)`. However, `phi` (the azimuthal angle) is
undefined when `theta = pi` (drogue hanging straight down) -- any rotation
about the vertical axis represents the same physical state. This makes the
equations of motion singular at the equilibrium configuration.

Instead, the pole direction is parameterized via stereographic projection from
the south pole onto a plane at the north pole:

    sin(theta) cos(phi) = 4u / (u^2 + v^2 + 4)
    sin(theta) sin(phi) = 4v / (u^2 + v^2 + 4)
    cos(theta)          = (u^2 + v^2 - 4) / (u^2 + v^2 + 4)

At equilibrium (`theta = pi`, drogue hanging down), `(u, v) = (0, 0)`. The
expressions are smooth everywhere near the origin, so no regularization or
special-case handling is needed. The only singularity is at `theta = 0` (drogue
pointing straight up), which is outside the physical operating regime.

The internal state vector is `[x, y, u, v, xd, yd, ud, vd]` (8 components).
The public API accepts and returns spherical angles `(theta, phi, thetad, phid)`
-- the conversion is handled transparently.

## Constructor parameters

```python
from drogued_drifters import DroguedDrifter

dd = DroguedDrifter(
    m_b=1.0,          # buoy dry mass [kg]
    m_d=2.7,          # drogue dry mass [kg]
    m_hat_d=1.0,      # drogue buoyancy correction [kg]
    m_tilde_d=101.0,  # drogue horizontal added mass [kg]
    m_tilde_b=1.9,    # buoy horizontal added mass [kg]
    l=3.0,            # pole length [m]
    k_b=12.0,         # buoy drag coefficient [kg/m]
    k_d=154.0,        # drogue drag coefficient [kg/m]
    g=9.81,           # gravitational acceleration [m/s^2]
    get_uv=None,      # scalar velocity callback (see below)
    sample_uv=None,   # batch velocity sampler (see below)
    backend="numpy",  # "numpy" or "numba"
)
```

All defaults match the Callies et al. (2017) drifter geometry at
`rho = 1025 kg/m^3`. The helper functions `drogue_horizontal_added_mass`,
`buoy_horizontal_added_mass`, `drogue_horizontal_drag_coeff`, and
`buoy_horizontal_drag_coeff` compute `m_tilde_d`, `m_tilde_b`, `k_d`, and `k_b`
from raw dimensions if you have a different drifter design.

### Velocity callback (`get_uv`)

The `get_uv` callback supplies ocean currents to the single-particle methods
(`get_full_solution`, `get_final_drift`). Its signature is:

```python
def get_uv(*, t, x, y, z) -> tuple[float, float]:
    """Return (U, V) current velocity [m/s] at position (x, y, z) and time t.

    z is positive upward: 0 = surface, negative = below.
    """
```

Use `functools.partial` to bind external data (e.g., an xarray dataset) before
passing it to the constructor. If `get_uv` is `None`, a hardcoded test profile
is used (surface velocity `(1, 1)`, subsurface `(-1, -1)`).

The batch method `get_final_drift_batch` uses a different callback interface
(`sample_uv(z)`) -- see below.

### Backend

The `backend` parameter selects the computational backend for the generalized
acceleration evaluator:

- `"numpy"` (default): pure NumPy evaluation. No extra dependencies.
- `"numba"`: JIT-compiles the lambdified function with `numba.njit` for ~25x
  speedup on the hot path. Requires numba to be installed. The JIT compilation
  happens at construction time, so the first `DroguedDrifter(backend="numba")`
  call is slower, but all subsequent ODE evaluations are faster.

## Public methods

### `get_full_solution`

```python
ds = dd.get_full_solution(
    t_span=(0, 120),
    t_eval=np.arange(0, 121),
    x=0.0, y=0.0,
    theta=np.pi, phi=0.0,       # pole direction (default: hanging down)
    xd=0.0, yd=0.0,             # buoy velocity
    thetad=0.0, phid=0.0,       # angular velocity
)
```

Integrates the full equations of motion for a single particle over `t_span`
using `scipy.integrate.solve_ivp`. Returns an `xarray.Dataset` with `time` as
coordinate and state variables `x, y, theta, phi, xd, yd, thetad, phid` as
data variables (all in spherical coordinates, converted from the internal
stereographic representation).

Use this when you need the full time series -- for example, to study transient
dynamics, visualize pole tilt over time, or verify that the system reaches
steady state.

### `get_final_drift`

```python
xd_final, yd_final, max_accel = dd.get_final_drift(t_span=(0, 120))
```

Integrates to steady state and returns only the final buoy drift velocity
`(xd, yd)` in m/s, plus `max_accel` as a convergence diagnostic (smaller is
better -- it measures the residual acceleration at the final time). Uses the
same `get_uv` callback as `get_full_solution`.

Use this for single-particle steady-state queries where you only need the drift
velocity, not the full trajectory.

### `get_final_drift_batch`

```python
xd_final, yd_final, Y_final, max_accel = dd.get_final_drift_batch(
    sample_uv=my_sampler,
    t_span=(0, 120),
    y0=None,           # optional initial state (N, 8) for warm-starting
    atol=1e-3,
    rtol=1e-3,
)
```

Computes steady-state drift for N particles simultaneously. All N particles are
stacked into a single `(8N,)` ODE system so that `solve_ivp` overhead is paid
once, and the vectorized RHS evaluates all particles in parallel using NumPy
broadcasting.

The `sample_uv` callback has a different interface from `get_uv`:

```python
def sample_uv(z) -> tuple[np.ndarray, np.ndarray]:
    """Return (U, V) arrays of shape (N,) at depth z (scalar or (N,) array)."""
```

This is the primary method used by the Parcels kernel and by batch-evaluation
notebooks. The `Y_final` array (shape `(N, 8)`) contains the full final state
in public coordinates `[x, y, theta, phi, xd, yd, thetad, phid]` and can be
passed back as `y0` for warm-starting subsequent calls.

## Direct EOM evaluation

The mass matrix `M`, force vector `F`, and generalized accelerations `qdd = M^{-1}F`
can be evaluated directly without running the ODE integrator. This is useful for
studying the equations of motion at a specific state, validating the physics, or
building custom integrators.

```python
from drogued_drifters import DrifterPhysics, EOMState, qdd_func, M_func, F_func
import numpy as np

physics = DrifterPhysics(
    m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
    l=3.0, g=9.81, k_b=12.0, k_d=154.0,
)
state = EOMState(
    u_stereo=0.0, v_stereo=0.0,  # drogue at equilibrium (hanging straight down)
    xd=0.0, yd=0.0,              # buoy at rest
    ud_stereo=0.0, vd_stereo=0.0,
    U_b=1.0, V_b=0.0,            # surface current 1 m/s east
    U_d=-1.0, V_d=0.0,           # drogue current 1 m/s west
)

qdd = qdd_func(physics, state)   # generalized accelerations, shape (4,)
M   = M_func(physics, state)     # mass matrix, shape (4, 4)
F   = F_func(physics, state)     # force vector, shape (4,)
# sanity check: M @ qdd ≈ F
assert np.allclose(M @ qdd, F)
```

All three functions accept scalar or batch (shape `(N,)` arrays in `EOMState`)
input. Batch calls return `(N, 4, 4)`, `(N, 4)`, and `(N, 4)` respectively.

`qdd_func` is a shortcut for `_make_qdd_func("numpy")`. For the numba backend,
construct `DroguedDrifter(backend="numba")` instead, which uses a JIT-compiled
evaluator internally.

## Standalone vs Parcels

Use `DroguedDrifter` standalone (with `get_full_solution`, `get_final_drift`, or
`get_final_drift_batch`) when:

- Exploring the drifter physics with idealized velocity profiles.
- Computing steady-state drift velocities for a grid of flow conditions.
- You do not need spatial advection -- just the drift velocity at a point.

Use with Parcels (via `make_dd_kernel`) when:

- Advecting drifters through spatially and temporally varying ocean fields.
- Comparing drogued drifters against point particles in the same simulation.
- Working with real ocean model output (CMEMS, NEMO, etc.) on structured or
  unstructured grids.

See [parcels-v4-coupling.md](parcels-v4-coupling.md) for details on the kernel
implementation, depth handling, spherical/flat mesh auto-detection, and error
handling.
