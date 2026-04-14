# Drifter model: physics and API

This document covers the `DroguedDrifter` -- the primary model. The package also
includes [PointSurfaceDrifter](point-surface-drifter.md) (a simpler baseline
model) and supports adding new models via
[LagrangianMechanicsModel](architecture.md).

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

## Stereographic coordinates

The pole direction is parameterized via stereographic projection from the south
pole onto a plane at the north pole. At equilibrium (`theta = pi`, drogue
hanging down), `(u, v) = (0, 0)`. The expressions are smooth everywhere near the
origin.

The internal state vector is `[x, y, u, v, xd, yd, ud, vd]` (8 components).
The public API (`integrate`) accepts and returns spherical angles
`(theta, phi, thetad, phid)` -- the conversion is handled transparently by the
`integrate()` override on `DroguedDrifter`.

## Constructor parameters

```python
from mechanical_drifters.models.drogued_drifter import DroguedDrifter, DroguedDrifterPhysics

# Option 1: defaults (Callies et al. 2017 at rho=1025 kg/m^3)
dd = DroguedDrifter()

# Option 2: keyword overrides
dd = DroguedDrifter(l=5.0, k_d=200.0)

# Option 3: explicit Physics instance
physics = DroguedDrifterPhysics(
    m_b=1.0, m_d=2.7, m_hat_d=1.0, m_tilde_d=101.0, m_tilde_b=1.9,
    l=3.0, g=9.81, k_b=12.0, k_d=154.0,
)
dd = DroguedDrifter(physics=physics, backend="numpy")
```

## Public methods

### `integrate`

```python
t, Y, max_accel = dd.integrate(
    sample_uv,
    t_span=(0, 120),
    y0=None,           # (N, 8) in public coords, or None for cold start
    t_eval=None,       # None for final state only, or array of times
    atol=1e-3,
    rtol=1e-3,
)
# t: (T,) time array
# Y: (T, N, 8) in public coords [x, y, theta, phi, xd, yd, thetad, phid]
# max_accel: scalar convergence diagnostic
```

DroguedDrifter overrides the base `integrate` to convert spherical coordinates
on the way in and out. Callers always see spherical coords.

### `drift_velocity`

```python
vel = dd.drift_velocity(Y[-1])  # (N, 2) array of [xd, yd]
```

### `to_xarray`

```python
ds = dd.to_xarray(t, Y)
# xr.Dataset with dims (time, traj) and variables x, y, theta, phi, xd, yd, thetad, phid
```

### Direct EOM evaluation

For exploring the equations of motion at specific states:

```python
from mechanical_drifters.eom import _get_eom_callables
from mechanical_drifters.models.drogued_drifter import DroguedDrifter, DroguedDrifterPhysics, DroguedDrifterState

dd = DroguedDrifter()
qdd_raw, M_raw, F_raw, pack_eom_args = _get_eom_callables(dd)

state = DroguedDrifterState(
    u_stereo=0.0, v_stereo=0.0,
    xd=0.0, yd=0.0,
    ud_stereo=0.0, vd_stereo=0.0,
    U_b=1.0, V_b=0.0, U_d=-1.0, V_d=0.0,
)
args = pack_eom_args(dd.physics, state)
M = M_raw(*args)   # (4, 4) numpy array
F = F_raw(*args)   # (4, 1) numpy array
```

## Standalone vs Parcels

Use `DroguedDrifter` standalone (with `integrate`) when exploring the drifter
physics with idealized velocity profiles or computing drift velocities for a
grid of flow conditions.

Use with Parcels (via `mechanical_drifters.parcels.make_kernel`) when advecting
drifters through spatially and temporally varying ocean fields. See
[parcels-v4-coupling.md](parcels-v4-coupling.md) for details.
