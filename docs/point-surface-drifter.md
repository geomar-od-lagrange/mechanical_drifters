# Point surface drifter

`PointSurfaceDrifter` models a point particle at the ocean surface with
quadratic drag. It is the simplest `LagrangianMechanicsModel` subclass --
two generalized coordinates (x, y), no pole, no drogue, no depth
dependence.

## Why it exists

- **Baseline comparison.** In sheared flow, DroguedDrifter drifts slower
  than the surface current because the drogue anchors it to deeper
  water. PointSurfaceDrifter drifts at exactly the surface current,
  making it the natural null hypothesis.
- **Pipeline validation.** The full Lagrangian machinery (symbolic
  derivation, caching, lambdification, batch integration, Parcels
  kernel) works identically for PointSurfaceDrifter and DroguedDrifter.
  If the pipeline breaks, the simpler model is easier to debug.

## Physics

The particle has mass `m`, horizontal added mass `m_tilde`, and
quadratic drag coefficient `k`. At the surface (z = 0), it experiences
drag from the ocean current `(U, V)`.

**Lagrangian:** `L = 1/2 (m + m_tilde) (xd^2 + yd^2)`

**Drag force:** `F = -k |v - u| (v - u)`

**Steady state:** at equilibrium, `qdd = 0`, which requires `v = u`.
The particle drifts at exactly the surface current velocity.

## API

```python
from mechanical_drifters.models.point_surface_drifter import PointSurfaceDrifter, PointSurfacePhysics

# Default physics: m=1, m_tilde=1, k=10
psd = PointSurfaceDrifter()

# Custom physics
psd = PointSurfaceDrifter(
    physics=PointSurfacePhysics(m=2.0, m_tilde=5.0, k=100.0),
)

# Integrate
def sample_uv(z):
    N = len(np.atleast_1d(z))
    return np.full(N, 0.3), np.zeros(N)

t, Y, max_accel = psd.integrate(sample_uv)
drift_vel = psd.drift_velocity(Y[-1])
# drift_vel ~ [[0.3, 0.0]]

# Parcels kernel
from mechanical_drifters.parcels import make_kernel
kernel = make_kernel(psd)
pset.execute(kernels=[kernel], dt=300, runtime=86400)
```

### State vector

`[x, y, xd, yd]` -- position and velocity. `state_size = 4`.

### Physics parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `m` | Particle mass [kg] | 1.0 |
| `m_tilde` | Horizontal added mass [kg] | 1.0 |
| `k` | Drag coefficient [kg/m] | 10.0 |
