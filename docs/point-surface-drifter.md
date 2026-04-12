# Point surface drifter

`PointSurfaceDrifter` models a point particle at the ocean surface with
quadratic drag. It is the simplest `LagrangianMechanicsModel` subclass —
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

**Lagrangian:**

```
L = 1/2 (m + m_tilde) (xd² + yd²)
```

No potential energy — the particle is constrained to the surface.

**Drag force:**

```
F = -k |v - u| (v - u)
```

where `v = (xd, yd)` is the particle velocity and `u = (U, V)` is the
surface current.

**Equations of motion:**

```
(m + m_tilde) qdd = -k |v - u| (v - u)
```

The mass matrix M is diagonal: `(m + m_tilde) I`. The force vector F
is the drag.

**Steady state:** at equilibrium, `qdd = 0`, which requires `v = u`.
The particle drifts at exactly the surface current velocity. This
holds regardless of the values of `m`, `m_tilde`, and `k` — the
steady state depends only on the drag vanishing, not on its magnitude.

Note: quadratic drag gives algebraic O(1/t) convergence to equilibrium,
not exponential. The time scale `(m + m_tilde) / (k * |u|)` controls
how fast the particle approaches the current, but the approach is a
power law, not an exponential decay.

## API

```python
from mechanical_drifters import PointSurfaceDrifter, PointSurfacePhysics

# Default physics: m=1, m_tilde=1, k=10
psd = PointSurfaceDrifter()

# Custom physics
psd = PointSurfaceDrifter(
    physics=PointSurfacePhysics(m=2.0, m_tilde=5.0, k=100.0),
)

# Steady-state drift
def sample_uv(z):
    N = len(np.atleast_1d(z))
    return np.full(N, 0.3), np.zeros(N)

drift_vel, Y_final, max_accel = psd.steady_state_batch(sample_uv)
# drift_vel ≈ [[0.3, 0.0]]

# Parcels kernel
kernel = psd.make_kernel()
pset.execute(kernels=[kernel], dt=300, runtime=86400)
```

### State vector

`[x, y, xd, yd]` — position and velocity. `state_size = 4`.

### Physics parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `m` | Particle mass [kg] | 1.0 |
| `m_tilde` | Horizontal added mass [kg] | 1.0 |
| `k` | Drag coefficient [kg/m] | 10.0 |

### `_max_depth`

Returns 0.0. The Parcels kernel only samples the surface — no depth
profile extraction is needed.

## Example

[`examples/point_drifter/01_surface_tracking`](../examples/point_drifter/01_surface_tracking.md)
demonstrates steady-state convergence in uniform and sheared flows.
