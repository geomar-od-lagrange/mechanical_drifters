"""Generic Parcels coupling for LagrangianMechanicsModel subclasses.

All Parcels-facing code lives here.  The kernel helper uses
``fieldset.UV.eval()`` per depth level using Parcels' native VectorField
interpolator (handles A-grids, C-grids, curvilinear, and unstructured
grids, including vector rotation and spherical mesh conversion).
"""

import numpy as np

_DEG2M = 1852.0 * 60.0


def _make_profile_sampler(depth_levels, U_profiles, V_profiles):
    """Build a fast ``sample_uv(z)`` interpolator from pre-sampled velocity profiles.

    Takes velocity profiles that have already been sampled at fixed depth
    levels (e.g. once per Parcels timestep) and returns a callable that
    performs linear interpolation in z.  This avoids repeated expensive
    queries to the original data source during ODE integration.

    Args:
        depth_levels: 1-D array of vertical positions [m], positive upward
            (0 = surface, negative = below MSL), shape ``(D,)``.  Must be
            sorted ascending (deepest first, e.g. ``[-20, -10, 0]``).
        U_profiles: Eastward velocity at each depth for each particle,
            shape ``(D, N)``.
        V_profiles: Northward velocity, same shape.

    Returns:
        Callable ``sample_uv(z) -> (U, V)`` where ``z`` is a scalar or
        ``(N,)`` array and the return arrays have shape ``(N,)``.
    """
    depth_levels = np.asarray(depth_levels, dtype=float).copy()
    U_profiles = np.asarray(U_profiles, dtype=float).copy()  # (D, N)
    V_profiles = np.asarray(V_profiles, dtype=float).copy()
    D, N = U_profiles.shape

    def sample_uv(z):
        z_arr = np.broadcast_to(np.asarray(z, dtype=float), N)
        # Vectorized linear interpolation in z
        idx = np.searchsorted(depth_levels, z_arr).clip(1, D - 1)
        z0 = depth_levels[idx - 1]
        z1 = depth_levels[idx]
        w = (z_arr - z0) / np.maximum(z1 - z0, 1e-30)
        # Fancy-index: U_profiles[idx-1, particle_index]
        p = np.arange(N)
        U = U_profiles[idx - 1, p] * (1 - w) + U_profiles[idx, p] * w
        V = V_profiles[idx - 1, p] * (1 - w) + V_profiles[idx, p] * w
        return U, V

    return sample_uv


def _extract_profiles(particles, fieldset, max_depth):
    """Extract velocity profiles from the fieldset and build a depth interpolator.

    Samples the fieldset at each depth level from the surface down to
    ``max_depth`` (plus one grid cell margin).  Converts Parcels' degree-based
    velocities to m/s on spherical grids.  Returns a ``sample_uv(z)``
    callable suitable for the ODE integrator.

    Args:
        particles: Parcels ParticleSet.
        fieldset: Parcels FieldSet with a ``UV`` VectorField.
        max_depth: Maximum depth [m, positive] to sample.

    Returns:
        Callable ``sample_uv(z) -> (U, V)`` where ``z`` is a scalar or
        ``(N,)`` depth array (m, positive upward) and the return arrays
        have shape ``(N,)`` in m/s.
    """
    lat = np.asarray(particles.lat)
    lon = np.asarray(particles.lon)
    time = particles.time
    N = len(lat)

    is_spherical = fieldset.U.grid._mesh == "spherical"

    # Depth levels: surface to first level beyond max_depth (or all if
    # max_depth covers the full water column).  At least 2 for interpolation.
    all_depths = np.asarray(fieldset.U.grid.depth, dtype=float)
    cutoff = min(
        np.searchsorted(all_depths, max_depth, side="right") + 1, len(all_depths)
    )
    depth_levels = all_depths[: max(cutoff, 2)]
    D = len(depth_levels)

    # Grid-agnostic profile extraction via default VectorField interpolator
    U_profiles = np.empty((D, N))
    V_profiles = np.empty((D, N))
    for iz, z_level in enumerate(depth_levels):
        z_arr = np.full(N, z_level)
        u, v = fieldset.UV.eval(time, z_arr, lat, lon)[:2]
        if is_spherical:
            cos_lat = np.cos(np.deg2rad(lat))
            u = u * _DEG2M * cos_lat
            v = v * _DEG2M
        U_profiles[iz, :] = u
        V_profiles[iz, :] = v

    # Convert to z-up ascending for make_profile_sampler
    depth_up = -depth_levels[::-1]
    U_profiles = U_profiles[::-1]
    V_profiles = V_profiles[::-1]

    return _make_profile_sampler(depth_up, U_profiles, V_profiles)


def _position_update(particles, xd_ms, yd_ms, fieldset):
    """Apply an Euler-forward position update to particle displacements.

    Converts drift velocities (m/s) to degree displacements on spherical
    grids, or applies them directly on flat grids.  Mutates
    ``particles.dlon`` and ``particles.dlat`` in place.

    Args:
        particles: Parcels ParticleSet.
        xd_ms: Eastward drift velocity array, shape ``(N,)``, in m/s.
        yd_ms: Northward drift velocity array, shape ``(N,)``, in m/s.
        fieldset: Parcels FieldSet (used to detect spherical vs flat mesh).
    """
    if fieldset.U.grid._mesh == "spherical":
        lat = np.asarray(particles.lat)
        cos_lat = np.cos(np.deg2rad(lat))
        particles.dlon += xd_ms / (_DEG2M * cos_lat) * particles.dt
        particles.dlat += yd_ms / _DEG2M * particles.dt
    else:
        particles.dlon += xd_ms * particles.dt
        particles.dlat += yd_ms * particles.dt


def make_kernel(model):
    """Create a Parcels kernel for any LagrangianMechanicsModel.

    Args:
        model: LagrangianMechanicsModel instance.

    Returns:
        Kernel function ``(particles, fieldset)`` for ``pset.execute``.
    """
    physics = model.physics
    max_depth_fn = getattr(model, '_max_depth', None)
    max_depth = max_depth_fn(physics) if max_depth_fn else 0.0

    def _kernel(particles, fieldset):
        sample_uv = _extract_profiles(particles, fieldset, max_depth)
        t, Y, _ = model.integrate(sample_uv)
        drift_vel = model.drift_velocity(Y[-1])
        _position_update(
            particles, drift_vel[:, 0], drift_vel[:, 1], fieldset,
        )

    return _kernel
