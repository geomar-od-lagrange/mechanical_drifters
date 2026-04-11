"""Parcels v4 coupling for the drogued-drifter model.

All Parcels-facing code lives here.  The kernel helper ``DDAdvectEE``
advects particles using a grid-agnostic approach: it calls
``fieldset.UV.eval()`` per depth level using Parcels' native VectorField
interpolator (handles A-grids, C-grids, curvilinear, and unstructured
grids, including vector rotation and spherical mesh conversion).
"""

import numpy as np

_DEG2M = 1852.0 * 60.0


def make_profile_sampler(depth_levels, U_profiles, V_profiles):
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
    depth_levels = np.asarray(depth_levels, dtype=float)
    U_profiles = np.asarray(U_profiles, dtype=float)  # (D, N)
    V_profiles = np.asarray(V_profiles, dtype=float)
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


def _extract_profiles(particles, fieldset, dd):
    """Extract velocity profiles from the fieldset and build a depth interpolator.

    Samples the fieldset at each depth level from the surface down to the
    drogue depth (plus one grid cell margin).  Converts Parcels' degree-based
    velocities to m/s on spherical grids.  Returns a ``sample_uv(z)``
    callable suitable for the ODE integrator.

    Args:
        particles: Parcels ParticleSet.
        fieldset: Parcels FieldSet with a ``UV`` VectorField.
        dd: DroguedDrifter instance.

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
    drogue_depth = dd.physics.l

    # Depth levels: surface to first level beyond drogue depth (or all if
    # drogue covers the full water column).  At least 2 for interpolation.
    all_depths = np.asarray(fieldset.U.grid.depth, dtype=float)
    cutoff = min(
        np.searchsorted(all_depths, drogue_depth, side="right") + 1, len(all_depths)
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
        U_profiles[iz] = u
        V_profiles[iz] = v

    # Convert to z-up ascending for make_profile_sampler
    depth_up = -depth_levels[::-1]
    U_profiles = U_profiles[::-1]
    V_profiles = V_profiles[::-1]

    return make_profile_sampler(depth_up, U_profiles, V_profiles)


def _position_update(particles, xd_ms, yd_ms, fieldset):
    """Apply an Euler-forward position update to particle displacements.

    Converts drift velocities (m/s) to degree displacements on spherical
    grids, or applies them directly on flat grids.  Mutates
    ``particles.dlon`` and ``particles.dlat`` in place.

    Args:
        particles: Parcels ParticleSet.  Must have ``dlon``, ``dlat``,
            ``lat``, and ``dt`` attributes.
        xd_ms: Eastward drift velocity array, shape ``(N,)``, in m/s.
        yd_ms: Northward drift velocity array, shape ``(N,)``, in m/s.
        fieldset: Parcels FieldSet (used to detect spherical vs flat mesh).
    """
    is_spherical = fieldset.U.grid._mesh == "spherical"
    if is_spherical:
        lat = np.asarray(particles.lat)
        cos_lat = np.cos(np.deg2rad(lat))
        particles.dlon += xd_ms / (_DEG2M * cos_lat) * particles.dt
        particles.dlat += yd_ms / _DEG2M * particles.dt
    else:
        particles.dlon += xd_ms * particles.dt
        particles.dlat += yd_ms * particles.dt


def DDAdvectEE(particles, fieldset, *, dd):
    """Advect particles using drogued-drifter steady-state drift (Euler forward).

    This is a helper function, not a Parcels kernel itself.  It has the
    non-standard signature ``(particles, fieldset, *, dd)`` and is wrapped
    by ``make_dd_kernel`` to produce a proper ``(particles, fieldset)``
    kernel closure.

    Each call cold-starts the ODE from un-sheared equilibrium.
    Spherical/flat mesh is auto-detected from ``fieldset.U.grid._mesh``.

    The function mutates ``particles.dlon`` and ``particles.dlat`` in place
    (Parcels displacement convention).

    Args:
        particles: Parcels ParticleSet.
        fieldset: Parcels FieldSet with a ``UV`` VectorField.
        dd: DroguedDrifter instance (bind via ``make_dd_kernel``).
    """
    sample_uv = _extract_profiles(particles, fieldset, dd)
    xd_ms, yd_ms, _, _ = dd.get_final_drift_batch(sample_uv=sample_uv)
    _position_update(particles, xd_ms, yd_ms, fieldset)


def make_dd_kernel(dd):
    """Create a Parcels-compatible kernel function for drogued-drifter advection.

    Parcels v4 alpha requires kernel arguments to be ``types.FunctionType``
    (plain ``def`` functions), so ``functools.partial`` cannot be used
    directly.  This factory captures ``dd`` in a closure and returns a
    real function with the ``(particles, fieldset)`` signature.

    The computational backend (numpy or numba) is determined by the
    ``dd.backend`` attribute, which is set at ``DroguedDrifter`` construction
    time.  This function does not select or modify the backend.

    Args:
        dd: A :class:`~drogued_drifters.drifter.DroguedDrifter` instance.

    Returns:
        A kernel function with signature ``(particles, fieldset)`` suitable
        for ``pset.execute(kernels=[...])``.

    Usage::

        from drogued_drifters.parcels_v4 import make_dd_kernel

        dd = DroguedDrifter()  # or DroguedDrifter(backend="numba")
        kernel = make_dd_kernel(dd)
        pset.execute(kernels=[kernel, DeleteOOB], dt=DT, runtime=RUNTIME)
    """

    def _kernel(particles, fieldset):
        DDAdvectEE(particles, fieldset, dd=dd)

    return _kernel
