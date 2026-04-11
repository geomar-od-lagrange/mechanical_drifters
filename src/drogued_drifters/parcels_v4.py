"""Parcels v4 coupling for the drogued-drifter model.

All Parcels-facing code lives here.  The kernel ``DDAdvectEE`` replaces
the old ``make_dd_velocity_interpolator`` factory with a grid-agnostic
approach: it calls ``fieldset.UV.eval()`` per depth level using Parcels'
native VectorField interpolator (handles A-grids, C-grids, curvilinear,
and unstructured grids, including vector rotation and spherical mesh
conversion).
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


def DDAdvectEE(particles, fieldset, *, dd):
    """Parcels kernel: advect particles using drogued-drifter steady-state
    drift velocity (Euler forward).

    Profile extraction uses ``fieldset.UV.eval()`` per depth level,
    leveraging Parcels' native interpolation (handles A-grids, C-grids,
    curvilinear, and unstructured grids, including vector rotation and
    spherical mesh conversion).

    Each call cold-starts the ODE from un-sheared equilibrium.  Spherical/flat
    mesh is auto-detected from ``fieldset.U.grid._mesh``.

    Only samples depths up to the drogue length plus one grid cell
    to avoid unnecessary work on deep levels.

    Args:
        particles: Parcels ParticleSet.
        fieldset: Parcels FieldSet.
        dd: DroguedDrifter instance (bind via ``make_dd_kernel``).

    Usage::

        from drogued_drifters.parcels_v4 import make_dd_kernel

        dd = DroguedDrifter()
        fieldset = FieldSet.from_sgrid_conventions(ds, mesh="spherical")
        pset = ParticleSet(fieldset=fieldset, pclass=Particle, ...)
        pset.execute(
            kernels=[make_dd_kernel(dd), DeleteOOB],
            dt=DT, runtime=RUNTIME,
        )
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

    sample_uv = make_profile_sampler(depth_up, U_profiles, V_profiles)

    # Cold-start from no-shear equilibrium
    xd_ms, yd_ms, _, _ = dd.get_final_drift_batch(sample_uv=sample_uv)

    # Position update (Euler forward)
    if is_spherical:
        cos_lat = np.cos(np.deg2rad(lat))
        particles.dlon += xd_ms / (_DEG2M * cos_lat) * particles.dt
        particles.dlat += yd_ms / _DEG2M * particles.dt
    else:
        particles.dlon += xd_ms * particles.dt
        particles.dlat += yd_ms * particles.dt


def make_dd_kernel(dd, backend="numpy"):
    """Create a Parcels-compatible kernel function for drogued-drifter advection.

    Parcels v4 alpha requires kernel arguments to be ``types.FunctionType``
    (plain ``def`` functions), so ``functools.partial`` cannot be used
    directly.  This factory captures ``dd`` in a closure and returns a
    real function with the ``(particles, fieldset)`` signature.

    Args:
        dd: A :class:`~drogued_drifters.drifter.DroguedDrifter` instance.
        backend: ``"numpy"`` (default) or ``"numba"``.  With ``"numba"``,
            the lambdified qdd function is JIT-compiled with ``numba.njit``
            for faster batch evaluation.  numba must be installed; users
            without numba can still use the default ``"numpy"`` backend.

    Returns:
        A kernel function suitable for ``pset.execute(kernels=[...])``.

    Raises:
        ValueError: If ``backend`` is not ``"numpy"`` or ``"numba"``.

    Usage::

        from drogued_drifters.parcels_v4 import make_dd_kernel

        dd = DroguedDrifter()
        kernel = make_dd_kernel(dd)
        pset.execute(kernels=[kernel, DeleteOOB], dt=DT, runtime=RUNTIME)
    """
    if backend == "numpy":

        def _kernel(particles, fieldset):
            DDAdvectEE(particles, fieldset, dd=dd)

        return _kernel

    # TODO: This logic should live deeper down. We should treat the numpy and numba _qdd eval as two first-class implementations. The asymmetry between the if and the elif branch reveals a problem!
    elif backend == "numba":
        from numba import njit

        from drogued_drifters.lagrange_model import _get_eom_callables

        qdd_raw, _, _, _, pack_eom_args = _get_eom_callables()
        qdd_jit = njit(qdd_raw)

        # Warm up the JIT so the first real call doesn't pay compilation cost.
        _dummy_args = tuple(
            np.ones(1) if i >= 9 else 1.0 for i in range(19)
        )
        qdd_jit(*_dummy_args)

        def _qdd_func_numba(physics, state):
            result = qdd_jit(*pack_eom_args(physics, state))
            if np.ndim(state.u_stereo) == 0:
                return np.array(result, dtype=float)
            return np.column_stack(result)

        dd._qdd_func = _qdd_func_numba

        def _kernel(particles, fieldset):
            DDAdvectEE(particles, fieldset, dd=dd)

        return _kernel

    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Must be 'numpy' or 'numba'."
        )
