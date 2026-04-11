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

    elif backend == "numba":
        from numba import njit

        from drogued_drifters.lagrange_model import _get_eom_callables
        import drogued_drifters.drifter as _drifter_mod

        qdd_raw, _, _, _, pack_eom_args = _get_eom_callables()
        qdd_jit = njit(qdd_raw)

        # Warm up the JIT with a single scalar-like call so that the first
        # real call doesn't pay compilation cost during particle advection.
        import drogued_drifters.lagrange_model as _lm

        _physics_dummy = _lm.DrifterPhysics(
            m_b=1.0,
            m_d=2.7,
            m_hat_d=1.0,
            m_tilde_d=101.0,
            m_tilde_b=1.9,
            l=3.0,
            g=9.81,
            k_b=12.0,
            k_d=154.0,
        )
        _state_dummy = _lm.EOMState(
            u_stereo=0.0,
            v_stereo=0.0,
            xd=0.0,
            yd=0.0,
            ud_stereo=0.0,
            vd_stereo=0.0,
            U_b=0.0,
            V_b=0.0,
            U_d=0.0,
            V_d=0.0,
        )
        try:
            qdd_jit(*pack_eom_args(_physics_dummy, _state_dummy))
        except Exception:
            pass  # warmup best-effort; real call will compile if this fails

        def _qdd_func_numba(physics, state):
            u_arr = np.asarray(state.u_stereo)
            batch_ndim = u_arr.ndim
            result = qdd_jit(*pack_eom_args(physics, state))
            if batch_ndim == 0:
                return np.array(result, dtype=float)
            else:
                return np.column_stack(result)

        _original_qdd_func = _drifter_mod._qdd_func

        def _kernel(particles, fieldset):
            _drifter_mod._qdd_func = _qdd_func_numba
            try:
                DDAdvectEE(particles, fieldset, dd=dd)
            finally:
                _drifter_mod._qdd_func = _original_qdd_func

        return _kernel

    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Must be 'numpy' or 'numba'."
        )
