"""Velocity profile interpolation utilities.

Provides :func:`make_profile_sampler`, a factory that builds a fast
``sample_uv(z)`` callable from pre-sampled velocity profiles.
"""

import numpy as np


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
