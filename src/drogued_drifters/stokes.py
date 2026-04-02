import numpy as np


def compute_stokes_profile(surface_u, surface_v, peak_period, depth_levels, g=9.81):
    """Compute Stokes drift profile at given depth levels.

    Uses deep-water dispersion (k = omega² / g) and exponential decay
    (exp(2kz), z <= 0) to extrapolate from the surface Stokes drift to depth.

    For multiple wave partitions, call once per partition and sum the results::

        u_total = np.zeros((len(depth_levels), *surface_u_p1.shape))
        v_total = np.zeros_like(u_total)
        for u_s, v_s, T_p in partitions:
            du, dv = compute_stokes_profile(u_s, v_s, T_p, depth_levels)
            u_total += du
            v_total += dv

    Note:
        Deep-water dispersion overestimates k in shallow water, leading to
        faster decay with depth than the true value. Use with caution in
        shallow basins such as the Baltic Sea.

    Args:
        surface_u: Surface Stokes drift eastward component, any shape ``(...)``.
        surface_v: Surface Stokes drift northward component, same shape.
        peak_period: Peak wave period [s], same shape as ``surface_u``.
        depth_levels: Vertical positions [m], positive upward (0 = surface,
            negative = below MSL), shape ``(D,)``.  Must be sorted ascending
            (deepest first, e.g. ``[-20, -10, -5, 0]``).
        g: Gravitational acceleration [m/s²].

    Returns:
        Tuple ``(stokes_u, stokes_v)`` arrays of shape ``(D, ...)`` with
        Stokes drift east and north components at each depth level.
    """
    surface_u = np.asarray(surface_u, dtype=float)
    surface_v = np.asarray(surface_v, dtype=float)
    peak_period = np.asarray(peak_period, dtype=float)
    depth_levels = np.asarray(depth_levels, dtype=float)

    omega = 2 * np.pi / peak_period
    k = omega**2 / g  # deep-water wavenumber [1/m]

    # Broadcast: depth_levels -> (D, 1, 1, ...) to match (...) inputs
    ndim = surface_u.ndim
    z = depth_levels.reshape(-1, *([1] * ndim))  # (D, 1, ..., 1)
    k_b = k[np.newaxis, ...]  # (1, ...)

    # z <= 0 (z-up), so exp(2*k*z) = exp(-2*k*|z|) — correct exponential decay.
    decay = np.exp(2 * k_b * z)  # (D, ...)
    return surface_u[np.newaxis, ...] * decay, surface_v[np.newaxis, ...] * decay
