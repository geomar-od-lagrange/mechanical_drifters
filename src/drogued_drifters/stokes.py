import numpy as np

# Gravitational acceleration [m/s^2].
_G = 9.81


def compute_stokes_profile(surface_u, surface_v, peak_period, depth_levels):
    """Compute Stokes drift profile at given depth levels.

    Uses the deep-water exponential Stokes drift profile (see Liu et al.,
    2021 [1]_ and Breivik et al., 2016 [2]_) with dispersion
    k = omega^2 / g and exponential decay exp(2kz) for z <= 0.

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
        shallow basins such as the Baltic Sea. Here, "shallow" means depth
        less than half the deep-water wavelength, i.e.
        h < g * T_p^2 / (8 * pi). For long swell (T_p ~ 10 s) this
        threshold is about 39 m; for short wind waves (T_p ~ 4 s) it is
        only about 6 m.

    References:
        .. [1] Liu, Q., et al. (2021). Bulk, Spectral and Deep Water
           Approximations for Stokes Drift: Implications for Coupled Ocean
           Circulation and Surface Wave Models. *Journal of Advances in
           Modeling Earth Systems*, 13, e2020MS002172.
           https://doi.org/10.1029/2020MS002172
        .. [2] Breivik, O., Bidlot, J.-R., & Janssen, P. A. E. M. (2016).
           A Stokes drift approximation based on the Phillips spectrum.
           *Ocean Modelling*, 100, 49-56.
           https://doi.org/10.1016/j.ocemod.2016.01.005

    Args:
        surface_u: Surface Stokes drift eastward component, any shape ``(...)``.
        surface_v: Surface Stokes drift northward component, same shape.
        peak_period: Peak wave period [s], same shape as ``surface_u``.
        depth_levels: Vertical positions [m], positive upward (0 = surface,
            negative = below MSL), shape ``(D,)``.  Must be sorted ascending
            (deepest first, e.g. ``[-20, -10, -5, 0]``).

    Returns:
        Tuple ``(stokes_u, stokes_v)`` arrays of shape ``(D, ...)`` with
        Stokes drift east and north components at each depth level.
    """
    surface_u = np.asarray(surface_u, dtype=float)
    surface_v = np.asarray(surface_v, dtype=float)
    peak_period = np.asarray(peak_period, dtype=float)
    depth_levels = np.asarray(depth_levels, dtype=float)

    omega = 2 * np.pi / peak_period
    k = omega**2 / _G  # deep-water wavenumber [1/m]

    # Broadcast: depth_levels -> (D, 1, 1, ...) to match (...) inputs
    ndim = surface_u.ndim
    z = depth_levels.reshape(-1, *([1] * ndim))  # (D, 1, ..., 1)
    k_b = k[np.newaxis, ...]  # (1, ...)

    # z <= 0 (z-up), so exp(2*k*z) = exp(-2*k*|z|) — correct exponential decay.
    decay = np.exp(2 * k_b * z)  # (D, ...)
    return surface_u[np.newaxis, ...] * decay, surface_v[np.newaxis, ...] * decay
