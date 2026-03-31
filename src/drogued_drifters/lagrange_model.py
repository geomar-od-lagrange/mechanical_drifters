import functools

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols


def _mag(vec):
    return sp.sqrt(vec.dot(vec))


@functools.lru_cache()
def _derive_symbolic():
    """Derive symbolic M and F in stereographic (u, v) coordinates.

    The drogue position relative to the buoy is parameterized directly
    via stereographic projection of the pole direction. This avoids the
    phi singularity at theta=pi entirely, with no regularization needed.

    Stereographic identities (projection from south pole onto plane at
    north pole, with s = u^2 + v^2):

        sin(theta) cos(phi) = 4u / (s + 4)
        sin(theta) sin(phi) = 4v / (s + 4)
        cos(theta)          = -(s - 4) / (s + 4)

    These are smooth everywhere, including at the equilibrium (u, v) = (0, 0)
    where theta = pi.

    Returns:
        Tuple ``(M_sub, F_sub, args)`` where M_sub is the 4x4 mass matrix
        and F_sub the 4x1 force vector, both in static (non-dynamic) symbols,
        and args is the ordered tuple of symbols for lambdification.
    """
    t = dynamicsymbols._t

    x, y = dynamicsymbols("x y")
    u_st, v_st = dynamicsymbols("u_st v_st")

    m_b, m_d, l, g = sp.symbols("m_b m_d l g", positive=True)
    m_hat_d = sp.Symbol("m_hat_d", positive=True)
    m_tilde_d = sp.Symbol("m_tilde_d", positive=True)
    m_tilde_b = sp.Symbol("m_tilde_b", positive=True)
    k_b, k_d = sp.symbols("k_b k_d", positive=True)
    U_b, V_b, U_d, V_d = sp.symbols("U_b V_b U_d V_d", real=True)

    # Stereographic identities — smooth at the origin
    s = u_st**2 + v_st**2
    denom = s + 4
    sin_theta_cos_phi = 4 * u_st / denom
    sin_theta_sin_phi = 4 * v_st / denom
    cos_theta = (s - 4) / denom

    # Drogue position relative to buoy
    r_b = sp.Matrix([x, y, 0])
    r = l * sp.Matrix([sin_theta_cos_phi, sin_theta_sin_phi, cos_theta])
    r_d = r_b + r

    # Velocities
    v_b = r_b.diff(t)
    v_d = r_d.diff(t)
    v_d_h = sp.Matrix([v_d[0], v_d[1], 0])

    u_b_vec = sp.Matrix([U_b, V_b, 0])
    u_d_vec = sp.Matrix([U_d, V_d, 0])

    # Drag forces
    F_b = -k_b * _mag(v_b - u_b_vec) * (v_b - u_b_vec)
    F_d = -k_d * _mag(v_d_h - u_d_vec) * (v_d_h - u_d_vec)

    # Kinetic energy
    T = (
        sp.Rational(1, 2) * m_d * v_d.dot(v_d)
        + sp.Rational(1, 2) * m_tilde_d * v_d_h.dot(v_d_h)
        + sp.Rational(1, 2) * (m_b + m_tilde_b) * v_b.dot(v_b)
    )

    # Potential energy
    V = (m_d - m_hat_d) * g * r_d[2]

    L = T - V

    # Generalized coordinates: q = [x, y, u_st, v_st]
    q = sp.Matrix([x, y, u_st, v_st])
    qd = q.diff(t)
    qdd = qd.diff(t)

    # Generalized forces
    Q = sp.Matrix([r_b.diff(qi).dot(F_b) + r_d.diff(qi).dot(F_d) for qi in q])
    Q = sp.simplify(Q)

    # Euler-Lagrange equations
    eoms = sp.Matrix(
        [L.diff(qdj).diff(t) - L.diff(qj) - Qj for qj, qdj, Qj in zip(q, qd, Q)]
    )
    eoms = sp.simplify(eoms)

    M, F = sp.simplify(sp.linear_eq_to_matrix(eoms, list(qdd)))

    # Substitute dynamic symbols with static ones for lambdify / codegen
    xd, yd = x.diff(t), y.diff(t)
    ud, vd = u_st.diff(t), v_st.diff(t)

    u_s, v_s = sp.symbols("u_s v_s", real=True)
    ud_s, vd_s = sp.symbols("ud_s vd_s", real=True)
    x_s, y_s = sp.symbols("x_s y_s", real=True)
    xd_s, yd_s = sp.symbols("xd_s yd_s", real=True)

    subs = {
        x: x_s, y: y_s, u_st: u_s, v_st: v_s,
        xd: xd_s, yd: yd_s, ud: ud_s, vd: vd_s,
    }
    M_sub = M.subs(subs)
    F_sub = F.subs(subs)

    args = (
        u_s,
        v_s,
        xd_s,
        yd_s,
        ud_s,
        vd_s,
        m_b,
        m_d,
        m_hat_d,
        m_tilde_d,
        m_tilde_b,
        l,
        g,
        k_b,
        k_d,
        U_b,
        V_b,
        U_d,
        V_d,
    )

    return M_sub, F_sub, args


@functools.lru_cache()
def _derive_and_lambdify():
    """Derive and lambdify M and F (cached).

    This runs the full sympy derivation once on first use.
    """
    M_sub, F_sub, args = _derive_symbolic()
    M_lbd = sp.lambdify(args, M_sub, modules="numpy")
    F_lbd = sp.lambdify(args, F_sub, modules="numpy")
    return M_lbd, F_lbd


def M_func(
    *,
    u,
    v,
    xd,
    yd,
    ud,
    vd,
    m_b,
    m_d,
    m_hat_d,
    m_tilde_d,
    m_tilde_b,
    l,
    g,
    k_b,
    k_d,
    U_b,
    V_b,
    U_d,
    V_d,
):
    """Numerically evaluate the mass matrix M in stereographic coordinates.

    Uses the lambdified sympy derivation. Primarily for verification;
    production code uses ``_generated_eom.compute_M``.

    Args:
        u: Stereographic u coordinate (dimensionless).
        v: Stereographic v coordinate (dimensionless).
        xd: Time derivative of x [m/s].
        yd: Time derivative of y [m/s].
        ud: Time derivative of u [1/s].
        vd: Time derivative of v [1/s].
        m_b: Buoy dry mass [kg].
        m_d: Drogue dry mass [kg].
        m_hat_d: Buoyancy correction for drogue [kg].
        m_tilde_d: Drogue added mass [kg] (horizontal).
        m_tilde_b: Buoy added mass [kg] (horizontal).
        l: Pole length [m].
        g: Gravitational acceleration [m/s^2].
        k_b: Buoy drag coefficient [kg/m].
        k_d: Drogue drag coefficient [kg/m].
        U_b: Eastward current velocity at buoy [m/s].
        V_b: Northward current velocity at buoy [m/s].
        U_d: Eastward current velocity at drogue [m/s].
        V_d: Northward current velocity at drogue [m/s].

    Returns:
        4x4 mass matrix as nested list (use np.array() to convert).
    """
    _M, _ = _derive_and_lambdify()
    return _M(
        u, v, xd, yd, ud, vd,
        m_b, m_d, m_hat_d, m_tilde_d, m_tilde_b,
        l, g, k_b, k_d, U_b, V_b, U_d, V_d,
    )


def F_func(
    *,
    u,
    v,
    xd,
    yd,
    ud,
    vd,
    m_b,
    m_d,
    m_hat_d,
    m_tilde_d,
    m_tilde_b,
    l,
    g,
    k_b,
    k_d,
    U_b,
    V_b,
    U_d,
    V_d,
):
    """Numerically evaluate the force vector F in stereographic coordinates.

    Uses the lambdified sympy derivation. Primarily for verification;
    production code uses ``_generated_eom.compute_F``.

    Args:
        Same as ``M_func``.

    Returns:
        Length-4 force vector as nested list (use np.array() to convert).
    """
    _, _F = _derive_and_lambdify()
    return _F(
        u, v, xd, yd, ud, vd,
        m_b, m_d, m_hat_d, m_tilde_d, m_tilde_b,
        l, g, k_b, k_d, U_b, V_b, U_d, V_d,
    )


def _uv_to_theta(u, v):
    """Convert stereographic (u, v) to polar angle theta.

    Args:
        u, v: Stereographic coordinates (scalar or array).

    Returns:
        theta: Polar angle [rad], where theta=pi is drogue hanging down.
    """
    r = np.sqrt(u**2 + v**2)
    delta = 2 * np.arctan2(r, 2)
    return np.pi - delta


def _uv_to_spherical(u, v, ud, vd):
    """Convert stereographic (u, v, ud, vd) to spherical (theta, phi, thetad, phid).

    Args:
        u, v: Stereographic coordinates (scalar or array).
        ud, vd: Stereographic velocities (scalar or array).

    Returns:
        (theta, phi, thetad, phid) tuple.
    """
    u, v, ud, vd = (
        np.asarray(u, float), np.asarray(v, float),
        np.asarray(ud, float), np.asarray(vd, float),
    )
    r = np.sqrt(u**2 + v**2)
    theta = _uv_to_theta(u, v)
    phi = np.arctan2(v, u)

    safe = r > 1e-14
    # d(theta)/dr = -4/(r^2+4); d(theta)/du = d(theta)/dr * u/r; etc.
    dtdr = np.where(safe, -4.0 / (r**2 + 4), -1.0)
    # At r=0, direction is ambiguous; use magnitude as a reasonable fallback.
    thetad = np.where(safe, dtdr * (u * ud + v * vd) / r, -np.sqrt(ud**2 + vd**2))
    phid = np.where(safe, (u * vd - v * ud) / r**2, 0.0)

    return theta, phi, thetad, phid


def _spherical_to_uv(theta, phi, thetad, phid):
    """Convert spherical (theta, phi, thetad, phid) to stereographic (u, v, ud, vd).

    Args:
        theta: Polar angle [rad] (theta=pi is drogue down).
        phi: Azimuthal angle [rad].
        thetad, phid: Angular velocities [rad/s].

    Returns:
        (u, v, ud, vd) tuple.
    """
    theta, phi = np.asarray(theta, float), np.asarray(phi, float)
    thetad, phid = np.asarray(thetad, float), np.asarray(phid, float)
    delta = np.pi - theta
    half_delta = delta / 2
    tan_hd = np.tan(half_delta)
    u = 2 * tan_hd * np.cos(phi)
    v = 2 * tan_hd * np.sin(phi)

    sec2 = 1.0 / np.cos(half_delta)**2
    ud = -sec2 * np.cos(phi) * thetad - 2 * tan_hd * np.sin(phi) * phid
    vd = -sec2 * np.sin(phi) * thetad + 2 * tan_hd * np.cos(phi) * phid

    return u, v, ud, vd
