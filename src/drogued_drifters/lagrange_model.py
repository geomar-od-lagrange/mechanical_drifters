import functools

import sympy as sp
from sympy.physics.mechanics import dynamicsymbols


def _mag(vec):
    return sp.sqrt(vec.dot(vec))


@functools.lru_cache()
def _derive_and_lambdify():
    """Derive equations of motion symbolically, then lambdify M and F.

    This runs once on first use (cached), not at import time.
    """
    t = dynamicsymbols._t

    x, y, z, theta, phi = dynamicsymbols("x y z theta phi")

    m_b, m_d, l, g = sp.symbols("m_b m_d l g", positive=True)
    k_b, k_d = sp.symbols("k_b k_d", positive=True)
    U_b, V_b, U_d, V_d = sp.symbols("U_b V_b U_d V_d", real=True)

    # positions
    r_b = sp.Matrix([x, y, 0])  # buoy
    r = l * sp.Matrix(
        [
            sp.sin(theta) * sp.cos(phi),
            sp.sin(theta) * sp.sin(phi),
            sp.cos(theta),
        ]
    )
    r_d = r_b + r  # drogue

    # velocities
    v_b = r_b.diff(t)
    v_d = r_d.diff(t)

    u_b = sp.Matrix([U_b, V_b, 0])  # external current at buoy
    u_d = sp.Matrix([U_d, V_d, 0])  # external current at drogue

    # drag forces
    F_b = -k_b * _mag(v_b - u_b) * (v_b - u_b)
    F_d = -k_d * _mag(v_d - u_d) * (v_d - u_d)

    # Lagrangian
    T = sp.Rational(1, 2) * m_b * v_b.dot(v_b) + sp.Rational(1, 2) * m_d * v_d.dot(v_d)
    V = m_d * g * r_d[2]
    L = T - V

    # equations of motion: q = [x, y, theta, phi]
    q = sp.Matrix([x, y, theta, phi])
    qd = q.diff(t)
    qdd = qd.diff(t)

    Q = sp.Matrix([r_b.diff(qi).dot(F_b) + r_d.diff(qi).dot(F_d) for qi in q])
    Q = sp.simplify(Q)

    eoms = sp.Matrix(
        [L.diff(qdj).diff(t) - L.diff(qj) - Qj for qj, qdj, Qj in zip(q, qd, Q)]
    )
    eoms = sp.simplify(eoms)

    # TODO: This derivation takes ~15s. Could be sped up by caching the
    # generated numpy code to a .py file instead of re-deriving every session.
    M, F = sp.simplify(sp.linear_eq_to_matrix(eoms, list(qdd)))

    xd, yd, thetad, phid = qd
    args = (
        t,
        x,
        y,
        theta,
        phi,
        xd,
        yd,
        thetad,
        phid,
        m_b,
        m_d,
        l,
        g,
        k_b,
        k_d,
        U_b,
        V_b,
        U_d,
        V_d,
    )

    M_lbd = sp.lambdify(args, M, modules="numpy")
    F_lbd = sp.lambdify(args, F, modules="numpy")
    return M_lbd, F_lbd


# The * forces keyword-only arguments, so callers must use named params.
def M_func(
    *,
    t,
    x,
    y,
    theta,
    phi,
    xd,
    yd,
    thetad,
    phid,
    m_b,
    m_d,
    l,
    g,
    k_b,
    k_d,
    U_b,
    V_b,
    U_d,
    V_d,
):
    """Numerically evaluate the mass matrix M.

    The caller passes named physical quantities. This function handles
    packing them into the generalized coordinate vectors q and qd
    internally.

    Args:
        t: Time [s].
        x: Buoy x position [m].
        y: Buoy y position [m].
        theta: Tether polar angle from vertical [rad]. theta=0 means drogue
            is directly below the buoy, theta=pi means directly above.
        phi: Tether azimuthal angle [rad].
        xd: Time derivative of x [m/s].
        yd: Time derivative of y [m/s].
        thetad: Time derivative of theta [rad/s].
        phid: Time derivative of phi [rad/s].
        m_b: Buoy mass [kg].
        m_d: Drogue mass [kg].
        l: Wire length [m].
        g: Gravitational acceleration [m/s^2].
        k_b: Buoy drag coefficient.
        k_d: Drogue drag coefficient.
        U_b: Eastward current velocity at buoy [m/s].
        V_b: Northward current velocity at buoy [m/s].
        U_d: Eastward current velocity at drogue [m/s].
        V_d: Northward current velocity at drogue [m/s].

    Returns:
        4x4 mass matrix as nested list (use np.array() to convert).
    """
    _M, _ = _derive_and_lambdify()
    return _M(
        t,
        x,
        y,
        theta,
        phi,
        xd,
        yd,
        thetad,
        phid,
        m_b,
        m_d,
        l,
        g,
        k_b,
        k_d,
        U_b,
        V_b,
        U_d,
        V_d,
    )


def F_func(
    *,
    t,
    x,
    y,
    theta,
    phi,
    xd,
    yd,
    thetad,
    phid,
    m_b,
    m_d,
    l,
    g,
    k_b,
    k_d,
    U_b,
    V_b,
    U_d,
    V_d,
):
    """Numerically evaluate the force vector F.

    Args:
        Same as ``M_func``.

    Returns:
        Length-4 force vector as nested list (use np.array() to convert).
    """
    _, _F = _derive_and_lambdify()
    return _F(
        t,
        x,
        y,
        theta,
        phi,
        xd,
        yd,
        thetad,
        phid,
        m_b,
        m_d,
        l,
        g,
        k_b,
        k_d,
        U_b,
        V_b,
        U_d,
        V_d,
    )
