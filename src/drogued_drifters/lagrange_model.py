import functools
from pathlib import Path
from typing import NamedTuple

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy.printing.numpy import NumPyPrinter


class LagrangeParams(NamedTuple):
    """19-parameter tuple for Lagrange model M and F functions.

    Fields are ordered to match the symbolic derivation (lambdify order).
    """
    u: float
    v: float
    xd: float
    yd: float
    ud: float
    vd: float
    m_b: float
    m_d: float
    m_hat_d: float
    m_tilde_d: float
    m_tilde_b: float
    l: float
    g: float
    k_b: float
    k_d: float
    U_b: float
    V_b: float
    U_d: float
    V_d: float


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

    # Lagrangian
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

    # Derive args tuple from LagrangeParams field names, mapping to symbolic values.
    # This ensures the args tuple always matches the NamedTuple definition.
    symbol_map = {
        'u': u_s,
        'v': v_s,
        'xd': xd_s,
        'yd': yd_s,
        'ud': ud_s,
        'vd': vd_s,
        'm_b': m_b,
        'm_d': m_d,
        'm_hat_d': m_hat_d,
        'm_tilde_d': m_tilde_d,
        'm_tilde_b': m_tilde_b,
        'l': l,
        'g': g,
        'k_b': k_b,
        'k_d': k_d,
        'U_b': U_b,
        'V_b': V_b,
        'U_d': U_d,
        'V_d': V_d,
    }
    args = tuple(symbol_map[field] for field in LagrangeParams._fields)

    return M_sub, F_sub, args


_SREPR_PATH = Path(__file__).resolve().parent / "data" / "symbolic_eom.srepr"


def _save_eom_cache(path):
    """Save the symbolic EOM to a .srepr cache file.

    Args:
        path: Path where the cache file should be written.
              Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    M_sub, F_sub, args = _derive_symbolic()

    arg_names = ','.join(str(s) for s in args)
    content = f"{sp.srepr(M_sub)}\n---\n{sp.srepr(F_sub)}\n---\n{arg_names}\n"

    path.write_text(content)


@functools.lru_cache()
def _load_or_derive():
    """Load or derive M_sub, F_sub, args; apply CSE; return raw lambdified callables.

    Attempts to load the `.srepr` file from `data/symbolic_eom.srepr`. If not found,
    falls back to symbolic derivation. In either case, applies CSE and Approach B exec
    to produce raw lambdified callables.

    Returns:
        (_raw_M_func, _raw_F_func, arg_symbols)
        where _raw_M_func/F_func are lambdified callables that accept positional args
        and return results (potentially with batch dimension last).
    """
    # Try to load .srepr file
    if _SREPR_PATH.exists():
        content = _SREPR_PATH.read_text().strip()
        parts = content.split("---")
        if len(parts) == 3:
            M_srepr, F_srepr, arg_names_csv = parts
            M_sub = sp.sympify(M_srepr)
            F_sub = sp.sympify(F_srepr)
            arg_names = arg_names_csv.split(",")
            arg_symbols = tuple(sp.Symbol(name, real=True) if '(' not in name
                               else sp.sympify(name) for name in arg_names)
        else:
            raise ValueError(f"Invalid .srepr format in {_SREPR_PATH}")
    else:
        # Fallback: derive symbolically
        M_sub, F_sub, arg_symbols = _derive_symbolic()

    # Apply CSE and build raw lambdified functions via Approach B
    return _apply_cse_and_lambdify(M_sub, F_sub, arg_symbols)


def _apply_cse_and_lambdify(M_sub, F_sub, args):
    """Apply CSE to M and F; build raw lambdified functions via exec.

    This is Approach B from test_cse_lambdify.py:
    Generate Python function string, exec it, return the functions.

    Args:
        M_sub: 4x4 sympy matrix
        F_sub: 4x1 sympy matrix
        args: tuple of sympy symbols in lambdify order

    Returns:
        (_raw_M_func, _raw_F_func, args)
        where _raw_M_func(*args) returns M elements as tuple
        and _raw_F_func(*args) returns F elements as tuple
        (both vectorized over numpy arrays; batch dim goes last).
    """
    # Extract elements for CSE
    m_exprs = []
    m_labels = []
    for i in range(4):
        for j in range(i, 4):  # Upper triangle only (symmetric)
            m_exprs.append(M_sub[i, j])
            m_labels.append((i, j))

    f_exprs = [F_sub[i] for i in range(4)]

    all_exprs = m_exprs + f_exprs

    # Apply CSE
    replacements, reduced = sp.cse(all_exprs, optimizations="basic")

    # Generate Python code using NumPyPrinter
    printer = NumPyPrinter()
    arg_names = [str(s) for s in args]

    # Build compute_M source
    lines_M = [f"def _raw_M({', '.join(arg_names)}):"]
    for sym, expr in replacements:
        lines_M.append(f"    {sym} = {printer.doprint(expr)}")
    for expr, (i, j) in zip(reduced[:len(m_exprs)], m_labels):
        lines_M.append(f"    M_{i}{j} = {printer.doprint(expr)}")
    ret_names_M = ", ".join(f"M_{i}{j}" for i, j in m_labels)
    lines_M.append(f"    return {ret_names_M}")

    # Build compute_F source
    lines_F = [f"def _raw_F({', '.join(arg_names)}):"]
    for sym, expr in replacements:
        lines_F.append(f"    {sym} = {printer.doprint(expr)}")
    for idx, expr in enumerate(reduced[len(m_exprs):]):
        lines_F.append(f"    F_{idx} = {printer.doprint(expr)}")
    ret_names_F = ", ".join(f"F_{idx}" for idx in range(len(f_exprs)))
    lines_F.append(f"    return {ret_names_F}")

    # Combine source
    source = "\n".join(lines_M) + "\n\n" + "\n".join(lines_F)
    source = source.replace("numpy.", "np.")

    # Exec to create functions
    local_ns = {"np": np}
    exec(source, local_ns)

    _raw_M = local_ns["_raw_M"]
    _raw_F = local_ns["_raw_F"]

    return _raw_M, _raw_F, args


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

    Wraps the raw lambdified callable from _load_or_derive().
    Detects scalar vs batch input and returns shaped output.

    Args:
        u: Stereographic u coordinate (scalar or (N,) array).
        v: Stereographic v coordinate (scalar or (N,) array).
        xd: Time derivative of x [m/s] (scalar or (N,) array).
        yd: Time derivative of y [m/s] (scalar or (N,) array).
        ud: Time derivative of u [1/s] (scalar or (N,) array).
        vd: Time derivative of v [1/s] (scalar or (N,) array).
        m_b: Buoy dry mass [kg] (scalar or (N,) array).
        m_d: Drogue dry mass [kg] (scalar or (N,) array).
        m_hat_d: Buoyancy correction for drogue [kg] (scalar or (N,) array).
        m_tilde_d: Drogue added mass [kg] (scalar or (N,) array).
        m_tilde_b: Buoy added mass [kg] (scalar or (N,) array).
        l: Pole length [m] (scalar or (N,) array).
        g: Gravitational acceleration [m/s^2] (scalar or (N,) array).
        k_b: Buoy drag coefficient [kg/m] (scalar or (N,) array).
        k_d: Drogue drag coefficient [kg/m] (scalar or (N,) array).
        U_b: Eastward current velocity at buoy [m/s] (scalar or (N,) array).
        V_b: Northward current velocity at buoy [m/s] (scalar or (N,) array).
        U_d: Eastward current velocity at drogue [m/s] (scalar or (N,) array).
        V_d: Northward current velocity at drogue [m/s] (scalar or (N,) array).

    Returns:
        4x4 mass matrix:
        - Scalar input: (4,4) array
        - Batch input: (N,4,4) array
    """
    _raw_M, _, arg_symbols = _load_or_derive()
    params = LagrangeParams(
        u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd,
        m_b=m_b, m_d=m_d, m_hat_d=m_hat_d,
        m_tilde_d=m_tilde_d, m_tilde_b=m_tilde_b,
        l=l, g=g, k_b=k_b, k_d=k_d,
        U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
    )

    # Detect batch size from first dynamic argument
    u_arr = np.asarray(u)
    batch_ndim = u_arr.ndim

    # Call raw function with positional args
    M_elems = _raw_M(*params)

    if batch_ndim == 0:
        # Scalar: assemble (4,4)
        M00, M01, M02, M03, M11, M12, M13, M22, M23, M33 = M_elems
        M = np.array([
            [M00, M01, M02, M03],
            [M01, M11, M12, M13],
            [M02, M12, M22, M23],
            [M03, M13, M23, M33],
        ], dtype=float)
    else:
        # Batch: assemble (N, 4, 4)
        N = u_arr.shape[0]
        M = np.zeros((N, 4, 4))
        labels = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
        for k, (i, j) in enumerate(labels):
            M[:, i, j] = M[:, j, i] = np.broadcast_to(M_elems[k], N)

    return M


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

    Wraps the raw lambdified callable from _load_or_derive().
    Detects scalar vs batch input and returns shaped output.

    Args:
        Same as M_func.

    Returns:
        4-element force vector:
        - Scalar input: (4,) array
        - Batch input: (N,4) array
    """
    _, _raw_F, arg_symbols = _load_or_derive()
    params = LagrangeParams(
        u=u, v=v, xd=xd, yd=yd, ud=ud, vd=vd,
        m_b=m_b, m_d=m_d, m_hat_d=m_hat_d,
        m_tilde_d=m_tilde_d, m_tilde_b=m_tilde_b,
        l=l, g=g, k_b=k_b, k_d=k_d,
        U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
    )

    # Detect batch size
    u_arr = np.asarray(u)
    batch_ndim = u_arr.ndim

    # Call raw function
    F_elems = _raw_F(*params)

    if batch_ndim == 0:
        # Scalar: (4,)
        F = np.array(F_elems, dtype=float)
    else:
        # Batch: (N, 4)
        N = u_arr.shape[0]
        F = np.column_stack([np.broadcast_to(f, N) for f in F_elems])

    return F


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
