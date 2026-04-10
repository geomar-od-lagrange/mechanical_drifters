import functools
import inspect
from pathlib import Path
from typing import NamedTuple

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols
from sympy.printing.numpy import NumPyPrinter


class DrifterPhysics(NamedTuple):
    """Physical constants for a drogued drifter — frozen, set once per instance.

    These are the 9 physical parameters that characterise the drifter geometry
    and drag properties.  They do not change during integration.
    """

    m_b: float  # buoy dry mass [kg]
    m_d: float  # drogue dry mass [kg]
    m_hat_d: float  # drogue buoyancy correction [kg]
    m_tilde_d: float  # drogue added mass [kg]
    m_tilde_b: float  # buoy added mass [kg]
    l: float  # pole length [m]
    g: float  # gravitational acceleration [m/s^2]
    k_b: float  # buoy drag coefficient [kg/m]
    k_d: float  # drogue drag coefficient [kg/m]


class EOMState(NamedTuple):
    """Per-timestep state variables and forcing.

    Fields hold scalars (in rhs) or (N,) arrays (in _rhs_batch).
    Not part of the public API — fields use stereographic coordinates.
    """

    u: float  # stereographic u
    v: float  # stereographic v
    xd: float  # buoy x velocity [m/s]
    yd: float  # buoy y velocity [m/s]
    ud: float  # stereographic u velocity [1/s]
    vd: float  # stereographic v velocity [1/s]
    U_b: float  # current at buoy, east [m/s]
    V_b: float  # current at buoy, north [m/s]
    U_d: float  # current at drogue, east [m/s]
    V_d: float  # current at drogue, north [m/s]


def _build_packer(raw_func):
    """Inspect raw_func's signature, return a pack_eom_args(physics, state) callable.

    Called once (cached). Maps each lambda parameter name to a field in
    DrifterPhysics or EOMState by name. Returns a closure that assembles
    the positional arg tuple from (physics, state).

    Raises KeyError immediately if a parameter name doesn't map to
    any struct field — no silent ordering bugs.
    """
    param_names = list(inspect.signature(raw_func).parameters)
    physics_fields = DrifterPhysics._fields
    state_fields = EOMState._fields

    indices = []  # list of ('p'|'s', field_index)
    for name in param_names:
        if name in physics_fields:
            indices.append(("p", physics_fields.index(name)))
        elif name in state_fields:
            indices.append(("s", state_fields.index(name)))
        else:
            raise KeyError(
                f"Lambda param {name!r} not found in DrifterPhysics or EOMState fields"
            )

    def pack_eom_args(physics, state):
        return tuple(physics[i] if src == "p" else state[i] for src, i in indices)

    return pack_eom_args


def _mag(vec):
    return sp.sqrt(vec.dot(vec))


@functools.lru_cache()
def _derive_symbolic():
    """Derive symbolic M and F in stereographic (u, v) coordinates.

    In spherical coordinates (theta, phi), the azimuthal angle phi is
    undefined at theta=pi: any rotation about the vertical axis represents
    the same physical configuration (drogue hanging straight down).  This
    makes the (theta, phi) equations of motion singular at equilibrium.

    The drogue position relative to the buoy is therefore parameterized via
    stereographic projection of the pole direction.  This avoids the phi
    singularity at theta=pi entirely, with no regularization needed.

    Stereographic identities (projection from south pole onto plane at
    north pole, with s = u^2 + v^2):

        sin(theta) cos(phi) = 4u / (s + 4)
        sin(theta) sin(phi) = 4v / (s + 4)
        cos(theta)          = -(s - 4) / (s + 4)

    These are smooth everywhere, including at the equilibrium (u, v) = (0, 0)
    where theta = pi.

    Returns:
        Tuple ``(M_static, F_static, args)`` where M_static is the 4x4 mass
        matrix and F_static the 4x1 force vector, both with static (non-dynamic)
        symbols substituted for the time-dependent ones, and args is the ordered
        tuple of symbols for lambdification.
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

    # In generalized mass form
    M, F = sp.simplify(sp.linear_eq_to_matrix(eoms, list(qdd)))

    # Substitute dynamic symbols with static ones for lambdify / codegen
    xd_dyn, yd_dyn = x.diff(t), y.diff(t)
    ud_dyn, vd_dyn = u_st.diff(t), v_st.diff(t)

    # Static substitution is required for inspectable lambda signatures.
    # dynamicsymbols (e.g. x(t)) have str() representations like "x(t)" and
    # "Derivative(x(t), t)" which are not valid Python identifiers. lambdify
    # replaces them with anonymous _Dummy_N parameters that cannot be mapped
    # back to struct fields. Substituting plain Symbols with names matching
    # our DrifterPhysics/EOMState field names produces named parameters,
    # enabling signature-based argument packing in _build_packer.
    u_static, v_static = sp.symbols("u v", real=True)
    ud_static, vd_static = sp.symbols("ud vd", real=True)
    x_static, y_static = sp.symbols("x_pos y_pos", real=True)
    xd_static, yd_static = sp.symbols("xd yd", real=True)

    subs = {
        x: x_static,
        y: y_static,
        u_st: u_static,
        v_st: v_static,
        xd_dyn: xd_static,
        yd_dyn: yd_static,
        ud_dyn: ud_static,
        vd_dyn: vd_static,
    }
    M_static = M.subs(subs)
    F_static = F.subs(subs)

    # Derive args tuple from DrifterPhysics and EOMState field names, mapping
    # to symbolic values. The symbol NAME (str of sympy symbol) must match the
    # struct field name exactly, enabling _build_packer to inspect lambda signatures.
    symbol_map = {
        "m_b": m_b,
        "m_d": m_d,
        "m_hat_d": m_hat_d,
        "m_tilde_d": m_tilde_d,
        "m_tilde_b": m_tilde_b,
        "l": l,
        "g": g,
        "k_b": k_b,
        "k_d": k_d,
        "u": u_static,
        "v": v_static,
        "xd": xd_static,
        "yd": yd_static,
        "ud": ud_static,
        "vd": vd_static,
        "U_b": U_b,
        "V_b": V_b,
        "U_d": U_d,
        "V_d": V_d,
    }
    # Collect all symbols that actually appear in M_static and F_static.
    # This is the canonical ordering used for lambdification.
    all_fields = list(DrifterPhysics._fields) + list(EOMState._fields)
    args = tuple(symbol_map[field] for field in all_fields)

    return M_static, F_static, args


_SREPR_PATH = Path(__file__).resolve().parent / "data" / "symbolic_eom.srepr"


def _save_eom_cache(path):
    """Save the symbolic EOM to a .srepr cache file.

    Args:
        path: Path where the cache file should be written.
              Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    M_static, F_static, args = _derive_symbolic()

    arg_names = ",".join(str(s) for s in args)
    content = f"{sp.srepr(M_static)}\n---\n{sp.srepr(F_static)}\n---\n{arg_names}\n"

    path.write_text(content)


@functools.lru_cache()
def _get_eom_callables():
    """Load or derive M_static, F_static, args; apply CSE; return raw lambdified callables.

    Attempts to load the `.srepr` file from `data/symbolic_eom.srepr`. If not found,
    falls back to symbolic derivation. In either case, applies CSE and Approach B exec
    to produce raw lambdified callables.

    Returns:
        (_raw_M_func, _raw_F_func, arg_symbols, pack_eom_args)
        where _raw_M_func/F_func are lambdified callables that accept positional args,
        arg_symbols is the ordered tuple of sympy symbols, and pack_eom_args is a
        closure built via _build_packer that assembles (physics, state) into positional
        args for the raw callables.
    """
    # Try to load .srepr file
    if _SREPR_PATH.exists():
        content = _SREPR_PATH.read_text().strip()
        parts = content.split("---")
        if len(parts) == 3:
            M_srepr, F_srepr, arg_names_csv = parts
            M_static = sp.sympify(M_srepr)
            F_static = sp.sympify(F_srepr)
            arg_names = [n.strip() for n in arg_names_csv.split(",")]
            arg_symbols = tuple(
                sp.Symbol(name, real=True) if "(" not in name else sp.sympify(name)
                for name in arg_names
            )
        else:
            raise ValueError(f"Invalid .srepr format in {_SREPR_PATH}")
    else:
        # Fallback: derive symbolically
        M_static, F_static, arg_symbols = _derive_symbolic()

    # Apply CSE and build raw lambdified functions via Approach B
    _raw_M, _raw_F, args = _apply_cse_and_lambdify(M_static, F_static, arg_symbols)

    # Build packer once by inspecting the lambda signature
    pack_eom_args = _build_packer(_raw_M)

    return _raw_M, _raw_F, args, pack_eom_args


def _apply_cse_and_lambdify(M_static, F_static, args):
    """Apply CSE to M and F; build raw lambdified functions via exec.

    Performs common-subexpression elimination (CSE) on the combined set of
    M and F matrix elements, then generates Python source code for two
    functions (_raw_M and _raw_F) that evaluate them using NumPy.  The
    generated source is exec'd into callable objects.  This avoids the
    overhead of repeated symbolic evaluation while sharing subexpressions
    between M and F.

    Args:
        M_static: 4x4 sympy matrix (static symbols)
        F_static: 4x1 sympy matrix (static symbols)
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
            m_exprs.append(M_static[i, j])
            m_labels.append((i, j))

    f_exprs = [F_static[i] for i in range(4)]

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
    for expr, (i, j) in zip(reduced[: len(m_exprs)], m_labels):
        lines_M.append(f"    M_{i}{j} = {printer.doprint(expr)}")
    ret_names_M = ", ".join(f"M_{i}{j}" for i, j in m_labels)
    lines_M.append(f"    return {ret_names_M}")

    # Build compute_F source
    lines_F = [f"def _raw_F({', '.join(arg_names)}):"]
    for sym, expr in replacements:
        lines_F.append(f"    {sym} = {printer.doprint(expr)}")
    for idx, expr in enumerate(reduced[len(m_exprs) :]):
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


def M_func(physics: DrifterPhysics, state: EOMState):
    """Numerically evaluate the mass matrix M in stereographic coordinates.

    Wraps the raw lambdified callable from _get_eom_callables().
    Detects scalar vs batch input and returns shaped output.

    Args:
        physics: DrifterPhysics instance with physical constants.
        state: EOMState instance with current state and forcing.
            Fields may be scalars or (N,) arrays for batch evaluation.

    Returns:
        4x4 mass matrix:
        - Scalar input: (4,4) array
        - Batch input: (N,4,4) array
    """
    _raw_M, _, _arg_symbols, pack_eom_args = _get_eom_callables()

    # Detect batch size from state.u
    u_arr = np.asarray(state.u)
    batch_ndim = u_arr.ndim

    # Pack args in the order the lambda expects (derived from its signature)
    M_elems = _raw_M(*pack_eom_args(physics, state))

    if batch_ndim == 0:
        # Scalar: assemble (4,4)
        M00, M01, M02, M03, M11, M12, M13, M22, M23, M33 = M_elems
        M = np.array(
            [
                [M00, M01, M02, M03],
                [M01, M11, M12, M13],
                [M02, M12, M22, M23],
                [M03, M13, M23, M33],
            ],
            dtype=float,
        )
    else:
        # Batch: assemble (N, 4, 4)
        N = u_arr.shape[0]
        M = np.zeros((N, 4, 4))
        labels = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 2),
            (2, 3),
            (3, 3),
        ]
        for k, (i, j) in enumerate(labels):
            M[:, i, j] = M[:, j, i] = np.broadcast_to(M_elems[k], N)

    return M


def F_func(physics: DrifterPhysics, state: EOMState):
    """Numerically evaluate the force vector F in stereographic coordinates.

    Wraps the raw lambdified callable from _get_eom_callables().
    Detects scalar vs batch input and returns shaped output.

    Args:
        physics: DrifterPhysics instance with physical constants.
        state: EOMState instance with current state and forcing.
            Fields may be scalars or (N,) arrays for batch evaluation.

    Returns:
        4-element force vector:
        - Scalar input: (4,) array
        - Batch input: (N,4) array
    """
    _, _raw_F, _arg_symbols, pack_eom_args = _get_eom_callables()

    # Detect batch size from state.u
    u_arr = np.asarray(state.u)
    batch_ndim = u_arr.ndim

    # Pack args in the order the lambda expects (derived from its signature)
    F_elems = _raw_F(*pack_eom_args(physics, state))

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
        np.asarray(u, float),
        np.asarray(v, float),
        np.asarray(ud, float),
        np.asarray(vd, float),
    )
    r = np.sqrt(u**2 + v**2)
    theta = _uv_to_theta(u, v)
    phi = np.arctan2(v, u)

    # Chain rule for the stereographic -> spherical conversion:
    #   theta = pi - 2*arctan(r/2),  so  d(theta)/dr = -4/(r^2 + 4).
    #   phi   = arctan2(v, u),       so  d(phi)/du   = -v/r^2,
    #                                     d(phi)/dv   =  u/r^2.
    # Then:
    #   thetad = d(theta)/dr * dr/dt = d(theta)/dr * (u*ud + v*vd) / r
    #   phid   = (u*vd - v*ud) / r^2
    #
    # The `safe` guard prevents division by zero at r=0 (the equilibrium
    # point u=v=0 where theta=pi).  There, the angular direction is
    # undefined; we use the speed magnitude for thetad and zero for phid.
    safe = r > 1e-14
    dtdr = np.where(safe, -4.0 / (r**2 + 4), -1.0)
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

    sec2 = 1.0 / np.cos(half_delta) ** 2
    ud = -sec2 * np.cos(phi) * thetad - 2 * tan_hd * np.sin(phi) * phid
    vd = -sec2 * np.sin(phi) * thetad + 2 * tan_hd * np.cos(phi) * phid

    return u, v, ud, vd
