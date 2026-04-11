import functools
import hashlib
import inspect
import os
import pickle
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols


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

    u_stereo: float  # stereographic u
    v_stereo: float  # stereographic v
    xd: float  # buoy x velocity [m/s]
    yd: float  # buoy y velocity [m/s]
    ud_stereo: float  # stereographic u velocity [1/s]
    vd_stereo: float  # stereographic v velocity [1/s]
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
    u_static, v_static = sp.symbols("u_stereo v_stereo", real=True)
    ud_static, vd_static = sp.symbols("ud_stereo vd_stereo", real=True)
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
        "u_stereo": u_static,
        "v_stereo": v_static,
        "xd": xd_static,
        "yd": yd_static,
        "ud_stereo": ud_static,
        "vd_stereo": vd_static,
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


_CACHE_PATH = Path(__file__).resolve().parent / "data" / "eom_cache.pkl"


def _cache_key():
    """Hash of _derive_symbolic source + sympy version for cache invalidation."""
    source = inspect.getsource(_derive_symbolic)
    return hashlib.sha256((source + sp.__version__).encode()).hexdigest()[:16]


def _load_or_derive():
    """Load symbolic EOM from pickle cache, or derive from scratch.

    Returns:
        (M_static, F_static, qdd_exprs, args)
        where qdd_exprs is a tuple of 4 scalar sympy expressions
        representing M^{-1}F (the generalized accelerations).
    """
    if _CACHE_PATH.exists():
        try:
            cached = pickle.loads(_CACHE_PATH.read_bytes())
            if cached.get("key") == _cache_key():
                return cached["M"], cached["F"], cached["qdd"], cached["args"]
        except Exception:
            pass  # corrupt or incompatible pickle — re-derive

    # Miss or stale — re-derive (slow, ~2 min)
    warnings.warn(
        "EOM cache miss — running symbolic derivation (~2 min). "
        "This happens once after code or sympy version changes.",
        stacklevel=2,
    )
    M_static, F_static, args = _derive_symbolic()
    qdd_vec = M_static.LUsolve(F_static)
    qdd_exprs = tuple(qdd_vec[i] for i in range(4))

    data = {
        "key": _cache_key(),
        "M": M_static,
        "F": F_static,
        "qdd": qdd_exprs,
        "args": args,
    }
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = _CACHE_PATH.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(data))
        os.replace(tmp, _CACHE_PATH)
    except OSError:
        pass  # read-only install — skip cache write

    return M_static, F_static, qdd_exprs, args


@functools.lru_cache()
def _get_eom_callables():
    """Load or derive symbolic EOM; lambdify qdd, M, F; return raw callables.

    Returns:
        (qdd_raw, M_raw, F_raw, args, pack_eom_args)
        where qdd_raw returns a tuple of 4 values,
        M_raw returns a tuple of 10 upper-triangle values,
        F_raw returns a tuple of 4 values,
        args is the ordered tuple of sympy symbols,
        and pack_eom_args assembles (physics, state) into positional args.
    """
    M_static, F_static, qdd_exprs, args = _load_or_derive()

    # Extract M upper-triangle elements (symmetric)
    m_exprs = tuple(M_static[i, j] for i in range(4) for j in range(i, 4))

    # Extract F elements
    f_exprs = tuple(F_static[i] for i in range(4))

    # Lambdify with CSE
    qdd_raw = sp.lambdify(args, qdd_exprs, modules="numpy", cse=True)
    M_raw = sp.lambdify(args, m_exprs, modules="numpy", cse=True)
    F_raw = sp.lambdify(args, f_exprs, modules="numpy", cse=True)

    # Build packer once by inspecting the lambda signature
    pack_eom_args = _build_packer(qdd_raw)

    return qdd_raw, M_raw, F_raw, args, pack_eom_args


def _qdd_func(physics, state):
    """Evaluate generalized accelerations qdd = M^{-1}F.

    Internal function — not part of the public API.

    Args:
        physics: DrifterPhysics instance.
        state: EOMState instance (scalar or batch).

    Returns:
        (4,) array for scalar input, (N, 4) array for batch input.
    """
    qdd_raw, _, _, _, pack_eom_args = _get_eom_callables()

    u_arr = np.asarray(state.u_stereo)
    batch_ndim = u_arr.ndim

    result = qdd_raw(*pack_eom_args(physics, state))

    if batch_ndim == 0:
        return np.array(result, dtype=float)
    else:
        return np.column_stack(result)


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
    _, M_raw, _, _, pack_eom_args = _get_eom_callables()

    # Detect batch size from state.u_stereo
    u_arr = np.asarray(state.u_stereo)
    batch_ndim = u_arr.ndim

    # Pack args in the order the lambda expects (derived from its signature)
    M_elems = M_raw(*pack_eom_args(physics, state))

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
    _, _, F_raw, _, pack_eom_args = _get_eom_callables()

    # Detect batch size from state.u_stereo
    u_arr = np.asarray(state.u_stereo)
    batch_ndim = u_arr.ndim

    # Pack args in the order the lambda expects (derived from its signature)
    F_elems = F_raw(*pack_eom_args(physics, state))

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
