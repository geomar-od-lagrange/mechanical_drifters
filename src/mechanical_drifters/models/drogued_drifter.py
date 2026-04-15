"""DroguedDrifter model: surface buoy + rigid pole + subsurface drogue."""

from typing import NamedTuple

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from ..base import LagrangianMechanicsModel


# ---------- Coordinate helpers (stereographic <-> spherical) ----------


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


# ---------- Physics ----------


class DroguedDrifterPhysics(NamedTuple):
    """Physical constants for a drogued drifter — frozen, set once per instance.

    These are the 9 physical parameters that characterise the drifter geometry
    and drag properties.  They do not change during integration.

    Default values correspond to the Callies et al. (2017) drifter design.
    ``DroguedDrifterPhysics()`` gives these defaults.
    """

    m_b: float = 1.0  # buoy dry mass [kg]
    m_d: float = 2.7  # drogue dry mass [kg]
    m_hat_d: float = 1.0  # drogue buoyancy correction [kg]
    m_tilde_d: float = 101.0  # drogue added mass [kg]
    m_tilde_b: float = 1.9  # buoy added mass [kg]
    l: float = 3.0  # pole length [m]
    g: float = 9.81  # gravitational acceleration [m/s^2]
    k_b: float = 12.0  # buoy drag coefficient [kg/m]
    k_d: float = 154.0  # drogue drag coefficient [kg/m]


class _State(NamedTuple):
    """Per-timestep state variables and forcing.

    Fields hold scalars (in rhs) or (N,) arrays (in _rhs_batch).
    Not part of the public API — fields use stereographic coordinates.
    """

    u_stereo: float | np.ndarray  # stereographic u
    v_stereo: float | np.ndarray  # stereographic v
    xd: float | np.ndarray  # buoy x velocity [m/s]
    yd: float | np.ndarray  # buoy y velocity [m/s]
    ud_stereo: float | np.ndarray  # stereographic u velocity [1/s]
    vd_stereo: float | np.ndarray  # stereographic v velocity [1/s]
    U_b: float | np.ndarray  # current at buoy, east [m/s]
    V_b: float | np.ndarray  # current at buoy, north [m/s]
    U_d: float | np.ndarray  # current at drogue, east [m/s]
    V_d: float | np.ndarray  # current at drogue, north [m/s]


# Backwards-compatible alias
DroguedDrifterState = _State


# ---------- Drag / added-mass helpers ----------


def drogue_horizontal_added_mass(*, rho, w_d, h_d, C_perp_d=np.pi / 4):
    """Horizontal added mass of the drogue: m_tilde_d = C_perp_d * rho * w_d^2 * h_d.

    The drogue is a cross of two vertical plates. For any horizontal
    acceleration, one plate is broadside. The added mass equals the mass
    of a cylindrical fluid volume of diameter w_d and height h_d.

    Args:
        rho: Sea water density [kg/m^3].
        w_d: Drogue plate width [m].
        h_d: Drogue plate height [m].
        C_perp_d: Added-mass coefficient for flat plate (default pi/4).

    Returns:
        Drogue added mass m_tilde_d [kg].
    """
    return C_perp_d * rho * w_d**2 * h_d


def buoy_horizontal_added_mass(*, rho, d_b, h_b, C_perp_b=1.0):
    """Horizontal added mass of the buoy: m_tilde_b = C_perp_b * rho * pi/4 * d_b^2 * h_b.

    The buoy is a nearly fully submerged cylinder accelerated broadside.

    Args:
        rho: Sea water density [kg/m^3].
        d_b: Buoy diameter [m].
        h_b: Buoy submerged height [m].
        C_perp_b: Added-mass coefficient for cylinder (default 1.0).

    Returns:
        Buoy added mass m_tilde_b [kg].
    """
    return C_perp_b * rho * np.pi / 4 * d_b**2 * h_b


def drogue_horizontal_drag_coeff(*, rho, w_d, h_d, C_D_d=1.2):
    """Horizontal drag coefficient of the drogue: k_d = 0.5 * rho * C_D_d * w_d * h_d.

    By cross symmetry, one plate is broadside for any horizontal flow,
    so the reference area is w_d * h_d.

    Args:
        rho: Sea water density [kg/m^3].
        w_d: Drogue plate width [m].
        h_d: Drogue plate height [m].
        C_D_d: Drag coefficient for flat plate (default 1.2).

    Returns:
        Drogue drag coefficient k_d [kg/m].
    """
    return 0.5 * rho * C_D_d * w_d * h_d


def buoy_horizontal_drag_coeff(*, rho, d_b, h_b, C_D_b=1.0):
    """Horizontal drag coefficient of the buoy: k_b = 0.5 * rho * C_D_b * d_b * h_b.

    The projected area is d_b * h_b (cylinder broadside to flow).

    Args:
        rho: Sea water density [kg/m^3].
        d_b: Buoy diameter [m].
        h_b: Buoy submerged height [m].
        C_D_b: Drag coefficient for cylinder (default 1.0).

    Returns:
        Buoy drag coefficient k_b [kg/m].
    """
    return 0.5 * rho * C_D_b * d_b * h_b


# ---------- Internal state vector layout ----------
# [x, y, u_stereo, v_stereo, xd, yd, ud_stereo, vd_stereo]
IX, IY, IU, IV, IXD, IYD, IUD, IVD = range(8)


# ---------- The model ----------


class DroguedDrifter(LagrangianMechanicsModel):
    """A surface buoy connected by a rigid pole to a subsurface drogue.

    Both bodies experience quadratic drag from the ocean current at their
    respective depths. The equations of motion are derived from a
    Lagrangian formulation with 4 generalized coordinates:
    buoy position (x, y) and stereographic pole direction (u, v).

    State vector: [x, y, u_stereo, v_stereo, xd, yd, ud_stereo, vd_stereo]
    """

    Physics = DroguedDrifterPhysics
    State = _State
    n_q = 4
    state_names = ("x", "y", "theta", "phi", "xd", "yd", "thetad", "phid")

    def __init__(self, physics=DroguedDrifterPhysics(), *, backend="numpy", **kwargs):
        """Create a DroguedDrifter.

        Accepts either a DroguedDrifterPhysics instance or individual keyword
        parameters::

            DroguedDrifter()                              # defaults
            DroguedDrifter(my_physics)                    # Physics instance
            DroguedDrifter(l=5.0, k_d=200.0)             # override defaults
            DroguedDrifter(backend="numba")               # numba backend
        """
        if kwargs:
            defaults = physics._asdict()
            defaults.update(kwargs)
            physics = DroguedDrifterPhysics(**defaults)
        super().__init__(physics, backend=backend)

    # --- Symbolic derivation (the physics) ---

    def _derive_symbolic(self):
        """Derive M, F for the buoy-pole-drogue system in stereographic coords.

        The pole direction is parameterized via stereographic projection
        from the south pole, avoiding the azimuthal singularity at
        theta=pi (drogue hanging straight down).

        Returns:
            (M_static, F_static, args) where M is 4x4, F is 4x1, and args
            is the ordered symbol tuple matching Physics + State field names.
        """

        def _sym_norm(vec):
            return sp.sqrt(vec.dot(vec))

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
        F_b = -k_b * _sym_norm(v_b - u_b_vec) * (v_b - u_b_vec)
        F_d = -k_d * _sym_norm(v_d_h - u_d_vec) * (v_d_h - u_d_vec)

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
            [L.diff(qdj).diff(t) - L.diff(qj) - Qj
             for qj, qdj, Qj in zip(q, qd, Q)]
        )
        eoms = sp.simplify(eoms)

        # In generalized mass form
        M, F = sp.simplify(sp.linear_eq_to_matrix(eoms, list(qdd)))

        # Substitute dynamic symbols with static ones for lambdify / codegen
        xd_dyn, yd_dyn = x.diff(t), y.diff(t)
        ud_dyn, vd_dyn = u_st.diff(t), v_st.diff(t)

        u_static, v_static = sp.symbols("u_stereo v_stereo", real=True)
        ud_static, vd_static = sp.symbols("ud_stereo vd_stereo", real=True)
        x_static, y_static = sp.symbols("x_pos y_pos", real=True)
        xd_static, yd_static = sp.symbols("xd yd", real=True)

        subs = {
            x: x_static, y: y_static,
            u_st: u_static, v_st: v_static,
            xd_dyn: xd_static, yd_dyn: yd_static,
            ud_dyn: ud_static, vd_dyn: vd_static,
        }
        M_static = M.subs(subs)
        F_static = F.subs(subs)

        symbol_map = {
            "m_b": m_b, "m_d": m_d, "m_hat_d": m_hat_d,
            "m_tilde_d": m_tilde_d, "m_tilde_b": m_tilde_b,
            "l": l, "g": g, "k_b": k_b, "k_d": k_d,
            "u_stereo": u_static, "v_stereo": v_static,
            "xd": xd_static, "yd": yd_static,
            "ud_stereo": ud_static, "vd_stereo": vd_static,
            "U_b": U_b, "V_b": V_b, "U_d": U_d, "V_d": V_d,
        }
        all_fields = list(DroguedDrifterPhysics._fields) + list(_State._fields)
        args = tuple(symbol_map[field] for field in all_fields)

        return M_static, F_static, args

    # --- The RHS (the hot path) ---

    def _z_eff(self, u, v):
        """Effective drogue depth from stereographic coordinates.

        Args:
            u, v: Stereographic coordinates, scalar or ``(N,)`` array.

        Returns:
            z_eff: Drogue vertical position [m], positive upward (non-positive),
                shape ``(N,)``.  At equilibrium (pole vertical) returns ``-l``.
                Clamped to ``<= 0``.
        """
        s = u**2 + v**2
        cos_theta = (s - 4) / (s + 4)
        return np.minimum(0.0, self.physics.l * cos_theta)

    def _rhs_batch(self, Y, sample_uv):
        """Vectorized RHS for N particles.

        Args:
            Y: (N, 8) state array.
            sample_uv: callable(z) -> (U, V) for (N,) arrays.

        Returns:
            dY: (N, 8) derivatives.
        """
        N = Y.shape[0]
        u_stereo = Y[:, IU]
        v_stereo = Y[:, IV]
        xd = Y[:, IXD]
        yd = Y[:, IYD]
        ud_stereo = Y[:, IUD]
        vd_stereo = Y[:, IVD]

        # Velocity at buoy (surface) and drogue (effective depth)
        U_b, V_b = sample_uv(np.zeros(N))
        U_d, V_d = sample_uv(self._z_eff(u_stereo, v_stereo))

        state = _State(
            u_stereo=u_stereo, v_stereo=v_stereo,
            xd=xd, yd=yd,
            ud_stereo=ud_stereo, vd_stereo=vd_stereo,
            U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
        )

        qdd = self._qdd_func(self.physics, state)

        bad = ~np.isfinite(qdd).all(axis=1)
        if np.any(bad):
            qdd[bad] = 0.0

        dY = np.empty_like(Y)
        dY[:, IX] = xd
        dY[:, IY] = yd
        dY[:, IU] = ud_stereo
        dY[:, IV] = vd_stereo
        dY[:, IXD:] = qdd
        return dY

    @property
    def _max_depth(self):
        """Drogue hangs at most one pole-length below surface."""
        return self.physics.l

    def drift_velocity(self, Y):
        """Extract drift velocity from state array.

        Args:
            Y: State array, shape ``(N, state_size)``.

        Returns:
            Drift velocity array, shape ``(N, 2)``.
        """
        return Y[:, [IXD, IYD]]

    # --- Coordinate conversion helpers ---

    def _from_public_state(self, Y_public):
        """Convert public state (spherical) to internal (stereographic).

        Args:
            Y_public: ``(N, 8)`` array with columns
                ``[x, y, theta, phi, xd, yd, thetad, phid]``.

        Returns:
            ``(N, 8)`` array with columns
            ``[x, y, u_stereo, v_stereo, xd, yd, ud_stereo, vd_stereo]``.
        """
        Y = np.asarray(Y_public, dtype=float).reshape(-1, 8)
        u, v, ud, vd = _spherical_to_uv(Y[:, 2], Y[:, 3], Y[:, 6], Y[:, 7])
        return np.column_stack([
            Y[:, 0], Y[:, 1], u, v, Y[:, 4], Y[:, 5], ud, vd,
        ])

    def _to_public_state(self, Y_internal):
        """Convert internal state (stereographic) to public (spherical).

        Args:
            Y_internal: ``(N, 8)`` array with columns
                ``[x, y, u_stereo, v_stereo, xd, yd, ud_stereo, vd_stereo]``.

        Returns:
            ``(N, 8)`` array with columns
            ``[x, y, theta, phi, xd, yd, thetad, phid]``.
        """
        Y = np.asarray(Y_internal, dtype=float).reshape(-1, 8)
        theta, phi, thetad, phid = _uv_to_spherical(
            Y[:, IU], Y[:, IV], Y[:, IUD], Y[:, IVD],
        )
        return np.column_stack([
            Y[:, IX], Y[:, IY], theta, phi,
            Y[:, IXD], Y[:, IYD], thetad, phid,
        ])

    # --- Override integrate for public coords in/out ---

    def integrate(
        self,
        sample_uv,
        *,
        t_span=(0, 120),
        y0=None,
        t_eval=None,
        atol=1e-3,
        rtol=1e-3,
    ):
        """Integrate the ODE with public (spherical) coords in/out.

        Converts spherical y0 to stereographic on entry, runs the base
        integrator, and converts the result back to spherical.

        Args:
            sample_uv: Velocity sampler.
            t_span: Integration window [s].
            y0: Initial state ``(N, 8)`` in public format
                ``[x, y, theta, phi, xd, yd, thetad, phid]``, or None.
            t_eval: Times at which to store the solution.
            atol, rtol: ODE solver tolerances.

        Returns:
            Tuple ``(t, Y, max_acceleration)`` where Y is in public coords.
        """
        y0_internal = None
        if y0 is not None:
            y0_internal = self._from_public_state(y0)

        t, Y_internal, max_accel = super().integrate(
            sample_uv, t_span=t_span, y0=y0_internal,
            t_eval=t_eval, atol=atol, rtol=rtol,
        )

        # Convert each time step from internal to public coords
        T, N, ss = Y_internal.shape
        Y_public = np.empty_like(Y_internal)
        for i in range(T):
            Y_public[i] = self._to_public_state(Y_internal[i])

        return t, Y_public, max_accel
