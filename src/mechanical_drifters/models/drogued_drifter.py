"""DroguedDrifter model: surface buoy + rigid pole + subsurface drogue."""

from typing import NamedTuple

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from sympy.physics.mechanics import dynamicsymbols

from ..base import LagrangianMechanicsModel
from ..coords import _spherical_to_uv, _uv_to_spherical


# ---------- Physics ----------


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

    Physics = DrifterPhysics
    State = EOMState
    n_q = 4
    _drift_velocity_indices = (4, 5)  # xd, yd

    def __init__(self, physics=None, *, backend="numpy", **kwargs):
        """Create a DroguedDrifter.

        Accepts either a DrifterPhysics instance or individual keyword
        parameters::

            DroguedDrifter()                              # defaults
            DroguedDrifter(my_physics)                    # Physics instance
            DroguedDrifter(l=5.0, k_d=200.0)             # override defaults
            DroguedDrifter(backend="numba")               # numba backend
        """
        if physics is None and kwargs:
            defaults = self.default_physics()._asdict()
            defaults.update(kwargs)
            physics = DrifterPhysics(**defaults)
        super().__init__(physics, backend=backend)

    def default_physics(self):
        """Callies et al. (2017) drifter at rho=1025 kg/m^3."""
        return DrifterPhysics(
            m_b=1.0, m_d=2.7, m_hat_d=1.0,
            m_tilde_d=101.0, m_tilde_b=1.9,
            l=3.0, g=9.81, k_b=12.0, k_d=154.0,
        )

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
        all_fields = list(DrifterPhysics._fields) + list(EOMState._fields)
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

        state = EOMState(
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

    def _max_depth(self, physics):
        """Drogue hangs at most one pole-length below surface."""
        return physics.l

    # --- Scalar ODE path (DroguedDrifter-specific) ---

    def _rhs(self, t, y, sample_uv):
        """Scalar RHS for single-particle integration."""
        u_stereo, v_stereo = y[IU], y[IV]
        z_d = float(self._z_eff(np.array([u_stereo]), np.array([v_stereo]))[0])

        U_b, V_b = sample_uv(0.0)
        U_d, V_d = sample_uv(z_d)

        state = EOMState(
            u_stereo, v_stereo, y[IXD], y[IYD], y[IUD], y[IVD],
            U_b, V_b, U_d, V_d,
        )
        qdd = self._qdd_func(self.physics, state)
        return np.array([y[IXD], y[IYD], y[IUD], y[IVD], *qdd])

    # --- DroguedDrifter-specific convenience methods ---

    def get_final_drift_batch(
        self,
        sample_uv,
        *,
        t_span=(0, 120),
        y0=None,
        atol=1e-3,
        rtol=1e-3,
    ):
        """Compute steady-state drift for N particles.

        Wraps ``steady_state_batch`` with internal-to-public coordinate
        conversion. The public state uses spherical angles (theta, phi)
        instead of internal stereographic (u_stereo, v_stereo).

        Args:
            sample_uv: Velocity sampler ``sample_uv(z) -> (U, V)``.
            t_span: Integration window [s].
            y0: Initial state ``(N, 8)`` in public format
                ``[x, y, theta, phi, xd, yd, thetad, phid]``, or None.
            atol, rtol: ODE solver tolerances.

        Returns:
            ``(xd, yd, Y_public, max_accel)`` where Y_public columns are
            ``[x, y, theta, phi, xd, yd, thetad, phid]``.
        """
        # Convert public y0 (spherical) to internal (stereographic)
        y0_internal = None
        if y0 is not None:
            y0_arr = np.asarray(y0, dtype=float).reshape(-1, 8)
            u0, v0, ud0, vd0 = _spherical_to_uv(
                y0_arr[:, 2], y0_arr[:, 3], y0_arr[:, 6], y0_arr[:, 7],
            )
            y0_internal = np.column_stack([
                y0_arr[:, 0], y0_arr[:, 1],
                u0, v0,
                y0_arr[:, 4], y0_arr[:, 5],
                ud0, vd0,
            ])

        drift_vel, Y_final, max_accel = self.steady_state_batch(
            sample_uv, t_span=t_span, y0=y0_internal, atol=atol, rtol=rtol,
        )

        # Convert internal state to public (spherical) coordinates
        theta, phi, thetad, phid = _uv_to_spherical(
            Y_final[:, IU], Y_final[:, IV],
            Y_final[:, IUD], Y_final[:, IVD],
        )
        Y_public = np.column_stack([
            Y_final[:, IX], Y_final[:, IY],
            theta, phi,
            Y_final[:, IXD], Y_final[:, IYD],
            thetad, phid,
        ])

        return drift_vel[:, 0], drift_vel[:, 1], Y_public, max_accel

    def get_full_solution(
        self,
        sample_uv,
        *,
        t_span,
        x=0.0, y=0.0, theta=np.pi, phi=0.0,
        xd=0.0, yd=0.0, thetad=0.0, phid=0.0,
        t_eval=None, atol=1e-3, rtol=1e-3,
    ):
        """Integrate single-particle trajectory, return xr.Dataset.

        Returns a Dataset with spherical coordinates (theta, phi)
        converted from the internal stereographic representation.
        """
        import xarray as xr

        u0, v0, ud0, vd0 = _spherical_to_uv(theta, phi, thetad, phid)
        y0 = [x, y, u0, v0, xd, yd, ud0, vd0]

        sol = solve_ivp(
            lambda t, y_: self._rhs(t, y_, sample_uv),
            t_span, y0, t_eval=t_eval, atol=atol, rtol=rtol,
        )

        theta_arr, phi_arr, thetad_arr, phid_arr = _uv_to_spherical(
            sol.y[IU], sol.y[IV], sol.y[IUD], sol.y[IVD],
        )
        return xr.Dataset(
            {
                "x": ("time", sol.y[IX]),
                "y": ("time", sol.y[IY]),
                "theta": ("time", theta_arr),
                "phi": ("time", phi_arr),
                "xd": ("time", sol.y[IXD]),
                "yd": ("time", sol.y[IYD]),
                "thetad": ("time", thetad_arr),
                "phid": ("time", phid_arr),
            },
            coords={"time": sol.t},
        )

    def get_final_drift(
        self,
        sample_uv,
        *,
        t_span,
        x=0.0, y=0.0, theta=np.pi, phi=0.0,
        xd=0.0, yd=0.0, thetad=0.0, phid=0.0,
    ):
        """Scalar single-particle steady-state drift.

        Returns:
            ``(xd_final, yd_final, max_accel)``
        """
        u0, v0, ud0, vd0 = _spherical_to_uv(theta, phi, thetad, phid)
        y0 = [x, y, u0, v0, xd, yd, ud0, vd0]

        sol = solve_ivp(
            lambda t, y_: self._rhs(t, y_, sample_uv),
            t_span, y0,
        )
        y_final = sol.y[:, -1]

        dy_final = self._rhs(0.0, y_final, sample_uv)
        max_accel = float(max(abs(dy_final[IXD]), abs(dy_final[IYD])))

        return float(y_final[IXD]), float(y_final[IYD]), max_accel
