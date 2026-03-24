import numpy as np
from scipy.integrate import solve_ivp

from drogued_drifters.lagrange_model import M_func, F_func


def drogue_added_mass(*, rho, w_d, h_d, C_perp_d=np.pi / 4):
    """Drogue added mass: m_tilde_d = C_perp_d * rho * w_d^2 * h_d.

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


def buoy_added_mass(*, rho, d_b, h_b, C_perp_b=1.0):
    """Buoy added mass: m_tilde_b = C_perp_b * rho * pi/4 * d_b^2 * h_b.

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


def drogue_drag_coeff(*, rho, w_d, h_d, C_D_d=1.2):
    """Drogue drag coefficient: k_d = 0.5 * rho * C_D_d * w_d * h_d.

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


def buoy_drag_coeff(*, rho, d_b, h_b, C_D_b=1.0):
    """Buoy drag coefficient: k_b = 0.5 * rho * C_D_b * d_b * h_b.

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


class DroguedDrifter:
    """Simulator for a drogued drifter in ocean currents.

    A drogued drifter consists of a surface buoy connected by a pole of length
    ``l`` to a subsurface drogue. Both experience quadratic drag from the
    surrounding water. The equations of motion are derived from a Lagrangian
    formulation (see ``lagrange_model``).

    The state vector has 8 components: ``[x, y, theta, phi, xd, yd, thetad, phid]``
    where ``(x, y)`` is the buoy position, ``(theta, phi)`` are the pole angles,
    and ``d`` denotes time derivatives.

    Default parameters are for Callies et al. drifter geometry.

    Args:
        m_b: Buoy dry mass [kg].
        m_d: Drogue dry mass [kg].
        m_hat_d: Buoyancy correction for drogue [kg] (mass of displaced water).
        m_tilde_d: Drogue horizontal added mass [kg].
        m_tilde_b: Buoy horizontal added mass [kg].
        l: Pole length [m].
        k_b: Buoy drag coefficient [kg/m].
        k_d: Drogue drag coefficient [kg/m].
        g: Gravitational acceleration [m/s^2].
        get_uv: Callback that returns ocean currents at a given position.
            Must have signature ``get_uv(*, t, z_d, y_b, x_b)`` and return
            ``(U_b, V_b, U_d, V_d)``. If None, uses ``default_uv``.
            Use ``functools.partial`` to bind external data (e.g. an xarray
            dataset) before passing it here.
    """

    def __init__(
        self,
        *,
        m_b=1.0,
        m_d=2.7,
        m_hat_d=1.0,
        m_tilde_d=101.0,
        m_tilde_b=1.9,
        l=3.0,
        k_b=12.0,
        k_d=154.0,
        g=9.81,
        get_uv=None,
        phi_reg_eps=0.1,
        phi_reg_nu=20.0,
    ):
        self.m_b = m_b
        self.m_d = m_d
        self.m_hat_d = m_hat_d
        self.m_tilde_d = m_tilde_d
        self.m_tilde_b = m_tilde_b
        self.l = l
        self.k_b = k_b
        self.k_d = k_d
        self.g = g
        self.phi_reg_eps = phi_reg_eps  # regularization for sin²(theta) in M[3,3]
        self.phi_reg_nu = phi_reg_nu  # phi damping coefficient [kg m² / s]

        if get_uv is not None:
            self.get_uv = get_uv
        else:
            self.get_uv = self.default_uv

    def default_uv(self, *, t, z_d, y_b, x_b):
        """Default velocity callback for testing. Returns uniform currents.

        Args:
            t: Time [s].
            z_d: Drogue depth [m], positive downward.
            y_b: Buoy y position [m].
            x_b: Buoy x position [m].

        Returns:
            Tuple of ``(U_b, V_b, U_d, V_d)`` current velocities [m/s].
        """
        U_b, V_b = 1.0, 1.0
        U_d, V_d = -1.0, -1.0
        return U_b, V_b, U_d, V_d

    def _eval_M_F(self, t, x, y, theta, phi, xd, yd, thetad, phid, currents):
        """Evaluate mass matrix and force vector numerically."""
        U_b, V_b, U_d, V_d = currents
        kwargs = dict(
            t=t,
            x=x,
            y=y,
            theta=theta,
            phi=phi,
            xd=xd,
            yd=yd,
            thetad=thetad,
            phid=phid,
            m_b=self.m_b,
            m_d=self.m_d,
            m_hat_d=self.m_hat_d,
            m_tilde_d=self.m_tilde_d,
            m_tilde_b=self.m_tilde_b,
            l=self.l,
            g=self.g,
            k_b=self.k_b,
            k_d=self.k_d,
            U_b=U_b,
            V_b=V_b,
            U_d=U_d,
            V_d=V_d,
        )
        M = np.array(M_func(**kwargs), dtype=float)
        F = np.array(F_func(**kwargs), dtype=float).reshape(-1)
        return M, F

    def rhs(self, t, y):
        """Right-hand side of the ODE system for ``solve_ivp``.

        Args:
            t: Current time [s].
            y: State vector of length 8:
                ``[x, y, theta, phi, xd, yd, thetad, phid]``.

        Returns:
            Time derivatives of the state vector (length 8).
        """
        x_b, y_b, theta, phi, xd, yd, thetad, phid = y

        z_d = float(max(0.0, -self.l * np.cos(theta)))

        currents = self.get_uv(t=t, z_d=z_d, y_b=y_b, x_b=x_b)

        M, F = self._eval_M_F(
            t, x_b, y_b, theta, phi, xd, yd, thetad, phid, currents
        )

        # Smooth regularization of the phi singularity at theta=pi.
        # M[3,3] ~ sin²(theta) → 0 there, making the system singular.
        # We add eps² to sin²(theta) in M[3,3] and a dissipative torque
        # -nu * phid that activates smoothly near the singularity.
        eps2 = self.phi_reg_eps**2
        st2 = np.sin(theta) ** 2
        M[3, 3] += self.l**2 * (self.m_d + self.m_tilde_d) * eps2
        F[3] -= self.phi_reg_nu * phid * eps2 / (st2 + eps2)

        qdd = np.linalg.solve(M, F)

        return np.array([xd, yd, thetad, phid, *qdd])

    _state_vars = ["x", "y", "theta", "phi", "xd", "yd", "thetad", "phid"]

    def _rhs_batch(self, Y, U_b, V_b, U_d, V_d):
        """Vectorized RHS for N particles.

        Args:
            Y: State array of shape ``(N, 8)``.
            U_b, V_b, U_d, V_d: Current velocities, each of shape ``(N,)``.

        Returns:
            Time derivatives ``dY/dt`` of shape ``(N, 8)``.
        """
        N = Y.shape[0]
        x_b, y_b, theta, phi, xd, yd, thetad, phid = Y.T

        ct, st = np.cos(theta), np.sin(theta)
        cp, sp = np.cos(phi), np.sin(phi)

        l = self.l
        m_d = self.m_d
        m_td = self.m_tilde_d
        md_td = m_d + m_td  # combined drogue + added mass

        # -- Mass matrix M: (N, 4, 4), symmetric ----------------------------
        mt = self.m_b + m_d + self.m_tilde_b + m_td
        a = md_td * l

        M = np.zeros((N, 4, 4))
        M[:, 0, 0] = mt
        M[:, 1, 1] = mt
        M[:, 0, 2] = a * ct * cp;         M[:, 2, 0] = M[:, 0, 2]
        M[:, 0, 3] = -a * st * sp;        M[:, 3, 0] = M[:, 0, 3]
        M[:, 1, 2] = a * ct * sp;         M[:, 2, 1] = M[:, 1, 2]
        M[:, 1, 3] = a * st * cp;         M[:, 3, 1] = M[:, 1, 3]
        M[:, 2, 2] = l**2 * (m_d + m_td * ct**2)
        eps2 = self.phi_reg_eps**2
        M[:, 3, 3] = l**2 * md_td * (st**2 + eps2)

        # -- Force vector F: (N, 4) -----------------------------------------
        # Drogue horizontal velocity
        xd_d = xd + l * (thetad * ct * cp - phid * st * sp)
        yd_d = yd + l * (thetad * ct * sp + phid * st * cp)

        # Slip velocities
        du_b, dv_b = xd - U_b, yd - V_b
        du_d, dv_d = xd_d - U_d, yd_d - V_d

        speed_b = np.sqrt(du_b**2 + dv_b**2)
        speed_d = np.sqrt(du_d**2 + dv_d**2)

        k_b, k_d = self.k_b, self.k_d

        F = np.zeros((N, 4))

        # F[0], F[1]: buoy + drogue drag + centrifugal
        F[:, 0] = (
            -k_b * speed_b * du_b
            - k_d * speed_d * du_d
            + md_td * l * (thetad**2 * st * cp + 2 * thetad * phid * ct * sp + phid**2 * st * cp)
        )
        F[:, 1] = (
            -k_b * speed_b * dv_b
            - k_d * speed_d * dv_d
            + md_td * l * (thetad**2 * st * sp - 2 * thetad * phid * ct * cp + phid**2 * st * sp)
        )

        # F[2]: gravity + drogue drag projected onto theta + Coriolis
        proj_theta = cp * xd + sp * yd + l * thetad * ct - cp * U_d - sp * V_d
        F[:, 2] = (
            (m_d - self.m_hat_d) * self.g * l * st
            - l * ct * k_d * speed_d * proj_theta
            + l**2 * st * ct * (m_td * thetad**2 + md_td * phid**2)
        )

        # F[3]: drogue drag projected onto phi + Coriolis + smooth damping
        proj_phi = -sp * xd + cp * yd + l * phid * st + sp * U_d - cp * V_d
        F[:, 3] = (
            -l * st * k_d * speed_d * proj_phi
            - 2 * md_td * l**2 * thetad * phid * ct * st
            - self.phi_reg_nu * phid * eps2 / (st**2 + eps2)
        )

        # -- Solve M * qdd = F ----------------------------------------------
        qdd = np.linalg.solve(M, F[..., np.newaxis]).squeeze(-1)

        # Assemble dY/dt = [xd, yd, thetad, phid, xdd, ydd, thetadd, phidd]
        dY = np.empty_like(Y)
        dY[:, 0] = xd
        dY[:, 1] = yd
        dY[:, 2] = thetad
        dY[:, 3] = phid
        dY[:, 4:] = qdd
        return dY

    def get_final_drift_batch(
        self,
        *,
        U_b,
        V_b,
        U_d,
        V_d,
        t_span=(0, 120),
        y0=None,
        theta0=0.999 * np.pi,
        conv_tol=1e-4,
        atol=1e-3,
        rtol=1e-3,
    ):
        """Compute steady-state drift for N particles in one ``solve_ivp`` call.

        Stacks all N particles into a single ``(8N,)`` ODE system so that
        ``solve_ivp`` overhead is paid once, and the vectorized RHS
        (``_rhs_batch``) evaluates all particles simultaneously.

        Integration terminates early when the maximum acceleration across all
        particles drops below ``conv_tol`` (global convergence detection).

        Args:
            U_b, V_b: Eastward/northward current at buoy, shape ``(N,)``.
            U_d, V_d: Eastward/northward current at drogue, shape ``(N,)``.
            t_span: Integration window ``(t_start, t_end)`` in seconds.
            y0: Initial state array of shape ``(N, 8)``.  If ``None``,
                starts from rest with ``theta=theta0``.
            theta0: Initial pole angle [rad] (used only when ``y0`` is None).
            conv_tol: Stop when ``max(|xdd|, |ydd|, |thetadd|, |phidd|) < conv_tol``
                across all particles.
            atol: Absolute tolerance for the ODE solver.
            rtol: Relative tolerance for the ODE solver.

        Returns:
            Tuple ``(xd_final, yd_final, theta_final, Y_final)`` where
            the first three are ``(N,)`` arrays and ``Y_final`` is the full
            ``(N, 8)`` state (pass back as ``y0`` for warm-starting).
        """
        N = len(U_b)
        U_b, V_b = np.asarray(U_b, dtype=float), np.asarray(V_b, dtype=float)
        U_d, V_d = np.asarray(U_d, dtype=float), np.asarray(V_d, dtype=float)

        if y0 is not None:
            y0_flat = np.asarray(y0, dtype=float).reshape(N * 8)
        else:
            y0_flat = np.zeros(N * 8)
            y0_flat[2::8] = theta0

        def rhs_flat(t, y_flat):
            Y = y_flat.reshape(N, 8)
            dY = self._rhs_batch(Y, U_b, V_b, U_d, V_d)
            return dY.ravel()

        # Global convergence event: max velocity change rate across all particles.
        # We track |d(xd)/dt| and |d(yd)/dt| (i.e., accelerations in x and y),
        # which measures how fast the drift velocity is still changing.
        def converged(t, y_flat):
            Y = y_flat.reshape(N, 8)
            dY = self._rhs_batch(Y, U_b, V_b, U_d, V_d)
            # xdd = dY[:,4], ydd = dY[:,5]
            max_drift_accel = np.max(np.abs(dY[:, 4:6]))
            return max_drift_accel - conv_tol

        converged.terminal = True
        converged.direction = -1

        sol = solve_ivp(
            rhs_flat, t_span, y0_flat,
            atol=atol, rtol=rtol, events=converged,
        )
        Y_final = sol.y[:, -1].reshape(N, 8)
        return Y_final[:, 4], Y_final[:, 5], Y_final[:, 2], Y_final

    def _get_full_solution(self, t_span, y0, t_eval=None, atol=1e-3, rtol=1e-3):
        """Integrate the equations of motion (raw interface).

        Args:
            t_span: ``(t_start, t_end)`` in seconds.
            y0: Initial state vector of length 8.
            t_eval: Times at which to store the solution.
            atol: Absolute tolerance for the ODE solver.
            rtol: Relative tolerance for the ODE solver.

        Returns:
            ``scipy.integrate.OdeResult`` with fields ``.t`` and ``.y``.
        """
        return solve_ivp(self.rhs, t_span, y0, atol=atol, rtol=rtol, t_eval=t_eval)

    def get_full_solution(
        self,
        *,
        t_span,
        x=0.0,
        y=0.0,
        theta=0.999 * np.pi,
        phi=0.0,
        xd=0.0,
        yd=0.0,
        thetad=0.0,
        phid=0.0,
        t_eval=None,
        atol=1e-3,
        rtol=1e-3,
    ):
        """Integrate the equations of motion over a time span.

        Args:
            t_span: ``(t_start, t_end)`` in seconds.
            x: Initial buoy x position [m].
            y: Initial buoy y position [m].
            theta: Initial pole polar angle [rad] (default ~pi, hanging down).
            phi: Initial pole azimuthal angle [rad].
            xd: Initial buoy x velocity [m/s].
            yd: Initial buoy y velocity [m/s].
            thetad: Initial theta angular velocity [rad/s].
            phid: Initial phi angular velocity [rad/s].
            t_eval: Times at which to store the solution. If None, the solver
                chooses its own time steps.
            atol: Absolute tolerance for the ODE solver.
            rtol: Relative tolerance for the ODE solver.

        Returns:
            ``xarray.Dataset`` with time as coordinate and state variables
            ``x, y, theta, phi, xd, yd, thetad, phid`` as data variables.
        """
        import xarray as xr

        y0 = [x, y, theta, phi, xd, yd, thetad, phid]
        sol = self._get_full_solution(t_span, y0, t_eval=t_eval, atol=atol, rtol=rtol)
        return xr.Dataset(
            {name: ("time", sol.y[i]) for i, name in enumerate(self._state_vars)},
            coords={"time": sol.t},
        )

    def get_final_drift(
        self,
        *,
        t_span,
        x=0.0,
        y=0.0,
        theta=0.999 * np.pi,
        phi=0.0,
        xd=0.0,
        yd=0.0,
        thetad=0.0,
        phid=0.0,
        t_eval=None,
    ):
        """Integrate and return the buoy drift velocity at the end.

        Args:
            t_span: ``(t_start, t_end)`` in seconds. Must be long enough for
                the system to approach steady state.
            x, y, theta, phi, xd, yd, thetad, phid: Initial conditions
                (same as ``get_full_solution``).
            t_eval: Times at which to store the solution.

        Returns:
            ``xarray.Dataset`` with the same structure as ``get_full_solution``.
            The final buoy velocities are ``ds.xd.isel(time=-1)`` and
            ``ds.yd.isel(time=-1)``.
        """
        return self.get_full_solution(
            t_span=t_span,
            x=x,
            y=y,
            theta=theta,
            phi=phi,
            xd=xd,
            yd=yd,
            thetad=thetad,
            phid=phid,
            t_eval=t_eval,
        )
