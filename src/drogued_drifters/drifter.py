import numpy as np
from scipy.integrate import solve_ivp

from drogued_drifters._generated_eom import compute_F, compute_M


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
    u, v, ud, vd = np.asarray(u, float), np.asarray(v, float), np.asarray(ud, float), np.asarray(vd, float)
    r = np.sqrt(u**2 + v**2)
    theta = _uv_to_theta(u, v)
    phi = np.arctan2(v, u)

    # Velocity Jacobian: d(theta)/d(u,v) and d(phi)/d(u,v)
    safe = r > 1e-14
    # d(delta)/dr = 4/(r^2+4), and delta = 2*arctan(r/2)
    # d(theta)/dr = -d(delta)/dr = -4/(r^2+4)
    # d(theta)/du = d(theta)/dr * u/r, d(theta)/dv = d(theta)/dr * v/r
    # d(phi)/du = -v/r^2, d(phi)/dv = u/r^2
    dtdr = np.where(safe, -4.0 / (r**2 + 4), -1.0)  # limit: -4/4 = -1
    thetad = np.where(safe, dtdr * (u * ud + v * vd) / r, -(ud + vd) * 0.0)
    # At r=0, thetad from ud alone: dtdr * ud (along u-axis), but direction
    # is ambiguous. Set to magnitude:
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

    # dr/d(delta) = 1/cos^2(delta/2), d(delta)/d(theta) = -1
    # du/d(theta) = -cos(phi)/cos^2(delta/2)
    # du/d(phi)   = -2*tan(delta/2)*sin(phi)
    sec2 = 1.0 / np.cos(half_delta)**2
    ud = -sec2 * np.cos(phi) * thetad - 2 * tan_hd * np.sin(phi) * phid
    vd = -sec2 * np.sin(phi) * thetad + 2 * tan_hd * np.cos(phi) * phid

    return u, v, ud, vd


class DroguedDrifter:
    """Simulator for a drogued drifter in ocean currents.

    A drogued drifter consists of a surface buoy connected by a pole of length
    ``l`` to a subsurface drogue. Both experience quadratic drag from the
    surrounding water. The equations of motion are derived from a Lagrangian
    formulation (see ``lagrange_model``).

    The internal state vector has 8 components:
    ``[x, y, u, v, xd, yd, ud, vd]``
    where ``(x, y)`` is the buoy position and ``(u, v)`` are stereographic
    coordinates for the pole direction (equilibrium at origin).

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

    def _params(self):
        """Return the physical parameter dict for compute_M / compute_F."""
        return dict(
            m_b=self.m_b,
            m_d=self.m_d,
            m_hat_d=self.m_hat_d,
            m_tilde_d=self.m_tilde_d,
            m_tilde_b=self.m_tilde_b,
            l=self.l,
            g=self.g,
            k_b=self.k_b,
            k_d=self.k_d,
        )

    def _eval_M_F(self, t, x, y, u, v, xd, yd, ud, vd, currents):
        """Evaluate mass matrix and force vector numerically (scalar)."""
        U_b, V_b, U_d, V_d = currents
        p = self._params()
        M_elems = compute_M(
            u, v, xd, yd, ud, vd,
            **p, U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
        )
        F_elems = compute_F(
            u, v, xd, yd, ud, vd,
            **p, U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d,
        )
        # Assemble 4x4 symmetric M from upper-triangle elements
        M00, M01, M02, M03, M11, M12, M13, M22, M23, M33 = M_elems
        M = np.array([
            [M00, M01, M02, M03],
            [M01, M11, M12, M13],
            [M02, M12, M22, M23],
            [M03, M13, M23, M33],
        ], dtype=float)
        F = np.array(F_elems, dtype=float)
        return M, F

    def rhs(self, t, y):
        """Right-hand side of the ODE system for ``solve_ivp``.

        Args:
            t: Current time [s].
            y: State vector of length 8:
                ``[x, y, u, v, xd, yd, ud, vd]``.

        Returns:
            Time derivatives of the state vector (length 8).
        """
        x_b, y_b, u, v, xd, yd, ud, vd = y

        theta = _uv_to_theta(u, v)
        z_d = float(max(0.0, -self.l * np.cos(theta)))

        currents = self.get_uv(t=t, z_d=z_d, y_b=y_b, x_b=x_b)

        M, F = self._eval_M_F(
            t, x_b, y_b, u, v, xd, yd, ud, vd, currents
        )

        qdd = np.linalg.solve(M, F)

        return np.array([xd, yd, ud, vd, *qdd])

    _state_vars = ["x", "y", "u", "v", "xd", "yd", "ud", "vd"]

    def _rhs_batch(self, Y, U_b, V_b, U_d, V_d):
        """Vectorized RHS for N particles.

        Uses the generated numpy code from the sympy derivation. All
        arithmetic broadcasts over ``(N,)`` arrays, so no per-particle
        loop is needed.

        Args:
            Y: State array of shape ``(N, 8)``.
            U_b, V_b, U_d, V_d: Current velocities, each of shape ``(N,)``.

        Returns:
            Time derivatives ``dY/dt`` of shape ``(N, 8)``.
        """
        N = Y.shape[0]
        u = Y[:, 2]
        v = Y[:, 3]
        xd = Y[:, 4]
        yd = Y[:, 5]
        ud = Y[:, 6]
        vd = Y[:, 7]

        p = self._params()

        M_elems = compute_M(u, v, xd, yd, ud, vd, **p,
                            U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d)
        F_elems = compute_F(u, v, xd, yd, ud, vd, **p,
                            U_b=U_b, V_b=V_b, U_d=U_d, V_d=V_d)

        M00, M01, M02, M03, M11, M12, M13, M22, M23, M33 = M_elems
        F0, F1, F2, F3 = F_elems

        # Assemble (N, 4, 4) mass matrix and (N, 4) force vector.
        # Some elements (e.g. M01=0) may be scalar; np.broadcast_to handles this.
        M = np.empty((N, 4, 4))
        M[:, 0, 0] = M00
        M[:, 0, 1] = M01
        M[:, 0, 2] = M02
        M[:, 0, 3] = M03
        M[:, 1, 0] = M01
        M[:, 1, 1] = M11
        M[:, 1, 2] = M12
        M[:, 1, 3] = M13
        M[:, 2, 0] = M02
        M[:, 2, 1] = M12
        M[:, 2, 2] = M22
        M[:, 2, 3] = M23
        M[:, 3, 0] = M03
        M[:, 3, 1] = M13
        M[:, 3, 2] = M23
        M[:, 3, 3] = M33

        F = np.stack([np.broadcast_to(F0, N), np.broadcast_to(F1, N),
                      np.broadcast_to(F2, N), np.broadcast_to(F3, N)], axis=-1)

        # Batched solve: numpy >= 2.0 requires b to be (N, 4, 1) for batch mode.
        qdd = np.linalg.solve(M, F[:, :, np.newaxis])[:, :, 0]

        dY = np.empty_like(Y)
        dY[:, 0] = xd
        dY[:, 1] = yd
        dY[:, 2] = ud
        dY[:, 3] = vd
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
                starts from rest with ``theta=theta0`` (converted to (u, v)).
            theta0: Initial pole angle [rad] (used only when ``y0`` is None).
            conv_tol: Stop when ``max(|xdd|, |ydd|) < conv_tol``
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
            y0_arr = np.asarray(y0, dtype=float).reshape(N, 8)
            # y0 is in public (x, y, theta, phi, xd, yd, thetad, phid).
            # Convert to internal (x, y, u, v, xd, yd, ud, vd).
            u0, v0, ud0, vd0 = _spherical_to_uv(
                y0_arr[:, 2], y0_arr[:, 3], y0_arr[:, 6], y0_arr[:, 7],
            )
            y0_internal = np.column_stack([
                y0_arr[:, 0], y0_arr[:, 1],  # x, y
                u0, v0,
                y0_arr[:, 4], y0_arr[:, 5],  # xd, yd
                ud0, vd0,
            ])
            y0_flat = y0_internal.ravel()
        else:
            y0_flat = np.zeros(N * 8)
            # Convert theta0 to stereographic (u, v).
            # For default theta0 ~ pi, delta ~ 0, so u ~ v ~ 0.
            # u = 2*tan((pi-theta)/2)*cos(phi), with phi=0 -> v=0
            delta0 = np.pi - theta0
            u0 = 2 * np.tan(delta0 / 2)  # phi=0, so cos(phi)=1, sin(phi)=0
            y0_flat[2::8] = u0
            # v0 = 0 (already zero)

        def rhs_flat(t, y_flat):
            Y = y_flat.reshape(N, 8)
            dY = self._rhs_batch(Y, U_b, V_b, U_d, V_d)
            return dY.ravel()

        # Global convergence event: max velocity change rate across all particles.
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
        Y_internal = sol.y[:, -1].reshape(N, 8)

        # Convert internal (u, v, ud, vd) state to public (theta, phi, thetad, phid)
        u_final = Y_internal[:, 2]
        v_final = Y_internal[:, 3]
        ud_final = Y_internal[:, 6]
        vd_final = Y_internal[:, 7]
        theta_final, phi_final, thetad_final, phid_final = _uv_to_spherical(
            u_final, v_final, ud_final, vd_final,
        )

        Y_final = np.column_stack([
            Y_internal[:, 0], Y_internal[:, 1],  # x, y
            theta_final, phi_final,
            Y_internal[:, 4], Y_internal[:, 5],  # xd, yd
            thetad_final, phid_final,
        ])

        return Y_final[:, 4], Y_final[:, 5], theta_final, Y_final

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

        Accepts initial conditions in spherical coordinates (theta, phi)
        for backward compatibility, and converts internally to
        stereographic (u, v).

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
            ``x, y, theta, phi, xd, yd, thetad, phid`` as data variables,
            converted from the internal (u, v) representation.
        """
        import xarray as xr

        # Convert (theta, phi, thetad, phid) to (u, v, ud, vd)
        delta = np.pi - theta
        half_delta = delta / 2
        tan_hd = np.tan(half_delta)
        u0 = 2 * tan_hd * np.cos(phi)
        v0 = 2 * tan_hd * np.sin(phi)

        # Jacobian for velocity transform:
        # [thetad, phid] = J22 * [ud, vd]
        # We need the inverse: [ud, vd] = J22^{-1} * [thetad, phid]
        # Rather than computing J22^{-1} symbolically, use the forward
        # expressions:
        #   d(theta)/dt = d(theta)/du * ud + d(theta)/dv * vd
        #   d(phi)/dt   = d(phi)/du * ud + d(phi)/dv * vd
        #
        # For numerical purposes, compute J22 and invert.
        r_st = np.sqrt(u0**2 + v0**2)
        if r_st < 1e-14:
            # At the origin, the Jacobian is -I/2 (from the limit)
            # theta = pi - 2*atan(r/2), so d(theta)/d(r) = -1/(1+(r/2)^2)
            # At r=0: d(theta)/d(r) = -1
            # u = r*cos(phi), v = r*sin(phi)
            # d(theta)/du = -u/r * 1/(1+(r/2)^2), limit -> indeterminate
            # But for thetad=phid=0, ud=vd=0 regardless
            ud0 = 0.0
            vd0 = 0.0
        else:
            # Build J22 numerically and invert
            # d(theta)/du, d(theta)/dv, d(phi)/du, d(phi)/dv
            # theta = pi - 2*atan(r/2), r = sqrt(u^2+v^2)
            # d(theta)/du = -2/(4+r^2) * u/r * r = -u/(1+(r/2)^2) * 1/r ... let me compute directly
            # d(r)/du = u/r, d(r)/dv = v/r
            # d(theta)/dr = -1/(1 + (r/2)^2)
            # d(theta)/du = d(theta)/dr * u/r
            # d(theta)/dv = d(theta)/dr * v/r
            dtheta_dr = -1.0 / (1.0 + (r_st / 2)**2)
            dtheta_du = dtheta_dr * u0 / r_st
            dtheta_dv = dtheta_dr * v0 / r_st
            # phi = atan2(v, u)
            # d(phi)/du = -v/r^2, d(phi)/dv = u/r^2
            dphi_du = -v0 / r_st**2
            dphi_dv = u0 / r_st**2
            J22 = np.array([[dtheta_du, dtheta_dv],
                            [dphi_du, dphi_dv]])
            J22_inv = np.linalg.inv(J22)
            uv_dot = J22_inv @ np.array([thetad, phid])
            ud0, vd0 = uv_dot

        y0_internal = [x, y, u0, v0, xd, yd, ud0, vd0]
        sol = self._get_full_solution(t_span, y0_internal, t_eval=t_eval,
                                       atol=atol, rtol=rtol)

        # Convert internal (u, v, ud, vd) back to (theta, phi, thetad, phid)
        u_arr = sol.y[2]
        v_arr = sol.y[3]
        ud_arr = sol.y[6]
        vd_arr = sol.y[7]

        r_arr = np.sqrt(u_arr**2 + v_arr**2)
        theta_arr = _uv_to_theta(u_arr, v_arr)
        phi_arr = np.arctan2(v_arr, u_arr)

        # Convert velocities back: [thetad, phid] = J22 * [ud, vd]
        # Compute J22 element-wise
        dtheta_dr_arr = -1.0 / (1.0 + (r_arr / 2)**2)
        # Handle r~0 gracefully
        safe_r = np.where(r_arr > 1e-14, r_arr, 1.0)
        dtheta_du_arr = dtheta_dr_arr * np.where(r_arr > 1e-14, u_arr / safe_r, 0.0)
        dtheta_dv_arr = dtheta_dr_arr * np.where(r_arr > 1e-14, v_arr / safe_r, 0.0)
        dphi_du_arr = np.where(r_arr > 1e-14, -v_arr / safe_r**2, 0.0)
        dphi_dv_arr = np.where(r_arr > 1e-14, u_arr / safe_r**2, 0.0)

        thetad_arr = dtheta_du_arr * ud_arr + dtheta_dv_arr * vd_arr
        phid_arr = dphi_du_arr * ud_arr + dphi_dv_arr * vd_arr

        return xr.Dataset(
            {
                "x": ("time", sol.y[0]),
                "y": ("time", sol.y[1]),
                "theta": ("time", theta_arr),
                "phi": ("time", phi_arr),
                "xd": ("time", sol.y[4]),
                "yd": ("time", sol.y[5]),
                "thetad": ("time", thetad_arr),
                "phid": ("time", phid_arr),
            },
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
