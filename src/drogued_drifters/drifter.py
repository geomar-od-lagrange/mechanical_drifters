import numpy as np
from scipy.integrate import solve_ivp

from drogued_drifters.lagrange_model import (
    DrifterPhysics,
    EOMState,
    _qdd_func,
    _spherical_to_uv,
    _uv_to_spherical,
)

# Internal state vector layout: [x, y, u, v, xd, yd, ud, vd].
# These indices are used throughout `rhs`, `_rhs_batch`, and
# `get_final_drift_batch` to unpack and repack state arrays.
IX, IY, IU, IV, IXD, IYD, IUD, IVD = range(8)


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

    Default parameters are for the Callies et al. drifter geometry
    (rho = 1025 kg/m^3)::

        m_tilde_d = drogue_horizontal_added_mass(rho=1025, w_d=0.5, h_d=0.5)  ~ 101.0 kg
        m_tilde_b = buoy_horizontal_added_mass(rho=1025, d_b=0.1, h_b=0.24)   ~   1.9 kg
        k_d       = drogue_horizontal_drag_coeff(rho=1025, w_d=0.5, h_d=0.5)  ~ 154.0 kg/m
        k_b       = buoy_horizontal_drag_coeff(rho=1025, d_b=0.1, h_b=0.24)   ~  12.0 kg/m

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
        get_uv: Callback that returns the ocean current at a given position.
            Must have signature ``get_uv(*, t, x, y, z) -> (U, V)``.
            ``z`` is vertical position [m], positive upward (0 = surface,
            negative = below MSL).
            Called separately at ``z=0`` (buoy) and ``z=z_d`` (drogue).
            If None, uses ``_default_uv``.
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
        self.physics = DrifterPhysics(
            m_b=m_b,
            m_d=m_d,
            m_hat_d=m_hat_d,
            m_tilde_d=m_tilde_d,
            m_tilde_b=m_tilde_b,
            l=l,
            g=g,
            k_b=k_b,
            k_d=k_d,
        )

        if get_uv is not None:
            self.get_uv = get_uv
        else:
            self.get_uv = self._default_uv

    def _default_uv(self, *, t, x, y, z):
        """Default velocity callback for testing. Returns sheared currents.

        Args:
            t: Time [s].
            x: Position x [m].
            y: Position y [m].
            z: Vertical position [m], positive upward (0 = surface, negative = below MSL).

        Returns:
            Tuple ``(U, V)`` current velocity [m/s] at (x, y, z).
        """
        if z == 0.0:
            return 1.0, 1.0
        return -1.0, -1.0

    def _rhs(self, t, y):
        """Right-hand side of the ODE system for ``solve_ivp``.

        Args:
            t: Current time [s].
            y: State vector of length 8:
                ``[x, y, u, v, xd, yd, ud, vd]``.

        Returns:
            Time derivatives of the state vector (length 8).
        """
        x_b = y[IX]
        y_b = y[IY]
        u = y[IU]
        v = y[IV]
        xd = y[IXD]
        yd = y[IYD]
        ud = y[IUD]
        vd = y[IVD]

        z_d = float(self._z_eff(np.array([u]), np.array([v]))[0])

        U_b, V_b = self.get_uv(t=t, x=x_b, y=y_b, z=0.0)
        U_d, V_d = self.get_uv(t=t, x=x_b, y=y_b, z=z_d)

        state = EOMState(u, v, xd, yd, ud, vd, U_b, V_b, U_d, V_d)
        qdd = _qdd_func(self.physics, state)  # returns (4,)

        return np.array([xd, yd, ud, vd, *qdd])

    def _z_eff(self, u, v):
        """Compute effective drogue vertical position from stereographic (u, v).

        Returns:
            z_eff: Drogue vertical position [m], positive upward (non-positive),
                shape ``(N,)``.  At equilibrium (pole vertical) returns ``-l``.
                Clamped to ``<= 0`` (drogue cannot be above the surface).
        """
        s = u**2 + v**2
        cos_theta = (s - 4) / (s + 4)
        # At equilibrium cos_theta = -1, so l * cos_theta = -l (below surface).
        # The clamp to z_eff <= 0 is a safety net: with extreme initial
        # conditions (e.g. very large lateral velocity relative to water) the
        # pole could theoretically swing past horizontal (theta < pi/2),
        # placing the drogue above the surface.  This is outside the physical
        # operating regime of the model, and the uv callback will likely fail
        # on positive z anyway.  We accept this edge case and clamp silently
        # rather than adding per-call warnings on this hot path.
        return np.minimum(0.0, self.physics.l * cos_theta)

    def _rhs_batch(self, Y, sample_uv):
        """Vectorized RHS for N particles.

        Uses _qdd_func for fast evaluation. All arithmetic broadcasts
        over ``(N,)`` arrays, so no per-particle loop is needed.

        Args:
            Y: State array of shape ``(N, 8)``.
            sample_uv: Callable ``sample_uv(z) -> (U, V)`` that returns
                eastward and northward velocity arrays of shape ``(N,)``
                at depth ``z`` (scalar or ``(N,)`` array).

        Returns:
            Time derivatives ``dY/dt`` of shape ``(N, 8)``.
        """
        N = Y.shape[0]
        u = Y[:, IU]
        v = Y[:, IV]
        xd = Y[:, IXD]
        yd = Y[:, IYD]
        ud = Y[:, IUD]
        vd = Y[:, IVD]

        # Sample velocity at buoy (z=0) and drogue (z=z_eff)
        U_b, V_b = sample_uv(np.zeros(N))
        z_eff = self._z_eff(u, v)
        U_d, V_d = sample_uv(z_eff)

        state = EOMState(
            u=u,
            v=v,
            xd=xd,
            yd=yd,
            ud=ud,
            vd=vd,
            U_b=U_b,
            V_b=V_b,
            U_d=U_d,
            V_d=V_d,
        )

        qdd = _qdd_func(self.physics, state)  # returns (N, 4)

        # Guard NaN/inf
        bad = ~np.isfinite(qdd).all(axis=1)
        if np.any(bad):
            qdd[bad] = 0.0

        dY = np.empty_like(Y)
        dY[:, IX] = xd
        dY[:, IY] = yd
        dY[:, IU] = ud
        dY[:, IV] = vd
        dY[:, IXD:] = qdd

        return dY

    def get_final_drift_batch(
        self,
        *,
        sample_uv,
        t_span=(0, 120),
        y0=None,
        atol=1e-3,
        rtol=1e-3,
    ):
        """Compute steady-state drift for N particles in one ``solve_ivp`` call.

        Stacks all N particles into a single ``(8N,)`` ODE system so that
        ``solve_ivp`` overhead is paid once, and the vectorized RHS
        (``_rhs_batch``) evaluates all particles simultaneously.

        ``sample_uv`` must be a callable ``sample_uv(z) -> (U, V)`` that returns
        ``(N,)`` velocity arrays at depth ``z`` (scalar or ``(N,)`` array).
        The ODE solver queries the buoy velocity at z=0 and the drogue velocity
        at the current effective depth ``z_eff(theta)`` on every RHS evaluation,
        so the drogue depth tracks the pole tilt dynamically.

        To use fixed buoy/drogue velocities, pass a step-function sampler::

            def sample_uv(z):
                return (U_b, V_b) if np.all(z == 0) else (U_d, V_d)

        Integration runs for the full ``t_span``. After integration, ``max_accel``
        is computed as the maximum drift acceleration at the final state, which
        the caller can use to assess convergence.

        Args:
            sample_uv: Velocity profile sampler (see above).
            t_span: Integration window ``(t_start, t_end)`` in seconds.
            y0: Initial state array of shape ``(N, 8)`` in public format
                ``(x, y, theta, phi, xd, yd, thetad, phid)``.  If ``None``,
                starts from rest at equilibrium ``(u_stereo, v_stereo) = (0, 0)``
                i.e. ``theta=pi, phi=0``.
            atol: Absolute tolerance for the ODE solver.
            rtol: Relative tolerance for the ODE solver.

        Returns:
            Tuple ``(xd_final, yd_final, Y_final, max_accel)`` where
            ``xd_final`` and ``yd_final`` are ``(N,)`` arrays, ``Y_final`` is the
            full ``(N, 8)`` state in public format (pass back as ``y0`` for
            warm-starting), and ``max_accel`` is the maximum drift acceleration
            ``max(|xdd|, |ydd|)`` across all particles at the final state.
            Column layout of ``Y_final``:
            ``[x, y, theta, phi, xd, yd, thetad, phid]``.
        """
        # Determine N from y0 or by probing the sampler
        if y0 is not None:
            N = np.asarray(y0).reshape(-1, 8).shape[0]
        else:
            probe = sample_uv(np.array([0.0]))
            N = len(probe[0])

        # _flat: the (N, 8) state array raveled to a (8*N,) vector for
        # solve_ivp's 1-D interface.
        if y0 is not None:
            y0_arr = np.asarray(y0, dtype=float).reshape(N, 8)
            # y0 is in public (x, y, theta, phi, xd, yd, thetad, phid).
            # Convert to internal (x, y, u, v, xd, yd, ud, vd).
            u0, v0, ud0, vd0 = _spherical_to_uv(
                y0_arr[:, 2],
                y0_arr[:, 3],
                y0_arr[:, 6],
                y0_arr[:, 7],
            )
            y0_internal = np.column_stack(
                [
                    y0_arr[:, 0],
                    y0_arr[:, 1],  # x, y
                    u0,
                    v0,
                    y0_arr[:, 4],
                    y0_arr[:, 5],  # xd, yd
                    ud0,
                    vd0,
                ]
            )
            y0_flat = y0_internal.ravel()
        else:
            # Default: start at rest with drogue hanging straight down,
            # i.e. (u_stereo, v_stereo) = (0, 0) which is theta=pi, phi=0.
            y0_flat = np.zeros(N * 8)

        def rhs_flat(t, y_flat):
            Y = y_flat.reshape(N, 8)
            dY = self._rhs_batch(Y, sample_uv)
            return dY.ravel()

        sol = solve_ivp(
            rhs_flat,
            t_span,
            y0_flat,
            atol=atol,
            rtol=rtol,
        )
        Y_internal = sol.y[:, -1].reshape(N, 8)

        # Evaluate convergence diagnostic: max drift acceleration at final state
        dY_final = self._rhs_batch(Y_internal, sample_uv)
        max_accel = float(np.max(np.abs(dY_final[:, IXD : IYD + 1])))

        # Convert internal (u, v, ud, vd) state to public (theta, phi, thetad, phid)
        u_final = Y_internal[:, IU]
        v_final = Y_internal[:, IV]
        ud_final = Y_internal[:, IUD]
        vd_final = Y_internal[:, IVD]
        theta_final, phi_final, thetad_final, phid_final = _uv_to_spherical(
            u_final,
            v_final,
            ud_final,
            vd_final,
        )

        Y_final = np.column_stack(
            [
                Y_internal[:, IX],
                Y_internal[:, IY],  # x, y
                theta_final,
                phi_final,
                Y_internal[:, IXD],
                Y_internal[:, IYD],  # xd, yd
                thetad_final,
                phid_final,
            ]
        )

        return Y_final[:, IXD], Y_final[:, IYD], Y_final, max_accel

    def _solve(self, t_span, y0, t_eval=None, atol=1e-3, rtol=1e-3):
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
        return solve_ivp(self._rhs, t_span, y0, atol=atol, rtol=rtol, t_eval=t_eval)

    def get_full_solution(
        self,
        *,
        t_span,
        x=0.0,
        y=0.0,
        theta=np.pi,
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

        u0, v0, ud0, vd0 = _spherical_to_uv(theta, phi, thetad, phid)
        y0_internal = [x, y, u0, v0, xd, yd, ud0, vd0]
        sol = self._solve(t_span, y0_internal, t_eval=t_eval, atol=atol, rtol=rtol)

        theta_arr, phi_arr, thetad_arr, phid_arr = _uv_to_spherical(
            sol.y[IU],
            sol.y[IV],
            sol.y[IUD],
            sol.y[IVD],
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
        *,
        t_span,
        x=0.0,
        y=0.0,
        theta=np.pi,
        phi=0.0,
        xd=0.0,
        yd=0.0,
        thetad=0.0,
        phid=0.0,
    ):
        """Integrate and return the steady-state buoy drift velocity.

        Args:
            t_span: ``(t_start, t_end)`` in seconds. Must be long enough for
                the system to approach steady state.
            x, y, theta, phi, xd, yd, thetad, phid: Initial conditions
                (same as ``get_full_solution``).

        Returns:
            Tuple ``(xd_final, yd_final, max_accel)`` — the buoy drift velocity
            [m/s] at the end of the integration, and the maximum drift
            acceleration ``max(|xdd|, |ydd|)`` at the final state (a convergence
            diagnostic; smaller is better).
        """
        u0, v0, ud0, vd0 = _spherical_to_uv(theta, phi, thetad, phid)
        y0 = [x, y, u0, v0, xd, yd, ud0, vd0]
        sol = self._solve(t_span, y0)
        y_final = sol.y[:, -1]

        # Convergence diagnostic
        dy_final = self._rhs(0.0, y_final)
        max_accel = float(max(abs(dy_final[IXD]), abs(dy_final[IYD])))

        return float(y_final[IXD]), float(y_final[IYD]), max_accel
