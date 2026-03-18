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
    where ``(x, y)`` is the buoy position, ``(theta, phi)`` are the tether angles,
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

        eps = 0.1 / 180 * np.pi
        if abs(theta - np.pi) < eps:
            phid = phid * 0.9
            M, F = self._eval_M_F(
                t, x_b, y_b, theta, phi, xd, yd, thetad, phid, currents
            )
            qdd = np.empty(shape=(4,))
            qdd[:3] = np.linalg.solve(M[:3, :3], F[:3])
            qdd[3] = 0
        else:
            M, F = self._eval_M_F(
                t, x_b, y_b, theta, phi, xd, yd, thetad, phid, currents
            )
            qdd = np.linalg.solve(M, F)

        return np.array([xd, yd, thetad, phid, *qdd])

    _state_vars = ["x", "y", "theta", "phi", "xd", "yd", "thetad", "phid"]

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
            theta: Initial tether polar angle [rad] (default ~pi, hanging down).
            phi: Initial tether azimuthal angle [rad].
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
