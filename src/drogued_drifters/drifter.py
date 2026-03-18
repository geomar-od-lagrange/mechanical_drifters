import numpy as np
from scipy.integrate import solve_ivp

from drogued_drifters.lagrange_model import M_func, F_func


class DroguedDrifter:
    """Simulator for a drogued drifter in ocean currents.

    A drogued drifter consists of a surface buoy connected by a wire of length
    ``l`` to a subsurface drogue. Both experience quadratic drag from the
    surrounding water. The equations of motion are derived from a Lagrangian
    formulation (see ``lagrange_model``).

    The state vector has 8 components: ``[x, y, theta, phi, xd, yd, thetad, phid]``
    where ``(x, y)`` is the buoy position, ``(theta, phi)`` are the tether angles,
    and ``d`` denotes time derivatives.

    Args:
        m_b: Buoy mass [kg].
        m_d: Drogue mass [kg].
        l: Wire length [m].
        k_b: Buoy drag coefficient.
        k_d: Drogue drag coefficient.
        g: Gravitational acceleration [m/s^2].
        get_uv: Callback that returns ocean currents at a given position.
            Must have signature ``get_uv(*, t, z_d, y_b, x_b)`` and return
            ``(U_b, V_b, U_d, V_d)``. If None, uses ``default_uv``.
            Use ``functools.partial`` to bind external data (e.g. an xarray
            dataset) before passing it here.
    """

    def __init__(
        self, *, m_b=0.5, m_d=0.5, l=3.0, k_b=1.5, k_d=2.0, g=9.81, get_uv=None
    ):
        self.m_b = m_b
        self.m_d = m_d
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

    def get_full_solution(self, t_span, y0, t_eval=None, atol=1e-3, rtol=1e-3):
        """Integrate the equations of motion over a time span.

        Args:
            t_span: ``(t_start, t_end)`` in seconds.
            y0: Initial state vector of length 8.
            t_eval: Times at which to store the solution. If None, the solver
                chooses its own time steps.
            atol: Absolute tolerance for the ODE solver.
            rtol: Relative tolerance for the ODE solver.

        Returns:
            ``scipy.integrate.OdeResult`` with fields ``.t`` and ``.y``.
        """
        sol = solve_ivp(self.rhs, t_span, y0, atol=atol, rtol=rtol, t_eval=t_eval)
        return sol

    def get_final_drift(self, t_span, y0, t_eval=None):
        """Integrate and return the buoy drift velocity at the end.

        Args:
            t_span: ``(t_start, t_end)`` in seconds.
            y0: Initial state vector of length 8.
            t_eval: Times at which to store the solution.

        Returns:
            Tuple of ``(U_drift, V_drift, y_final, sol)`` where
            ``U_drift`` and ``V_drift`` are the buoy velocities at ``t_end``
            [m/s], ``y_final`` is the final state vector, and ``sol`` is the
            full ``OdeResult``.
        """
        sol = self.get_full_solution(t_span, y0, t_eval=t_eval)

        U_drift = sol.y[4, -1]
        V_drift = sol.y[5, -1]
        y_final = sol.y[:, -1]

        return U_drift, V_drift, y_final, sol
