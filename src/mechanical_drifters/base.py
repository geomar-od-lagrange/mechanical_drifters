"""Base class for drifting-object models derived via Lagrangian mechanics."""

import re
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


class LagrangianMechanicsModel:
    """Base class for drifting-object models derived via Lagrangian mechanics.

    Subclass this to define a new drifting object. You must provide:

    1. Class attributes: ``Physics``, ``State``, ``n_q``, ``state_names``.
    2. Methods: ``_derive_symbolic``, ``_rhs_batch``, ``drift_velocity``.

    The base class provides ODE integration (``integrate``).

    See ``DroguedDrifter`` for a complete example and
    ``PointSurfaceDrifter`` for a minimal one.
    """

    # --- Class attributes (override in subclass) ---

    Physics = None   # NamedTuple class for physical constants
    State = None     # NamedTuple class for per-timestep state + forcing
    n_q = None       # number of generalized coordinates
    state_names = None  # tuple of str, e.g. ("x", "y", "xd", "yd")

    # --- Constructor ---

    def __init__(self, physics, *, backend="numpy"):
        # Validate that the subclass set the required class attributes.
        for attr in ("Physics", "State", "n_q", "state_names"):
            if getattr(type(self), attr, None) is None:
                raise TypeError(
                    f"{type(self).__name__} must set class attribute {attr!r}"
                )

        self.physics = physics
        self.backend = backend

        from .eom import _get_eom_callables

        self._qdd_func = _get_eom_callables(self, backend)[0]

    # --- Properties ---

    @property
    def state_size(self):
        """Total state vector length (2 * n_q for standard layouts)."""
        return 2 * self.n_q

    @property
    def _cache_path(self):
        """Path to the pickled symbolic derivation cache.

        Auto-derived from the class name:
        DroguedDrifter -> data/eom_cache_drogued_drifter.pkl

        Override on the subclass if a different path is needed.
        """
        name = type(self).__name__
        snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()
        return Path(__file__).resolve().parent / "data" / f"eom_cache_{snake}.pkl"

    # --- Methods to override ---

    def _derive_symbolic(self):
        """Derive symbolic M and F from the Lagrangian.

        Use sympy to:
        1. Define generalized coordinates and physical parameters.
        2. Write down T, V, and the Lagrangian L = T - V.
        3. Compute generalized forces Q from non-conservative forces.
        4. Apply the Euler-Lagrange equations.
        5. Extract M and F such that M * qdd = F.

        Symbol names in the returned ``args`` tuple must exactly match
        field names in ``self.Physics`` and ``self.State``.

        Returns:
            Tuple ``(M_static, F_static, args)``.
        """
        raise NotImplementedError

    def drift_velocity(self, Y):
        """Extract drift velocity from state array.

        Args:
            Y: State array, shape ``(N, state_size)``.

        Returns:
            Drift velocity array, shape ``(N, 2)``.
        """
        raise NotImplementedError

    def _rhs_batch(self, Y, sample_uv):
        """Compute dY/dt for N particles.

        This is the complete right-hand side of the ODE. It queries
        velocities via ``sample_uv``, builds the State NamedTuple,
        evaluates accelerations via ``self._qdd_func``, guards NaN, and
        packs derivatives. Everything in one method, no sub-dispatch.

        Args:
            Y: State array, shape ``(N, state_size)``.
            sample_uv: Callable ``sample_uv(z) -> (U, V)`` where z is
                ``(N,)`` and U, V are ``(N,)`` arrays in m/s.

        Returns:
            dY: Derivative array, shape ``(N, state_size)``.
        """
        raise NotImplementedError

    # --- Provided by the base class ---

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
        """Integrate the ODE for the given time span.

        Args:
            sample_uv: Velocity sampler ``sample_uv(z) -> (U, V)``.
                Returns ``(N,)`` arrays for ``(N,)`` input.
            t_span: Integration window ``(t_start, t_end)`` in seconds.
            y0: Initial state, shape ``(N, state_size)``.
                If None, cold-start from zeros (equilibrium at rest).
            t_eval: Times at which to store the solution. If None,
                only the final state is returned (T=1).
            atol, rtol: ODE solver tolerances.

        Returns:
            Tuple ``(t, Y, max_acceleration)`` where:
            - t: ``(T,)`` time array
            - Y: ``(T, N, state_size)`` state array
            - max_acceleration: scalar (max |d(drift_vel)/dt| at final time)
        """
        ss = self.state_size

        if y0 is not None:
            y0_arr = np.asarray(y0, dtype=float).reshape(-1, ss)
            N = y0_arr.shape[0]
            y0_flat = y0_arr.ravel()
        else:
            # Determine N by probing the sampler
            probe = sample_uv(np.array([0.0]))
            N = len(probe[0])
            y0_flat = np.zeros(N * ss)

        def rhs_flat(t, y_flat):
            Y = y_flat.reshape(N, ss)
            dY = self._rhs_batch(Y, sample_uv)
            return dY.ravel()

        sol = solve_ivp(
            rhs_flat, t_span, y0_flat,
            t_eval=t_eval, atol=atol, rtol=rtol,
        )

        if t_eval is None:
            # Return only the final state: T=1
            Y_final = sol.y[:, -1].reshape(1, N, ss)
            t = np.array([sol.t[-1]])
        else:
            T = len(sol.t)
            Y_full = np.empty((T, N, ss))
            for i in range(T):
                Y_full[i] = sol.y[:, i].reshape(N, ss)
            Y_final = Y_full
            t = sol.t

        # Convergence diagnostic: max drift acceleration at final state
        Y_last = Y_final[-1]  # (N, ss)
        dY_last = self._rhs_batch(Y_last, sample_uv)
        drift_accel = self.drift_velocity(dY_last)
        max_accel = float(np.max(np.abs(drift_accel)))

        return t, Y_final, max_accel

    def to_xarray(self, t, Y):
        """Wrap integrate() output into an xr.Dataset using state_names.

        Args:
            t: ``(T,)`` time array from integrate().
            Y: ``(T, N, state_size)`` state array from integrate().

        Returns:
            xr.Dataset with dims ``(time, traj)`` and one DataArray per
            state_names entry.
        """
        import xarray as xr

        T, N, ss = Y.shape
        names = self.state_names
        data_vars = {}
        for k, name in enumerate(names):
            data_vars[name] = (("time", "traj"), Y[:, :, k])
        return xr.Dataset(
            data_vars,
            coords={"time": t, "traj": np.arange(N)},
        )

