"""Base class for drifting-object models derived via Lagrangian mechanics."""

import re
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


class LagrangianMechanicsModel:
    """Base class for drifting-object models derived via Lagrangian mechanics.

    Subclass this to define a new drifting object. You must provide:

    1. Class attributes: ``Physics``, ``State``, ``n_q``,
       ``_drift_velocity_indices``.
    2. Methods: ``default_physics``, ``_derive_symbolic``, ``_rhs_batch``,
       ``_max_depth``.

    The base class provides ODE integration (``steady_state_batch``)
    and a Parcels kernel factory (``make_kernel``).

    See ``DroguedDrifter`` for a complete example and
    ``PointSurfaceDrifter`` for a minimal one.
    """

    # --- Class attributes (override in subclass) ---

    Physics = None   # NamedTuple class for physical constants
    State = None     # NamedTuple class for per-timestep state + forcing
    n_q = None       # number of generalized coordinates

    # Indices into the state vector [q0, ..., q_{n-1}, qdot0, ..., qdot_{n-1}]
    # that are the "drift velocity" output.
    # DroguedDrifter: (4, 5) = (xd, yd).
    _drift_velocity_indices = None

    # --- Constructor ---

    def __init__(self, physics=None, *, backend="numpy"):
        # Validate that the subclass set the required class attributes.
        for attr in ("Physics", "State", "n_q", "_drift_velocity_indices"):
            if getattr(type(self), attr, None) is None:
                raise TypeError(
                    f"{type(self).__name__} must set class attribute {attr!r}"
                )

        if physics is None:
            physics = self.default_physics()
        self.physics = physics
        self.backend = backend

        from .eom import _make_qdd_func

        self._qdd_func = _make_qdd_func(self, backend)

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

    def default_physics(self):
        """Return the default Physics instance for this model."""
        raise NotImplementedError

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

    def _max_depth(self, physics):
        """Maximum depth [m, positive] to sample from the fieldset.

        The Parcels coupling uses this to decide how many depth levels to
        extract from the ocean model data.

        Args:
            physics: Physics NamedTuple.

        Returns:
            float, positive depth in meters.
        """
        raise NotImplementedError

    # --- Provided by the base class ---

    def steady_state_batch(
        self,
        sample_uv,
        *,
        t_span=(0, 120),
        y0=None,
        atol=1e-3,
        rtol=1e-3,
    ):
        """Compute steady-state drift velocities for N particles.

        Stacks N particles into a single ``(state_size*N,)`` ODE system
        and integrates to steady state.

        Args:
            sample_uv: Velocity sampler ``sample_uv(z) -> (U, V)``.
                Returns ``(N,)`` arrays for ``(N,)`` input.
            t_span: Integration window ``(t_start, t_end)`` in seconds.
            y0: Initial internal state, shape ``(N, state_size)``.
                If None, cold-start from zeros (equilibrium at rest).
            atol, rtol: ODE solver tolerances.

        Returns:
            Tuple ``(drift_vel, Y_final, max_accel)`` where:
            - drift_vel: ``(N, len(_drift_velocity_indices))``
            - Y_final: ``(N, state_size)`` internal state (for warm-start)
            - max_accel: scalar convergence diagnostic
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

        sol = solve_ivp(rhs_flat, t_span, y0_flat, atol=atol, rtol=rtol)
        Y_final = sol.y[:, -1].reshape(N, ss)

        # Convergence diagnostic: max drift acceleration at final state
        dY_final = self._rhs_batch(Y_final, sample_uv)
        idx = list(self._drift_velocity_indices)
        max_accel = float(np.max(np.abs(dY_final[:, idx])))

        drift_vel = Y_final[:, idx]
        return drift_vel, Y_final, max_accel

    def make_kernel(self):
        """Create a Parcels-compatible kernel for this model.

        Returns a ``(particles, fieldset)`` function suitable for
        ``pset.execute(kernels=[...])``.  Uses ``self.physics`` and
        ``self.backend``.
        """
        from .parcels import make_kernel

        return make_kernel(self)
