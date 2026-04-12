"""SparBuoy model: always-vertical buoy drifting with depth-averaged velocity.

A spar buoy is modeled as a rigid vertical cylinder of length ``l``
extending from the surface to depth ``-l``.  It drifts with the
depth-averaged horizontal velocity over its length.  There is no tilt
dynamics — the buoy is always vertical.

This is not a Lagrangian mechanics model (no generalized coordinates for
orientation, no sympy derivation).  It overrides ``__init__`` and
``steady_state_batch`` to bypass the EOM machinery entirely.
"""

from typing import NamedTuple

import numpy as np

from ..base import LagrangianMechanicsModel


class SparBuoyPhysics(NamedTuple):
    """Physical parameters for a spar buoy."""

    l: float  # buoy length [m] — extends from surface to depth -l


class SparBuoyState(NamedTuple):
    """Placeholder state (unused — spar buoy has no internal dynamics)."""

    xd: float | np.ndarray  # eastward drift velocity [m/s]
    yd: float | np.ndarray  # northward drift velocity [m/s]


class SparBuoy(LagrangianMechanicsModel):
    """An always-vertical spar buoy drifting with depth-averaged velocity.

    The buoy extends from the surface (z=0) to depth z=-l. Its drift
    velocity is the depth-averaged horizontal velocity over that range,
    computed by querying the velocity sampler at the surface and at -l
    and taking the mean.

    This model has no internal dynamics — ``steady_state_batch`` returns
    the depth-averaged velocity directly without ODE integration.

    State vector: ``[x, y, xd, yd]`` (position + velocity).
    """

    Physics = SparBuoyPhysics
    State = SparBuoyState
    n_q = 2
    _drift_velocity_indices = (2, 3)  # xd, yd

    def __init__(self, physics=None, *, l=None, backend="numpy"):
        """Create a SparBuoy.

        Args:
            physics: SparBuoyPhysics instance, or None for defaults.
            l: Buoy length [m]. Shorthand for ``SparBuoyPhysics(l=l)``.
            backend: Ignored (no EOM compilation needed).
        """
        # Validate class attributes without calling _make_qdd_func
        for attr in ("Physics", "State", "n_q", "_drift_velocity_indices"):
            if getattr(type(self), attr, None) is None:
                raise TypeError(
                    f"{type(self).__name__} must set class attribute {attr!r}"
                )

        if physics is None:
            if l is not None:
                physics = SparBuoyPhysics(l=l)
            else:
                physics = self.default_physics()
        self.physics = physics
        self.backend = backend
        self._qdd_func = None  # not used

    def default_physics(self):
        """Default 15 m spar buoy."""
        return SparBuoyPhysics(l=15.0)

    def _derive_symbolic(self):
        """Not applicable — spar buoy has no Lagrangian dynamics."""
        raise NotImplementedError(
            "SparBuoy has no Lagrangian dynamics. "
            "It drifts with depth-averaged velocity."
        )

    def _rhs_batch(self, Y, sample_uv):
        """Compute dY/dt: velocity is the depth-averaged current.

        Args:
            Y: (N, 4) state array [x, y, xd, yd].
            sample_uv: callable(z) -> (U, V) for (N,) arrays.

        Returns:
            dY: (N, 4) derivatives.
        """
        N = Y.shape[0]

        # Sample at surface and at depth -l, take mean
        U_surface, V_surface = sample_uv(np.zeros(N))
        U_deep, V_deep = sample_uv(np.full(N, -self.physics.l))

        U_avg = 0.5 * (U_surface + U_deep)
        V_avg = 0.5 * (V_surface + V_deep)

        dY = np.empty_like(Y)
        dY[:, 0] = U_avg  # dx/dt = U_avg
        dY[:, 1] = V_avg  # dy/dt = V_avg
        dY[:, 2] = 0.0    # d(xd)/dt = 0 (no acceleration)
        dY[:, 3] = 0.0    # d(yd)/dt = 0
        return dY

    def _max_depth(self, physics):
        """Buoy extends to depth l."""
        return physics.l

    def steady_state_batch(
        self,
        sample_uv,
        *,
        t_span=(0, 120),
        y0=None,
        atol=1e-3,
        rtol=1e-3,
    ):
        """Return depth-averaged velocity directly (no ODE integration needed).

        Args:
            sample_uv: Velocity sampler ``sample_uv(z) -> (U, V)``.
            t_span: Ignored (no integration needed).
            y0: Ignored.
            atol, rtol: Ignored.

        Returns:
            Tuple ``(drift_vel, Y_final, max_accel)`` matching the base
            class contract.
        """
        # Determine N by probing the sampler
        probe = sample_uv(np.array([0.0]))
        N = len(probe[0])

        # Sample at surface and at depth -l
        U_surface, V_surface = sample_uv(np.zeros(N))
        U_deep, V_deep = sample_uv(np.full(N, -self.physics.l))

        U_avg = 0.5 * (U_surface + U_deep)
        V_avg = 0.5 * (V_surface + V_deep)

        drift_vel = np.column_stack([U_avg, V_avg])

        # Y_final: [x, y, xd, yd] — positions zero, velocities = drift
        Y_final = np.column_stack([
            np.zeros(N), np.zeros(N),
            U_avg, V_avg,
        ])

        # No dynamics, so max_accel is always 0
        max_accel = 0.0

        return drift_vel, Y_final, max_accel
