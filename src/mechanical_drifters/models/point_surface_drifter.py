"""PointSurfaceDrifter model: point particle at the surface with quadratic drag."""

from typing import NamedTuple

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from ..base import LagrangianMechanicsModel


class PointSurfacePhysics(NamedTuple):
    """Physical constants for a point surface drifter.

    Default values: ``PointSurfacePhysics()`` gives m=1, m_tilde=1, k=10.
    """

    m: float = 1.0  # mass [kg]
    m_tilde: float = 1.0  # added mass [kg]
    k: float = 10.0  # drag coefficient [kg/m]


class PointSurfaceState(NamedTuple):
    """Per-timestep state variables and forcing."""

    xd: float | np.ndarray  # eastward drift velocity [m/s]
    yd: float | np.ndarray  # northward drift velocity [m/s]
    U: float | np.ndarray  # current at surface, east [m/s]
    V: float | np.ndarray  # current at surface, north [m/s]


# State vector layout: [x, y, xd, yd]
IX, IY, IXD, IYD = range(4)


class PointSurfaceDrifter(LagrangianMechanicsModel):
    """A point particle at the ocean surface with quadratic drag.

    Two generalized coordinates (x, y). The Lagrangian is
    L = 1/2 (m + m_tilde)(xd^2 + yd^2), drag force F = -k|v-u|(v-u).
    At steady state: drift equals the surface current exactly.

    State vector: [x, y, xd, yd]
    """

    Physics = PointSurfacePhysics
    State = PointSurfaceState
    n_q = 2
    state_names = ("x", "y", "xd", "yd")

    def __init__(self, physics=PointSurfacePhysics(), *, backend="numpy"):
        super().__init__(physics, backend=backend)

    def _derive_symbolic(self):
        """Derive M, F for a point particle with quadratic drag.

        Lagrangian: L = 1/2 (m + m_tilde)(xd^2 + yd^2)
        No potential energy (surface-constrained particle).
        Drag: F_drag = -k |v - u| (v - u) where v = (xd, yd), u = (U, V).
        """
        t = dynamicsymbols._t
        x, y = dynamicsymbols("x y")

        m = sp.Symbol("m", positive=True)
        m_tilde = sp.Symbol("m_tilde", positive=True)
        k = sp.Symbol("k", positive=True)
        U, V = sp.symbols("U V", real=True)

        q = sp.Matrix([x, y])
        qd = q.diff(t)

        # Lagrangian: pure kinetic energy
        L = sp.Rational(1, 2) * (m + m_tilde) * qd.dot(qd)

        # Velocity and current vectors
        v = sp.Matrix([qd[0], qd[1]])
        u = sp.Matrix([U, V])
        rel = v - u
        rel_speed = sp.sqrt(rel.dot(rel))

        # Quadratic drag as generalized force
        F_drag = -k * rel_speed * rel
        Q = sp.Matrix([F_drag[i] for i in range(2)])

        # Euler-Lagrange: d/dt(dL/dqd) - dL/dq = Q
        # => (m + m_tilde) * qdd = Q
        qdd = qd.diff(t)
        eoms = sp.Matrix([
            L.diff(qdj).diff(t) - L.diff(qj) - Qj
            for qj, qdj, Qj in zip(q, qd, Q)
        ])

        M_sym, F_sym = sp.linear_eq_to_matrix(eoms, list(qdd))

        # Substitute dynamic symbols with static ones for lambdify
        xd_dyn, yd_dyn = x.diff(t), y.diff(t)
        xd_static = sp.Symbol("xd", real=True)
        yd_static = sp.Symbol("yd", real=True)

        subs = {xd_dyn: xd_static, yd_dyn: yd_static}
        M_static = M_sym.subs(subs)
        F_static = F_sym.subs(subs)

        symbol_map = {
            "m": m, "m_tilde": m_tilde, "k": k,
            "xd": xd_static, "yd": yd_static, "U": U, "V": V,
        }
        all_fields = list(PointSurfacePhysics._fields) + list(PointSurfaceState._fields)
        args = tuple(symbol_map[field] for field in all_fields)

        return M_static, F_static, args

    def _rhs_batch(self, Y, sample_uv):
        """Compute dY/dt for N particles.

        Args:
            Y: ``(N, state_size)`` state array [x, y, xd, yd].
            sample_uv: callable(z) -> (U, V) for (N,) arrays.

        Returns:
            dY: ``(N, state_size)`` derivatives.
        """
        N = Y.shape[0]
        xd = Y[:, IXD]
        yd = Y[:, IYD]

        U, V = sample_uv(np.zeros(N))

        state = PointSurfaceState(xd=xd, yd=yd, U=U, V=V)
        qdd = self._qdd_func(self.physics, state, batch=True)

        bad = ~np.isfinite(qdd).all(axis=1)
        if np.any(bad):
            qdd[bad] = 0.0

        n_q = self.n_q
        dY = np.empty_like(Y)
        dY[:, :n_q] = Y[:, n_q:]   # d/dt(q) = qd  (kinematic identity)
        dY[:, n_q:] = qdd          # d/dt(qd) = qdd
        return dY

    def drift_velocity(self, Y):
        """Extract drift velocity from state array.

        Args:
            Y: State array, shape ``(N, state_size)``.

        Returns:
            Drift velocity array, shape ``(N, 2)``.
        """
        return Y[:, [IXD, IYD]]

