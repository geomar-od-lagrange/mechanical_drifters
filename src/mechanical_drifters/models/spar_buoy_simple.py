"""SparBuoySimple model: a vertical spar buoy with depth-averaged quadratic drag."""

from typing import NamedTuple

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from ..base import LagrangianMechanicsModel


class SparBuoyPhysics(NamedTuple):
    """Physical constants for a spar buoy.

    ``SparBuoyPhysics()`` gives m=1, m_tilde=1, k_air=10, k_water=10, draft=10,
    height_air=9, n_air=3, n_water=4.  ``n_air``/``n_water`` are the number of
    drag-sample levels in the air column and the submerged hull; they are fixed
    by the ``SparBuoyState`` layout (see ``SparBuoySimple``).
    """

    m: float = 1.0  # mass [kg]
    m_tilde: float = 1.0  # added mass [kg]
    k_air: float = 10.0  # lumped quadratic drag factor in air [kg/m]
    k_water: float = 10.0  # lumped quadratic drag factor in water [kg/m]
    draft: float = 10.0  # submerged hull length below the surface [m]
    height_air: float = 9.0  # column height above the surface [m]
    n_air: int = 3  # number of air sample levels
    n_water: int = 4  # number of water sample levels


class SparBuoyState(NamedTuple):
    """Per-timestep state: drift velocity and the current at each sample level."""

    xd: float | np.ndarray  # eastward drift velocity [m/s]
    yd: float | np.ndarray  # northward drift velocity [m/s]

    U_air_0: float | np.ndarray
    V_air_0: float | np.ndarray
    U_air_1: float | np.ndarray
    V_air_1: float | np.ndarray
    U_air_2: float | np.ndarray
    V_air_2: float | np.ndarray

    U_water_0: float | np.ndarray
    V_water_0: float | np.ndarray
    U_water_1: float | np.ndarray
    V_water_1: float | np.ndarray
    U_water_2: float | np.ndarray
    V_water_2: float | np.ndarray
    U_water_3: float | np.ndarray
    V_water_3: float | np.ndarray


# State vector layout: [x, y, xd, yd]
IX, IY, IXD, IYD = range(4)


class SparBuoySimple(LagrangianMechanicsModel):
    """A vertical spar buoy with depth-averaged quadratic drag in air and water.

    Two generalized coordinates (x, y); pure-kinetic Lagrangian
    ``L = 1/2 (m + m_tilde)(xd^2 + yd^2)``.  Drag is the mean over ``n_air`` air
    levels and ``n_water`` water levels of ``F = -k |v - u| (v - u)``, assembled
    symbolically as a generalized force ``Q = sum_i (d r_i / d q) . F_i`` over
    drag points ``r_i = r_b + z_i * pole_hat`` along the (currently vertical)
    pole.  Because each level carries its own current in the State, ``n_air=3``
    and ``n_water=4`` are fixed.

    State vector: [x, y, xd, yd]
    """

    Physics = SparBuoyPhysics
    State = SparBuoyState
    n_q = 2
    state_names = ("x", "y", "xd", "yd")

    def __init__(self, physics=SparBuoyPhysics(), *, backend="numpy"):
        if (physics.n_air, physics.n_water) != (3, 4):
            raise ValueError(
                "SparBuoySimple is wired for n_air=3, n_water=4 — the "
                "SparBuoyState carries one current per level."
            )
        super().__init__(physics, backend=backend)

    def _derive_symbolic(self):
        """Derive M, F for the spar buoy with depth-averaged quadratic drag.

        Each sample level is a drag point at ``r_i = r_b + z_i * pole_hat`` with
        relative velocity ``v_i - u_i`` and quadratic drag
        ``F_i = -(k/n) |v_i - u_i| (v_i - u_i)``; the generalized force is
        ``Q = sum_i (d r_i / d q) . F_i``.  ``pole_hat`` is vertical, so the
        level height ``z_i`` cancels from ``Q`` (leaving the mean horizontal
        drag), but it is carried through so that with tilt coordinates in ``q``
        and ``pole_hat = pole_hat(tilt)`` the same assembly yields the torques.
        """
        t = dynamicsymbols._t
        x, y = dynamicsymbols("x y")

        m = sp.Symbol("m", positive=True)
        m_tilde = sp.Symbol("m_tilde", positive=True)
        k_air = sp.Symbol("k_air", positive=True)
        k_water = sp.Symbol("k_water", positive=True)
        draft = sp.Symbol("draft", positive=True)
        height_air = sp.Symbol("height_air", positive=True)
        n_air = sp.Symbol("n_air", positive=True)
        n_water = sp.Symbol("n_water", positive=True)

        U_air = sp.symbols("U_air_0 U_air_1 U_air_2", real=True)
        V_air = sp.symbols("V_air_0 V_air_1 V_air_2", real=True)
        U_water = sp.symbols("U_water_0 U_water_1 U_water_2 U_water_3", real=True)
        V_water = sp.symbols("V_water_0 V_water_1 V_water_2 V_water_3", real=True)

        q = sp.Matrix([x, y])
        qd = q.diff(t)

        # Lagrangian: pure kinetic energy (no potential; surface-constrained).
        L = sp.Rational(1, 2) * (m + m_tilde) * qd.dot(qd)

        # Buoy reference position and (vertical) pole direction.
        r_b = sp.Matrix([x, y, 0])
        pole_hat = sp.Matrix([0, 0, 1])

        # Symbolic sample heights [positive up]; air above, water below surface.
        z_air = [height_air * sp.Integer(i + 1) / n_air for i in range(len(U_air))]
        z_water = [-draft * sp.Integer(i) / (n_water - 1) for i in range(len(U_water))]

        # Generalized drag: sum of per-level quadratic drag, averaged per medium.
        Q = sp.zeros(len(q), 1)
        for k, n, Us, Vs, Zs in (
            (k_air, n_air, U_air, V_air, z_air),
            (k_water, n_water, U_water, V_water, z_water),
        ):
            for Ui, Vi, zi in zip(Us, Vs, Zs):
                r_i = r_b + zi * pole_hat
                v_i = sp.Matrix([r_i[0].diff(t), r_i[1].diff(t), 0])
                rel = v_i - sp.Matrix([Ui, Vi, 0])
                F_i = -(k / n) * sp.sqrt(rel.dot(rel)) * rel
                for j, qj in enumerate(q):
                    Q[j] += r_i.diff(qj).dot(F_i)

        # Euler-Lagrange: d/dt(dL/dqd) - dL/dq = Q  =>  (m + m_tilde) qdd = Q
        qdd = qd.diff(t)
        eoms = sp.Matrix([
            L.diff(qdj).diff(t) - L.diff(qj) - Qj
            for qj, qdj, Qj in zip(q, qd, Q)
        ])

        M_sym, F_sym = sp.linear_eq_to_matrix(eoms, list(qdd))

        # Substitute dynamic velocity symbols with static ones for lambdify.
        xd_static = sp.Symbol("xd", real=True)
        yd_static = sp.Symbol("yd", real=True)
        subs = {x.diff(t): xd_static, y.diff(t): yd_static}
        M_static = M_sym.subs(subs)
        F_static = F_sym.subs(subs)

        symbol_map = {
            "m": m, "m_tilde": m_tilde, "k_air": k_air, "k_water": k_water,
            "draft": draft, "height_air": height_air,
            "n_air": n_air, "n_water": n_water,
            "xd": xd_static, "yd": yd_static,
            "U_air_0": U_air[0], "V_air_0": V_air[0],
            "U_air_1": U_air[1], "V_air_1": V_air[1],
            "U_air_2": U_air[2], "V_air_2": V_air[2],
            "U_water_0": U_water[0], "V_water_0": V_water[0],
            "U_water_1": U_water[1], "V_water_1": V_water[1],
            "U_water_2": U_water[2], "V_water_2": V_water[2],
            "U_water_3": U_water[3], "V_water_3": V_water[3],
        }
        all_fields = list(SparBuoyPhysics._fields) + list(SparBuoyState._fields)
        args = tuple(symbol_map[field] for field in all_fields)

        return M_static, F_static, args

    def _rhs_batch(self, Y, sample_uv):
        """Compute dY/dt for N particles.

        Args:
            Y: ``(N, state_size)`` state array [x, y, xd, yd].
            sample_uv: callable(z) -> (U, V) for (N,) arrays, z positive up.

        Returns:
            dY: ``(N, state_size)`` derivatives.
        """
        N = Y.shape[0]
        xd = Y[:, IXD]
        yd = Y[:, IYD]
        p = self.physics

        # Air levels above the surface (z > 0), water levels below (z < 0):
        # sample_uv(z) takes z positive upward.
        z_air = np.linspace(p.height_air / p.n_air, p.height_air, p.n_air)
        z_water = -np.linspace(0.0, p.draft, p.n_water)

        U_air, V_air = [], []
        for z in z_air:
            U, V = sample_uv(np.full(N, z))
            U_air.append(U)
            V_air.append(V)

        U_water, V_water = [], []
        for z in z_water:
            U, V = sample_uv(np.full(N, z))
            U_water.append(U)
            V_water.append(V)

        state = SparBuoyState(
            xd=xd, yd=yd,
            U_air_0=U_air[0], V_air_0=V_air[0],
            U_air_1=U_air[1], V_air_1=V_air[1],
            U_air_2=U_air[2], V_air_2=V_air[2],
            U_water_0=U_water[0], V_water_0=V_water[0],
            U_water_1=U_water[1], V_water_1=V_water[1],
            U_water_2=U_water[2], V_water_2=V_water[2],
            U_water_3=U_water[3], V_water_3=V_water[3],
        )

        qdd = self._qdd_func(self.physics, state, batch=True)

        bad = ~np.isfinite(qdd).all(axis=1)
        if np.any(bad):
            qdd[bad] = 0.0

        n_q = self.n_q
        dY = np.empty_like(Y)
        dY[:, :n_q] = Y[:, n_q:]   # d/dt(q) = qd  (kinematic identity)
        dY[:, n_q:] = qdd          # d/dt(qd) = qdd
        return dY

    @property
    def _max_depth(self):
        """Submerged hull extends to the draft below the surface."""
        return self.physics.draft

    def drift_velocity(self, Y):
        """Extract drift velocity from state array.

        Args:
            Y: State array, shape ``(N, state_size)``.

        Returns:
            Drift velocity array, shape ``(N, 2)``.
        """
        return Y[:, [IXD, IYD]]
