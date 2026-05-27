"""SparBuoyDrifter model: spar buoy at the surface with two quadratic drag forces (surface/water)."""

from typing import NamedTuple

import numpy as np
import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

from ..base import LagrangianMechanicsModel


class SparBuoyPhysics(NamedTuple):
    """Physical constants for a spar buoy.

    Default values: ``SparBuoyPhysics()`` gives m=1, m_tilde=1, k_air=10, k_water=10, draft=7, n_z=7.
    """

    m: float = 1.0  # mass [kg]
    m_tilde: float = 1.0  # added mass [kg]
    k_air: float = 10.0  # drag coefficient (air) [kg/m]
    k_water: float = 10.0 # drag coefficient (water) [kg/m]
    draft: float = 10.0 # draft of the buoy
    height_air: float = 9.0 # height above water
    n_air: int = 3 # sampling rate in air
    n_water: int = 4 # sampling rate in water


class SparBuoyState(NamedTuple):
    """Per-timestep state variables."""

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


class SparBuoyDrifter(LagrangianMechanicsModel):
    """A spar buoy with 7m draft.

    Two generalized coordinates (x, y). The Lagrangian is
    L = 1/2 (m + m_tilde)(xd^2 + yd^2), drag force F = -k|v-u|(v-u).

    State vector: [x, y, xd, yd]
    """

    Physics = SparBuoyPhysics
    State = SparBuoyState
    n_q = 2
    state_names = ("x", "y", "xd", "yd")

    def __init__(self, physics=SparBuoyPhysics(), *, backend="numpy"):
        super().__init__(physics, backend=backend)

    def _derive_symbolic(self):
        """Derive M, F for a spar buoy with two quadratic drag forces.

        Lagrangian: L = 1/2 (m + m_tilde)(xd^2 + yd^2)
        No potential energy (surface-constrained particle).
        Drag: F_drag = -k |v - u| (v - u) where v = (xd, yd), u = (U, V).
        """
        t = dynamicsymbols._t
        x, y = dynamicsymbols("x y")

        m = sp.Symbol("m", positive=True)
        m_tilde = sp.Symbol("m_tilde", positive=True)
        
        k_air = sp.Symbol("k_air", real=True)
        k_water = sp.Symbol("k_water", real=True)
        
        draft = sp.Symbol("draft", real = True)
        height_air = sp.Symbol("height_air", real = True)

        n_air = sp.Symbol("n_air", real = True)
        n_water = sp.Symbol("n_water", real = True)

        U_air_0 = sp.Symbol("U_air_0", real = True)
        U_air_1 = sp.Symbol("U_air_1", real = True)
        U_air_2 = sp.Symbol("U_air_2", real = True)

        V_air_0 = sp.Symbol("V_air_0", real = True)
        V_air_1 = sp.Symbol("V_air_1", real = True)
        V_air_2 = sp.Symbol("V_air_2", real = True)

        U_water_0 = sp.Symbol("U_water_0", real = True)
        U_water_1 = sp.Symbol("U_water_1", real = True)
        U_water_2 = sp.Symbol("U_water_2", real = True)
        U_water_3 = sp.Symbol("U_water_3", real = True)
        
        V_water_0 = sp.Symbol("V_water_0", real = True)
        V_water_1 = sp.Symbol("V_water_1", real = True)
        V_water_2 = sp.Symbol("V_water_2", real = True)
        V_water_3 = sp.Symbol("V_water_3", real = True)

        q = sp.Matrix([x, y])
        qd = q.diff(t)

        # Lagrangian: pure kinetic energy
        L = sp.Rational(1, 2) * (m + m_tilde) * qd.dot(qd)

        # Derive Q
        r = sp.Matrix([x, y]) # position
        q = sp.Matrix([x, y]) # generalized coordinates (might later change to polar coordinates)

        # Substitute dynamic symbols with static ones for lambdify
        xd_dyn, yd_dyn = x.diff(t), y.diff(t)
        xd_static = sp.Symbol("xd", real=True)
        yd_static = sp.Symbol("yd", real=True)
        
        # symbolic drag air
        
        Fx_air_terms = []
        Fy_air_terms = []

        U_air = [U_air_0, U_air_1, U_air_2]
        V_air = [V_air_0, V_air_1, V_air_2]

        for Ui, Vi in zip(U_air, V_air):
            rel_vel_x = xd_static - Ui
            rel_vel_y = yd_static - Vi

            rel_speed = sp.sqrt(rel_vel_x**2 + rel_vel_y**2)

            Fx_air_terms.append(-k_air * rel_speed * rel_vel_x)
            Fy_air_terms.append(-k_air * rel_speed * rel_vel_y)

        Fx_air_sym = sum(Fx_air_terms) / 3
        Fy_air_sym = sum(Fy_air_terms) / 3

        # symbolic drag water
        
        Fx_water_terms = []
        Fy_water_terms = []

        U_water = [U_water_0, U_water_1, U_water_2, U_water_3]
        V_water = [V_water_0, V_water_1, V_water_2, V_water_3]

        for Ui, Vi in zip(U_water, V_water):
            rel_vel_x = xd_static - Ui
            rel_vel_y = yd_static - Vi

            rel_speed = sp.sqrt(rel_vel_x**2 + rel_vel_y**2)

            Fx_water_terms.append(-k_water * rel_speed * rel_vel_x)
            Fy_water_terms.append(-k_water * rel_speed * rel_vel_y)

        Fx_water_sym = sum(Fx_water_terms) / 4
        Fy_water_sym = sum(Fy_water_terms) / 4

        Fx_drag_total = Fx_water_sym + Fx_air_sym
        Fy_drag_total = Fy_water_sym + Fy_air_sym
        
        F = sp.Matrix([Fx_drag_total, Fy_drag_total])

        Q = sp.Matrix([F.dot(r.diff(qi)) for qi in q])

        # Euler-Lagrange: d/dt(dL/dqd) - dL/dq = Q
        # => (m + m_tilde) * qdd = Q
        qdd = qd.diff(t)
        eoms = sp.Matrix([
            L.diff(qdj).diff(t) - L.diff(qj) - Qj
            for qj, qdj, Qj in zip(q, qd, Q)
        ])

        M_sym, F_sym = sp.linear_eq_to_matrix(eoms, list(qdd))
        

        subs = {xd_dyn: xd_static, yd_dyn: yd_static}
        M_static = M_sym.subs(subs)
        F_static = F_sym.subs(subs)

        symbol_map = {"m": m, "m_tilde": m_tilde, "k_air": k_air, "k_water": k_water, "draft": draft, "height_air": height_air, "n_air": n_air, "n_water": n_water, "xd": xd_static, "yd": yd_static, "U_air_0": U_air_0, "V_air_0": V_air_0, "U_air_1": U_air_1, "V_air_1": V_air_1, "U_air_2": U_air_2, "V_air_2": V_air_2, "U_water_0": U_water_0, "V_water_0": V_water_0, "U_water_1": U_water_1, "V_water_1": V_water_1, "U_water_2": U_water_2, "V_water_2": V_water_2, "U_water_3": U_water_3, "V_water_3": V_water_3}
        all_fields = (list(SparBuoyPhysics._fields) +
                      list(SparBuoyState._fields))
        
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

        # air drag
        z_air_levels = np.linspace(self.physics.height_air / self.physics.n_air, self.physics.height_air, int(self.physics.n_air))

        U_air_values = []
        V_air_values = []

        for z in z_air_levels:
            U_air, V_air = sample_uv(np.full(N, z))

            U_air_values.append(U_air)
            V_air_values.append(V_air)

        # water drag

        z_water_levels = np.linspace(0.0, self.physics.draft, int(self.physics.n_water))

        U_water_values = []
        V_water_values = []

        for z in z_water_levels:
            U_water, V_water = sample_uv(np.full(N, z))

            U_water_values.append(U_water)
            V_water_values.append(V_water)

        state = SparBuoyState(xd = xd, yd = yd, U_air_0 = U_air_values[0], V_air_0 = V_air_values[0], U_air_1 = U_air_values[1], V_air_1 = V_air_values[1], U_air_2 = U_air_values[2], V_air_2 = V_air_values[2], U_water_0 = U_water_values[0], V_water_0 = V_water_values[0], U_water_1 = U_water_values[1], V_water_1 = V_water_values[1], U_water_2 = U_water_values[2], V_water_2 = V_water_values[2], U_water_3 = U_water_values[3], V_water_3 = V_water_values[3])

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