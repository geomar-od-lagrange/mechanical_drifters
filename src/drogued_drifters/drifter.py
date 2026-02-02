import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
from drogued_drifters.lagrange_model import M, F, args


class DroguedDrifter:
    MF_CACHE = None  # use functools

    def __init__(
        self,
        m_b: float = 0.5,
        m_d: float = 0.5,
        l: float = 3.0,
        k_b: float = 0.5,
        k_d: float = 2.0,
        g: float = 9.81,
        get_uv=None,
    ):
        # DroguedDrifter attributes hold physics parameters (masses, drag coeffs, ...)
        self.k_b = float(k_b)
        self.k_d = float(k_d)
        self.m_b = float(m_b)
        self.m_d = float(m_d)
        self.l = float(l)
        self.g = float(g)

        if get_uv is not None:
            self.get_uv = get_uv
        else:
            self.get_uv = self.default_uv

        self.M_lbd, self.F_lbd = self.solve_sp_MF()

    def solve_sp_MF(self):

        if DroguedDrifter.MF_CACHE is not None:
            return DroguedDrifter.MF_CACHE

        M_lbd = sp.lambdify(args, M, modules="numpy")
        F_lbd = sp.lambdify(args, F, modules="numpy")

        DroguedDrifter.MF_CACHE = (M_lbd, F_lbd)

        return M_lbd, F_lbd

    par_syms = args[9:15]
    
    def par_dict(self):
        return {
        "m_b": self.m_b,
        "m_d": self.m_d,
        "l":   self.l,
        "g":   self.g,
        "k_b": self.k_b,
        "k_d": self.k_d,
        }

    def par_vals(self):
        par_dict = self.par_dict()
        return tuple(par_dict[s] for s in par_syms)
    
    def default_uv(self, t, z_d, y_b, x_b, ds_subset):
        U_b, V_b = 1.0, 1.0
        # factor = np.exp(-abs(z_d) / 6.0)
        # U_d, V_d = U_b * factor, V_b * factor
        U_d, V_d = -1.0, -1.0
        return U_b, V_b, U_d, V_d

    def rhs(self, t, y, ds_subset=None):
        q = y[:4]
        qd = y[4:]

        x_b, y_b, th, ph = q

        z_d = float(max(0.0, self.l * np.cos(th)))

        U_b, V_b, U_d, V_d = self.get_uv(t, z_d, y_b, x_b, ds_subset)

        dyn_params = (*par_vals, U_b, V_b, U_d, V_d)

        eps = 0.1 / 180 * np.pi
        theta = q[2]
        if abs(theta - np.pi) < eps:
            qd[3] *= 0.9
            M_num = np.array(
                self.M_lbd(t, *q, *qd, *dyn_params), dtype=float
            )  # als dict
            F_num = np.array(self.F_lbd(t, *q, *qd, *dyn_params), dtype=float).reshape(
                -1
            )
            qdd = np.empty(shape=(4,))
            qdd[:3] = np.linalg.solve(M_num[:3, :3], F_num[:3])
            qdd[3] = 0
        else:
            M_num = np.array(
                self.M_lbd(t, *q, *qd, *dyn_params), dtype=float
            )  # als dict
            F_num = np.array(self.F_lbd(t, *q, *qd, *dyn_params), dtype=float).reshape(
                -1
            )
            qdd = np.linalg.solve(M_num, F_num)

        return np.concatenate([qd, qdd])

    def get_full_solution(
        self, t_span, y0, ds_subset=None, t_eval=None, atol=1e-3, rtol=1e-3
    ):
        # TODO: this method gets initial conditions and runtime etc.
        sol = solve_ivp(
            self.rhs, t_span, y0, args=(ds_subset,), atol=atol, rtol=rtol, t_eval=t_eval
        )
        return sol

    def get_netto_uv(self, t_span, y0, ds_subset=None, t_eval=None):
        # TODO: this method gets initial conditions and runtime etc.
        # TODO: this method gets U,V profile
        # TODO: Calls .get_full_solution() and extracts netto equilibrium drift

        sol = self.get_full_solution(t_span, y0, ds_subset=ds_subset, t_eval=t_eval)

        Eq_U_Drift = sol.y[4, -1]
        Eq_V_Drift = sol.y[5, -1]
        y_next = sol.y[:, -1]

        return Eq_U_Drift, Eq_V_Drift, y_next, sol
