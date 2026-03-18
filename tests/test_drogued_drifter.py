from drogued_drifters.drifter import DroguedDrifter
from drogued_drifters.lagrange_model import M_func, F_func

import numpy as np


def test_drogued_drifter_instantiation():
    dd = DroguedDrifter()


def test_MF_callable():
    assert callable(M_func)
    assert callable(F_func)


def test_MF_evaluates():
    dd = DroguedDrifter()

    t = 0.0
    currents = dd.get_uv(t=t, z_d=0.0, y_b=0.0, x_b=0.0)

    M, F = dd._eval_M_F(
        t,
        x=0.0,
        y=0.0,
        theta=0.3,
        phi=0.1,
        xd=0.0,
        yd=0.0,
        thetad=0.0,
        phid=0.0,
        currents=currents,
    )

    assert len(M.squeeze().shape) == 2, "M not 2dim"
    assert len(F.squeeze().shape) == 1, "F not 1dim"


def test_no_drift_for_zero_currents():
    t_span = (0.0, 30.0)
    y_0 = (0, 0, 0.999 * np.pi, 0, 0, 0, 0, 0)

    def _getuv_zero(*, t, z_d, y_b, x_b):
        return 0.0, 0.0, 0.0, 0.0

    dd = DroguedDrifter(get_uv=_getuv_zero)

    U_drift, V_drift, y_final, sol = dd.get_final_drift(
        t_span=t_span, y0=y_0, t_eval=(0, 30.0)
    )

    np.testing.assert_almost_equal(U_drift, 0.0, decimal=1)
    np.testing.assert_almost_equal(V_drift, 0.0, decimal=1)


def test_no_drift_for_theta_pi_zero_currents():
    """Drogue hangs straight down (theta=pi), no currents: should stay at rest."""
    t_span = (0.0, 30.0)
    y_0 = (0, 0, np.pi, 0, 0, 0, 0, 0)

    def _getuv_zero(*, t, z_d, y_b, x_b):
        return 0.0, 0.0, 0.0, 0.0

    dd = DroguedDrifter(get_uv=_getuv_zero)

    U_drift, V_drift, y_final, sol = dd.get_final_drift(
        t_span=t_span, y0=y_0, t_eval=(0, 30.0)
    )

    np.testing.assert_almost_equal(U_drift, 0.0, decimal=1)
    np.testing.assert_almost_equal(V_drift, 0.0, decimal=1)
