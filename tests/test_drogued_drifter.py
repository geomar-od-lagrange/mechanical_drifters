from drogued_drifters.drifter import DroguedDrifter

import numpy as np


def test_drogued_drifter_instantiation():
    dd = DroguedDrifter()


def test_MF_callable():
    dd = DroguedDrifter()
    assert callable(dd.M_lbd)
    assert callable(dd.F_lbd)


def test_MF_evaluates():
    dd = DroguedDrifter()

    t = 0.0
    q = np.array([0.0, 0.0, 0.3, 0.1])
    qd = np.array([0.0, 0.0, 0.0, 0.0])

    x_b, y_b, th, ph = q
    z_d = max(0.0, dd.l * np.cos(th))

    U_b, V_b, U_d, V_d = dd.get_uv(t, z_d, y_b, x_b, None)

    dyn_params = (U_b, V_b, U_d, V_d)

    M = dd.M_num(t, *q, *qd, *dyn_params)
    F = dd.F_num(t, *q, *qd, *dyn_params)

    assert len(np.array(M).squeeze().shape) == 2, "M not 2dim"
    assert len(np.array(F).squeeze().shape) == 1, "F not 1dim"


def test_no_netto_drift_for_no_curents():
    t_span = (0.0, 30.0)
    y_0 = (0, 0, 0.999 * np.pi, 0, 0, 0, 0, 0)

    def _getuv_zero(t, z_d, y_b, x_b, ds_subset):
        return 0.0, 0.0, 0.0, 0.0

    dd = DroguedDrifter(get_uv=_getuv_zero)

    U_netto, V_netto, y_next, sol = dd.get_netto_uv(
        t_span=t_span, y0=y_0, t_eval=(0, 30.0)
    )

    np.testing.assert_almost_equal(U_netto, 0.0, decimal=1)
    np.testing.assert_almost_equal(V_netto, 0.0, decimal=1)
