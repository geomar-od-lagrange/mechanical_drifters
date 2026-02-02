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
