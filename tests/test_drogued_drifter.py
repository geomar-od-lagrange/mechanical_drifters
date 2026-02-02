from drogued_drifters.drifter import DroguedDrifter


def test_drogued_drifter_instantiation():
    dd = DroguedDrifter()

def test_MF_callable():
    dd = DroguedDrifter()
    assert callable(dd.M_lbd)
    assert callable(dd.F_lbd)
