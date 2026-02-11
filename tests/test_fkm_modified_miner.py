import numpy as np

from typhoon.typhoon import MinerDamageMode, fkm_miner_damage


def test_fkm_miner_damage_collective():
    goodman = {
        200.0: 10,
        160.0: 90,
        120.0: 1000,
        80.0: 9000,
        40.0: 10000,
    }

    d_mm = fkm_miner_damage(goodman, n_d=1e6, sigma_d=100.0, k=5.0)
    assert np.isclose(d_mm, 4.963e-3, rtol=2e-4)

    d_om = fkm_miner_damage(
        goodman,
        n_d=1e6,
        sigma_d=100.0,
        k=5.0,
        mode=MinerDamageMode.Original,
    )
    assert np.isclose(d_om, 0.0037520384, rtol=1e-12)
