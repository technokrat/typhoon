from __future__ import annotations

import numpy as np

from typhoon.woehler import (
    MinerType,
    WoehlerCurveParams,
    woehler_loads,
    woehler_loads_basic,
    woehler_log_space,
)


def test_woehler_log_space_bounds_and_length() -> None:
    xs = woehler_log_space(1.0, 1.0e8, 11)
    assert isinstance(xs, np.ndarray)
    assert xs.shape == (11,)
    assert np.isclose(xs[0], 1.0)
    assert np.isclose(xs[-1], 1.0e8)


def test_woehler_basic_monotone_decreasing() -> None:
    params = WoehlerCurveParams(sd=100.0, nd=1.0e6, k1=5.0)
    cycles = woehler_log_space(1.0e3, 1.0e7, 50)

    loads = woehler_loads_basic(cycles, params, miner=MinerType.NONE)

    # For positive k1 the S-N curve should decrease with increasing cycles.
    assert np.all(np.diff(loads) <= 1e-9)
    assert loads[0] > loads[-1]


def test_woehler_probability_influence() -> None:
    params = WoehlerCurveParams(sd=300, nd=1.5e6, k1=6.2)
    cycles = np.array([1.0e5, 1.0e6, 1.0e7])

    loads_med = woehler_loads(cycles, params, failure_probability=0.5)
    loads_low = woehler_loads(cycles, params, failure_probability=0.1)
    loads_high = woehler_loads(cycles, params, failure_probability=0.9)

    # Lower failure probability should correspond to higher allowable loads,
    # higher failure probability to lower allowable loads.
    assert np.all(loads_low <= loads_med)
    assert np.all(loads_med <= loads_high)


def test_woehler_miner_type_effect() -> None:
    params = WoehlerCurveParams(sd=100.0, nd=1.0e6, k1=5.0)
    cycles = np.array([1.0e5, 1.0e6, 1.0e7])

    loads_none = woehler_loads_basic(cycles, params, miner=MinerType.NONE)
    loads_haibach = woehler_loads_basic(cycles, params, miner=MinerType.HAIBACH)

    # Different Miner type should generally lead to different loads once the
    # second slope becomes relevant (here for the higher cycle counts).
    assert not np.allclose(loads_none, loads_haibach)


def test_woehler_predamage_params() -> None:
    params = WoehlerCurveParams(sd=100.0, nd=1.0e6, k1=5.0)

    damaged = params.with_predamage(d_predamage=0.2)

    # Default q is (k1 - 1) = 4
    expected_sd = 100.0 * (1.0 - 0.2) ** (1.0 / 4.0)
    expected_nd = 1.0e6 * (expected_sd / 100.0) ** (-(5.0 - 4.0))

    assert np.isclose(damaged.sd, expected_sd)
    assert np.isclose(damaged.nd, expected_nd)
    assert damaged.k1 == params.k1
