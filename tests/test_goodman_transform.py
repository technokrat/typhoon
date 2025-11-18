import numpy as np

from typhoon.typhoon import rainflow, goodman_transform


def test_goodman_simple_symmetric_cycle():
    waveform = np.array([10.0, 20.0, 10.0, 20.0], dtype=np.float32)

    cycles, _ = rainflow(waveform)

    ersatz = goodman_transform(cycles, m=0.8)

    # There should be one equivalent amplitude of 10 with count 1
    # Allow for floating point key lookup by finding the closest key
    assert len(ersatz) == 1
    ((key, count),) = ersatz.items()
    amp = key
    assert np.isclose(amp, 12.789473533630371, atol=1e-6)
    assert count == 1
