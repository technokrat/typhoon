import numpy as np

from typhoon.typhoon import rainflow, goodman_transform
import typhoon


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


def test_goodman_bin_size_compensation_widens_cycle_free_function():
    cycles = {(0.0, 1.0): 1}

    no_comp = goodman_transform(cycles, m=0.0, bin_size_compensation=0.0)
    with_comp = goodman_transform(cycles, m=0.0, bin_size_compensation=1.0)

    assert len(no_comp) == 1
    assert len(with_comp) == 1

    (amp0, count0), = no_comp.items()
    (amp1, count1), = with_comp.items()

    assert np.isclose(amp0, 0.5, atol=1e-7)
    assert np.isclose(amp1, 1.0, atol=1e-7)
    assert count0 == 1
    assert count1 == 1


def test_goodman_bin_size_compensation_supported_in_context():
    waveform = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    ctx = typhoon.RainflowContext(bin_size=0.0, threshold=0.0)
    ctx.process(waveform)

    no_comp = ctx.goodman_transform(m=0.0, bin_size_compensation=0.0)
    with_comp = ctx.goodman_transform(m=0.0, bin_size_compensation=1.0)

    assert len(no_comp) == 1
    assert len(with_comp) == 1

    (amp0, count0), = no_comp.items()
    (amp1, count1), = with_comp.items()

    assert np.isclose(amp0, 0.5, atol=1e-7)
    assert np.isclose(amp1, 1.0, atol=1e-7)
    assert count0 == 1
    assert count1 == 1
