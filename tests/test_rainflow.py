import typhoon
import numpy as np
import numpy.testing as npt

from collections import Counter


def test_rainflow():
    waveform = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0])
    last_peaks = np.array([])
    cycles, remaining_peaks = typhoon.rainflow(
        waveform=waveform,
        last_peaks=last_peaks,
        bin_size=1.0,
    )

    print("Cycles:", cycles)
    print("Remaining Peaks:", remaining_peaks)

    npt.assert_array_equal(
        remaining_peaks,
        np.array([0.0, 4.0]),
    )

    assert cycles == {(2.0, 1.0): 2}


def test_stitching():
    waveform1 = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0])
    waveform2 = np.array([3.0, 5.0])

    cycles1, remaining_peaks = typhoon.rainflow(
        waveform=waveform1,
        last_peaks=None,
        bin_size=1.0,
    )

    print("Waveform 1:", waveform1)
    print("Cycles 1:", cycles1)
    print("Remaining Peaks 1:", remaining_peaks)

    cycles2, remaining_peaks = typhoon.rainflow(
        waveform=waveform2,
        last_peaks=remaining_peaks,
        bin_size=1.0,
    )

    print("Waveform 2:", waveform2)
    print("Cycles 2:", cycles2)
    print("Remaining Peaks 2:", remaining_peaks)

    cycles_merged = Counter(cycles1) + Counter(cycles2)

    print("Merged Cycles:", cycles_merged)

    npt.assert_array_equal(
        remaining_peaks,
        np.array([0., 5.]),
    )

    assert cycles_merged == {(2.0, 1.0): 2, (4.0, 3.0): 1}
