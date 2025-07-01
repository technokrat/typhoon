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
        min_chunk_size=64 * 1024,
    )

    print("Cycles:", cycles)
    print("Remaining Peaks:", remaining_peaks)

    npt.assert_array_equal(
        remaining_peaks,
        np.array([0.0, 4.0]),
    )

    assert cycles == {(2.0, 1.0): 2}


def test_waveform_stitching():
    waveform1 = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0])
    waveform2 = np.array([3.0, 5.0])

    cycles1, remaining_peaks = typhoon.rainflow(
        waveform=waveform1,
        last_peaks=None,
        bin_size=1.0,
        min_chunk_size=64 * 1024,
    )

    print("Waveform 1:", waveform1)
    print("Cycles 1:", cycles1)
    print("Remaining Peaks 1:", remaining_peaks)

    cycles2, remaining_peaks = typhoon.rainflow(
        waveform=waveform2,
        last_peaks=remaining_peaks,
        bin_size=1.0,
        min_chunk_size=64 * 1024,
    )

    print("Waveform 2:", waveform2)
    print("Cycles 2:", cycles2)
    print("Remaining Peaks 2:", remaining_peaks)

    cycles_merged = Counter(cycles1) + Counter(cycles2)

    print("Merged Cycles:", cycles_merged)

    npt.assert_array_equal(
        remaining_peaks,
        np.array([0.0, 5.0]),
    )

    assert cycles_merged == {(2.0, 1.0): 2, (4.0, 3.0): 1}


def test_benchmark_128m_samples(benchmark):
    """Will benchmark the rainflow counting algorithm on a random waveform with 512 * 4096 samples"""

    def large_waveform_rainflow_counting(waveform: np.typing.NDArray[np.float64]):
        cycles, remaining_peaks = typhoon.rainflow(
            waveform=waveform,
            last_peaks=None,
            bin_size=0.1,
            min_chunk_size=64 * 1024,
        )

        print("Remaining Peaks:", remaining_peaks.shape)

    np.random.seed(42)
    waveform = np.random.random_sample(128 * 1024 * 1024)

    @benchmark
    def run():
        large_waveform_rainflow_counting(waveform)
