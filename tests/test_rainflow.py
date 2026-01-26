import typhoon
import numpy as np
import numpy.testing as npt

from collections import Counter

import logging

from typhoon import helper

logger = logging.getLogger(__name__)


def test_rainflow():
    waveform = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0], dtype=np.float32)
    last_peaks = np.array([], dtype=np.float32)
    cycles, remaining_peaks = typhoon.rainflow(
        waveform=waveform,
        last_peaks=last_peaks,
        bin_size=1.0,
        min_chunk_size=64 * 1024,
    )

    logger.info("Cycles:", cycles)
    logger.info("Remaining Peaks:", remaining_peaks)

    npt.assert_array_equal(
        remaining_peaks,
        np.array([0.0, 4.0]),
    )

    assert cycles == {(2.0, 1.0): 2}


def test_waveform_stitching():
    waveform1 = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0], dtype=np.float32)
    waveform2 = np.array([3.0, 5.0], dtype=np.float32)

    cycles1, remaining_peaks = typhoon.rainflow(
        waveform=waveform1,
        last_peaks=None,
        bin_size=1.0,
        min_chunk_size=64 * 1024,
    )

    logger.info("Waveform 1:", waveform1)
    logger.info("Cycles 1:", cycles1)
    logger.info("Remaining Peaks 1:", remaining_peaks)

    cycles2, remaining_peaks = typhoon.rainflow(
        waveform=waveform2,
        last_peaks=remaining_peaks,
        bin_size=1.0,
        min_chunk_size=64 * 1024,
    )

    logger.info("Waveform 2:", waveform2)
    logger.info("Cycles 2:", cycles2)
    logger.info("Remaining Peaks 2:", remaining_peaks)

    cycles_merged = Counter(cycles1) + Counter(cycles2)

    logger.info("Merged Cycles:", cycles_merged)

    npt.assert_array_equal(
        remaining_peaks,
        np.array([0.0, 5.0]),
    )

    assert cycles_merged == {(2.0, 1.0): 2, (4.0, 3.0): 1}


def test_reference_load_waveform():
    waveform = np.array(
        [
            0.0,
            90.0,
            30.0,
            120.0,
            -150.0,
            -30.0,
            -60.0,
            60.0,
            0.0,
            120.0,
            60.0,
            180.0,
            -30.0,
            0.0,
            -120.0,
            30.0,
        ],
        dtype=np.float32,
    )
    cycles, remaining_peaks = typhoon.rainflow(
        waveform=waveform,
        last_peaks=None,
        bin_size=0,
        min_chunk_size=64 * 1024,
    )
    logger.info("Reference Cycles:", cycles)
    logger.info("Reference Half-Cycles:", max(0, len(remaining_peaks) - 1))

    assert cycles == {
        (-30.0, -60.0): 1,
        (90.0, 30.0): 1,
        (120.0, 60.0): 1,
        (-30.0, 0.0): 1,
        (60.0, 0.0): 1,
    }
    assert len(remaining_peaks) - 1 == 5


def test_benchmark_20m_samples(benchmark):
    """Will benchmark the rainflow counting algorithm on a random waveform with 512 * 4096 samples"""

    def large_waveform_rainflow_counting(waveform: np.typing.NDArray[np.float32]):
        cycles, remaining_peaks = typhoon.rainflow(
            waveform=waveform,
            last_peaks=None,
            bin_size=0.001,
            min_chunk_size=64 * 1024,
        )

    np.random.seed(42)
    waveform = np.random.random_sample(20 * 1024 * 1024).astype(dtype=np.float32)

    @benchmark
    def run():
        large_waveform_rainflow_counting(waveform)


def test_benchmark_accumulation(benchmark):
    """Will benchmark the rainflow counting accumulation on a random waveform"""

    def large_waveform_rainflow_counting(
        waveform: np.typing.NDArray[np.float32],
        residuals: np.typing.NDArray[np.float32],
    ):
        cycles, remaining_peaks = typhoon.rainflow(
            waveform=waveform,
            last_peaks=residuals,
            bin_size=0.001,
            min_chunk_size=64 * 1024,
        )

        return cycles, remaining_peaks

    np.random.seed(42)

    @benchmark
    def run():
        cycles: helper.CycleCounter = Counter()
        remaining_peaks = np.array([])

        for i in range(10):
            waveform = np.random.random_sample(10000).astype(dtype=np.float32)
            new_cycles, remaining_peaks = large_waveform_rainflow_counting(
                waveform, remaining_peaks
            )
            cycles = helper.merge_cycle_counters([cycles, new_cycles])

            hist = typhoon.goodman_transform(cycles, m=0.3)
            # summed = typhoon.summed_histogram(hist)

            #helper.counter_to_full_interval_df(cycles)
            #print(len(cycles))



