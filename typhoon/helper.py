from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal, Mapping, Tuple, cast

import numpy as np
import pandas as pd


CycleKey = Tuple[float, float]
CycleCounter = Counter


def merge_cycle_counters(counters: Iterable[Mapping[CycleKey, float]]) -> CycleCounter:
    """Merge multiple rainflow() cycle dicts/Counter objects using Counter.

    Each input mapping should be like the first return value from ``rainflow``:
    ``{(s_lower, s_upper): count}``.
    """

    total: CycleCounter = Counter()
    for c in counters:
        total.update(c)
    return total


def add_residual_half_cycles(
    counter: Mapping[CycleKey, float],
    residual_peaks: np.ndarray,
) -> CycleCounter:
    """Add half-cycles from residual waveform peaks to an existing Counter.

    The residual peaks array is expected to be the second return value from
    ``rainflow``. It represents half-cycles between adjacent peaks.

    Each half-cycle contributes 0.5 to the count for its (from, to) key.
    """

    result: CycleCounter = Counter(counter)

    if residual_peaks.size < 2:
        return result

    for i in range(len(residual_peaks) - 1):
        f = float(residual_peaks[i])
        t = float(residual_peaks[i + 1])
        key: CycleKey = (f, t)
        result[key] += 0.5  # type: ignore

    return result


def counter_to_full_interval_df(
    counter: Mapping[CycleKey, float],
    bin_size: float = 0.1,
    closed: Literal["left", "right"] = "right",
    round_decimals: int = 12,
) -> pd.DataFrame:
    """Convert a (from, to): count mapping to a full 2D interval DataFrame.

    The returned DataFrame has a MultiIndex of (from_interval, to_interval)
    covering the entire range, with zero counts where no cycles exist.
    """

    if not counter:
        # Return empty but well-formed DataFrame
        return pd.DataFrame(
            [],
            index=pd.MultiIndex.from_arrays(
                [pd.IntervalIndex([], name="from"), pd.IntervalIndex([], name="to")]
            ),
            columns=["value"],
        )

    half = bin_size / 2.0

    from_vals = sorted({f for (f, _) in counter.keys()})
    to_vals = sorted({t for (_, t) in counter.keys()})

    min_val = min(min(from_vals), min(to_vals))
    max_val = max(max(from_vals), max(to_vals))

    centers = np.arange(min_val, max_val + bin_size / 2.0, bin_size)

    def make_interval(c: float) -> pd.Interval:
        left = round(c - half, round_decimals)
        right = round(c + half, round_decimals)
        return pd.Interval(left, right, closed=closed)

    from_bins = pd.IntervalIndex([make_interval(cast(float, c)) for c in centers], name="from")
    to_bins = pd.IntervalIndex([make_interval(cast(float, c)) for c in centers], name="to")

    full_idx = pd.MultiIndex.from_product([from_bins, to_bins], names=["from", "to"])

    data = {
        (make_interval(f), make_interval(t)): float(v) for (f, t), v in counter.items()
    }

    s = pd.Series(data, name="value", dtype="float64")
    s = s.reindex(full_idx, fill_value=0.0)

    return s.to_frame()
