from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal, Mapping, Tuple

import numpy as np
import pandas as pd

CycleKey = Tuple[float, float]
CycleCounter = Counter[CycleKey]

def merge_cycle_counters(
    counters: Iterable[Mapping[CycleKey, float]],
) -> CycleCounter: ...
def add_residual_half_cycles(
    counter: Mapping[CycleKey, float],
    residual_peaks: np.ndarray,
) -> CycleCounter: ...
def counter_to_full_interval_df(
    counter: Mapping[CycleKey, float],
    bin_size: float = 0.1,
    closed: Literal["left", "right"] = "right",
    round_decimals: int = 12,
) -> pd.DataFrame: ...
