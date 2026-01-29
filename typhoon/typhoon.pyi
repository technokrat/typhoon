from collections import Counter
from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Array1D: TypeAlias = NDArray[np.float32]

def init_tracing() -> None: ...


def rainflow(
    waveform: Array1D,
    last_peaks: Array1D | None = ..., 
    bin_size: float = ..., 
    threshold: float | None = ..., 
    min_chunk_size: int = ...,
) -> tuple[dict[tuple[float, float], int], Array1D]:
    ...


def goodman_transform(
    cycles: Mapping[tuple[float, float], float] | Mapping[tuple[float, float], int],
    m: float,
    m2: float | None = ..., 
) -> dict[float, float]:
    ...


def summed_histogram(
    hist: Mapping[float, float] | Mapping[float, int],
) -> list[tuple[float, float]]:
    ...


class RainflowContext:
    def __init__(self, bin_size: float = ..., threshold: float = ...) -> None: ...

    def process(self, waveform: Array1D) -> None: ...
    def reset(self) -> None: ...

    def cycles_len(self) -> int: ...
    def get_last_peaks(self) -> Array1D: ...

    def to_dict(self) -> dict[tuple[float, float], int]: ...
    def to_counter(self) -> Counter[tuple[float, float]]: ...
    def to_heatmap(self) -> tuple[np.ndarray[Any, np.dtype[np.float64]], Array1D]: ...

    def goodman_transform(
        self,
        m: float,
        m2: float | None = ...,
        include_half_cycles: bool = ...,
    ) -> dict[float, float]: ...

    def summed_histogram(
        self,
        m: float,
        m2: float | None = ...,
        include_half_cycles: bool = ...,
    ) -> list[tuple[float, float]]: ...
