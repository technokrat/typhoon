from typing import Dict, Optional, Tuple, TypeAlias
import numpy as np
from numpy.typing import NDArray

# Use a simple alias Pylance understands well.
Array1D: TypeAlias = NDArray[np.float32]


def rainflow(
    waveform: Array1D,
    last_peaks: Optional[Array1D] = None,
    bin_size: float = 0,
    min_chunk_size: int = 64 * 1024,
) -> Tuple[Dict[Tuple[float, float], int], Array1D]: ...


def init_tracing() -> None: ...
