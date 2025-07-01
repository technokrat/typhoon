from typing import Annotated, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

Array1D = Annotated[NDArray[np.float64], (None,)]

def rainflow(
    waveform: Array1D,
    last_peaks: Optional[Array1D] = None,
    bin_size: float = 0,
    min_chunk_size: float = 64 * 1024,
) -> Tuple[Dict[Tuple[float, float], int], Array1D]: ...