from typing import Annotated, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

Array1D = Annotated[NDArray[np.float64], (None,)]

def rainflow(
    waveform: Array1D,
    last_peaks: Optional[Array1D] = None,
    bin_size: float = ...
) -> Tuple[Dict[Tuple[float, float], int], Array1D]: ...