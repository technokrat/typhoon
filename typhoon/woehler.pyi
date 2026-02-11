from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np

class MinerType(str, Enum):
    NONE: str
    ORIGINAL: str
    ELEMENTARY: str
    HAIBACH: str

@dataclass
class WoehlerCurveParams:
    sd: float
    nd: float
    k1: float
    k2: float | None = None
    ts: float | None = None
    tn: float | None = None

    def with_predamage(self, d_predamage: float, q: float | None = ...) -> WoehlerCurveParams: ...

def woehler_loads_basic(
    cycles: Iterable[float] | np.ndarray,
    params: WoehlerCurveParams,
    miner: MinerType = ...,
) -> np.ndarray: ...
def woehler_loads(
    cycles: Iterable[float] | np.ndarray,
    params: WoehlerCurveParams,
    miner: MinerType = ...,
    failure_probability: float = ...,
) -> np.ndarray: ...
def woehler_log_space(
    minimum: float = 1.0,
    maximum: float = 10.0**8,
    n: int = 101,
) -> np.ndarray: ...
