"""Helpers for evaluating Woehler (S-N) curves.

The functions in this module are a NumPy-based translation of the TypeScript
implementation used in the UI project.  They provide utilities for computing
load amplitudes for a given number of cycles and for generating a convenient
logarithmic cycle axis.

The central entry points are:

* :class:`WoehlerCurveParams` – container for curve parameters.
* :func:`woehler_loads` – probability-dependent Woehler curve.
* :func:`woehler_loads_basic` – Woehler curve without probability/scattering.
* :func:`woehler_log_space` – helper to create a log-spaced cycle axis.
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from enum import Enum
from math import log10
from typing import Iterable




class MinerType(str, Enum):
    NONE = "none"
    ORIGINAL = "original"
    ELEMENTARY = "elementary"
    HAIBACH = "haibach"


@dataclass
class WoehlerCurveParams:
    sd: float
    nd: float
    k1: float
    k2: float | None = None
    ts: float | None = None
    tn: float | None = None


_DEFAULT_FAILURE_PROBABILITY = 0.5


def _norm_ppf(p: float) -> float:
    """Approximate the inverse CDF (ppf) of the standard normal distribution.

    Implementation based on Peter J. Acklam's algorithm. This avoids adding a
    dependency on SciPy while being sufficiently accurate for engineering
    purposes.
    """

    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


_NATIVE_PPF = 0.0  # _norm_ppf(_DEFAULT_FAILURE_PROBABILITY)


def _scattering_range_to_std(t: float) -> float:
    return 0.39015207303618954 * log10(t)


def _derive_k2(params: WoehlerCurveParams, miner: MinerType) -> float:
    if miner is MinerType.ORIGINAL:
        return float("inf")
    if miner is MinerType.ELEMENTARY:
        return params.k1
    if miner is MinerType.HAIBACH:
        return 2.0 * params.k1 - 1.0
    return float("inf")


def _derive_ts(params: WoehlerCurveParams) -> float:
    if params.ts is not None:
        return params.ts
    if params.tn is not None:
        return float(params.tn) ** (1.0 / params.k1)
    return 1.0


def _derive_tn(params: WoehlerCurveParams) -> float:
    if params.tn is not None:
        return params.tn
    if params.ts is not None:
        return float(params.ts) ** params.k1
    return 1.0


def _make_k(
    src: float, ref: float, params: WoehlerCurveParams, miner: MinerType
) -> float:
    k2_derived = _derive_k2(params, miner)
    if src < ref:
        return k2_derived
    return params.k1


def woehler_loads_basic(
    cycles: Iterable[float] | np.ndarray,
    params: WoehlerCurveParams,
    miner: MinerType = MinerType.NONE,
) -> np.ndarray:
    """Return Woehler curve loads for given cycle counts.

    This variant corresponds to the "native" Woehler curve and does not apply
    any probability or scattering transformation.  It still honours the
    ``miner`` setting and thus the possible change of slope between the
    finite-life and the endurance region.
    """

    cyc = np.asarray(list(cycles), dtype=float)
    if cyc.ndim != 1:
        raise ValueError("cycles must be 1D")

    sd = params.sd
    nd_transformed = params.nd

    sd_transformed = sd

    ref = -nd_transformed
    k_values = np.array(
        [_make_k(-float(c), ref, params, miner) for c in cyc], dtype=float
    )

    loads = np.empty_like(cyc, dtype=float)
    mask_finite = np.isfinite(k_values)
    loads[mask_finite] = sd_transformed * (cyc[mask_finite] / nd_transformed) ** (
        -1.0 / k_values[mask_finite]
    )
    loads[~mask_finite] = sd_transformed

    return loads


def woehler_loads(
    cycles: Iterable[float] | np.ndarray,
    params: WoehlerCurveParams,
    miner: MinerType = MinerType.NONE,
    failure_probability: float = _DEFAULT_FAILURE_PROBABILITY,
) -> np.ndarray:
    """Return Woehler curve loads for given cycle counts.

    Parameters
    ----------
    cycles:
        Iterable of cycle counts (e.g. values from :func:`woehler_log_space`).
    params:
        Woehler curve parameters such as fatigue strength and slopes.
    miner:
        Miner damage rule variant determining the second slope ``k2``.
    failure_probability:
        Target failure probability :math:`P_f` in the interval ``(0, 1)``.

    Notes
    -----
    The implementation mirrors the TypeScript logic from the UI and is
    vectorised for NumPy arrays.  It uses an internal approximation of the
    standard normal inverse CDF and applies the same transformations to
    ``sd`` and ``nd`` that are used in the UI.
    """

    if not (0.0 < failure_probability < 1.0):
        raise ValueError("failure_probability must be in (0, 1)")

    cyc = np.asarray(list(cycles), dtype=float)
    if cyc.ndim != 1:
        raise ValueError("cycles must be 1D")

    goal_ppf = _norm_ppf(failure_probability)

    ts_derived = _derive_ts(params)
    tn_derived = _derive_tn(params)

    sd = params.sd
    nd = params.nd

    # Transform sd
    sd_transformed = sd / 10.0 ** (
        (_NATIVE_PPF - goal_ppf) * _scattering_range_to_std(ts_derived)
    )

    # Transform nd
    transformed_nd = nd / 10.0 ** (
        (_NATIVE_PPF - goal_ppf) * _scattering_range_to_std(tn_derived)
    )
    if sd_transformed != 0.0:
        nd_transformed = transformed_nd * (sd_transformed / sd) ** (-params.k1)
    else:
        nd_transformed = transformed_nd

    ref = -nd_transformed
    k_values = np.array(
        [_make_k(-float(c), ref, params, miner) for c in cyc], dtype=float
    )

    loads = np.empty_like(cyc, dtype=float)
    mask_finite = np.isfinite(k_values)
    loads[mask_finite] = sd_transformed * (cyc[mask_finite] / nd_transformed) ** (
        -1.0 / k_values[mask_finite]
    )
    loads[~mask_finite] = sd_transformed

    return loads


def woehler_log_space(
    minimum: float = 1.0,
    maximum: float = 10.0**8,
    n: int = 101,
) -> np.ndarray:
    """Return logarithmically spaced cycle counts between ``minimum`` and ``maximum``.

    This is a small convenience wrapper around :func:`numpy.logspace` with
    defaults that are suitable for typical Woehler curves.
    """

    if n < 2:
        raise ValueError("n must be >= 2")

    log_min = log10(minimum)
    log_max = log10(maximum)
    # step = (log_max - log_min) / (n - 1)
    exponents = np.linspace(log_min, log_max, num=n)
    return 10.0**exponents
