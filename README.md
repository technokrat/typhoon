# typhoon

[![CI](https://github.com/technokrat/typhoon/actions/workflows/CI.yml/badge.svg)](https://github.com/technokrat/typhoon/actions/workflows/CI.yml) ![PyPI - Version](https://img.shields.io/pypi/v/typhoon-rainflow)

Typhoon is a rainflow counting Python module written in Rust by Markus Wegmann (mw@technokrat.ch).

It uses a new windowed four-point counting method which can be run in parallel on multiple cores and allows for chunk-based sample stream processing, preserving half cycles for future chunks.

It is therefore intended for real-time processing of load captures and serves as as crucial part of i-Spring's in-edge data processing chain.

## Installation

Add the package `typhoon-rainflow` to your Python project, e.g.

```python
poetry add typhoon-rainflow
```

## Python API

The Python package exposes two main namespaces:

- `typhoon.typhoon`: low-level, performance‑critical functions implemented in Rust.
- `typhoon.helper`: convenience utilities for working with the rainflow output.

The top-level package re-exports everything from `typhoon.typhoon`, so you can either

```python
import typhoon            # recommended for normal use
from typhoon import rainflow, goodman_transform
```

or

```python
from typhoon.typhoon import rainflow
from typhoon import helper  # for helper utilities
```

### Core functions (`typhoon.typhoon`)

All arguments are keyword-compatible with the examples below.

- `init_tracing() -> None`
  - Initialize verbose tracing/logging from the Rust implementation.
  - Intended mainly for debugging and performance analysis; it writes to stdout.

- `rainflow(waveform, last_peaks=None, bin_size=0.0, threshold=None, min_chunk_size=64*1024)`
  - Perform windowed four-point rainflow counting on a 1D NumPy waveform.
  - `waveform`: 1D `numpy.ndarray` of `float32` or `float64`.
  - `last_peaks`: optional 1D array of peaks from the previous chunk (for streaming).
  - `bin_size`: bin width for quantizing ranges; `0.0` disables quantization.
  - `threshold`: minimum cycle amplitude to count; default `0.0`.
  - `min_chunk_size`: minimum chunk size for internal parallelization.
  - Returns `(cycles, residual_peaks)` where
    - `cycles` is a dict `{(s_lower, s_upper): count}` and
    - `residual_peaks` is a 1D NumPy array of remaining peaks to pass to the next call.

- `goodman_transform(cycles, m, m2=None, bin_size_compensation=0.0)`
  - Apply a (piecewise) Goodman-like mean stress correction to rainflow cycles.
  - `cycles`: mapping `{(s_lower, s_upper): count}` (e.g. first return value of `rainflow`).
  - `m`: main slope/parameter.
  - `m2`: optional secondary slope; defaults to `m / 3` if omitted.
  - `bin_size_compensation`: optional widening of each cycle by `±bin_size_compensation/2` (worst-case compensation for quantization to bin centers).
  - Returns a dict `{s_a_ers: count}` where `s_a_ers` is the equivalent range.

- `summed_histogram(hist)`
  - Build a descending cumulative histogram from the Goodman-transformed result.
  - `hist`: mapping `{s_a_ers: count}` such as returned from `goodman_transform`.
  - Returns a list of `(s_a_ers, cumulative_count)` pairs sorted from high to low range.

- `fkm_miner_damage(goodman_result, n_d, sigma_d, k, q=None, mode=MinerDamageMode.Modified) -> float`
  - Compute a Miner damage coefficient $D$ from a Goodman-transformed collective.
  - `goodman_result`: mapping `{sigma_a: count}` as returned from `goodman_transform`.
  - `n_d`: $N_D$ (cycles allowed at endurance limit).
  - `sigma_d`: $\sigma_D$ (endurance limit level).
  - `k`: Woehler exponent $k$.
  - `q`: exponent modifier for amplitudes below $\sigma_D$; defaults to `k - 1`.
  - `mode`:
    - `MinerDamageMode.Modified` (default): returns $D_{MM}$ (FKM modified Miner).
    - `MinerDamageMode.Original`: returns $D_{OM}$ (original Miner, ignoring amplitudes below $\sigma_D$).

### Stateful streaming (`RainflowContext`)

If you process signals chunk-by-chunk, repeatedly calling `rainflow()` and merging dicts/Counters can become a bottleneck.

`RainflowContext` keeps the accumulated cycle map and the residual `last_peaks` inside the Rust extension, so each new chunk only updates the existing state.

Key methods:

- `process(waveform)`: update the internal state from one waveform chunk.
- `to_counter()`: export the accumulated cycles as a Python `collections.Counter`.
- `to_heatmap(include_half_cycles=False)`: export a dense 2D NumPy array for plotting (and the corresponding bin centers).
  - When `include_half_cycles=True`, the current residual `last_peaks` are treated as half-cycles (each adjacent peak-pair contributes `0.5`).
- `goodman_transform(m, m2=None, include_half_cycles=False, bin_size_compensation=0.0)`: Goodman transform directly on the internal state.
  - When `include_half_cycles=True`, the current residual `last_peaks` are treated as half-cycles (each adjacent peak-pair contributes `0.5`).
- `summed_histogram(m, m2=None, include_half_cycles=False, bin_size_compensation=0.0)`: convenience wrapper that returns the descending cumulative histogram (same format as `typhoon.summed_histogram`).
- `fkm_miner_damage(m, n_d, sigma_d, k, m2=None, include_half_cycles=False, bin_size_compensation=0.0, q=None, mode=MinerDamageMode.Modified)`: compute Miner damage directly from the internal accumulated cycles.

Example:

```python
import numpy as np
import typhoon

ctx = typhoon.RainflowContext(bin_size=1.0, threshold=0.0)

for chunk in chunks:  # iterable of 1D numpy arrays
    ctx.process(chunk)

# Export accumulated cycles
cycles = ctx.to_counter()

# Goodman transform (optionally including the current residual half-cycles)
hist = ctx.goodman_transform(m=0.3, include_half_cycles=True)

# Summed histogram directly from the context
summed = ctx.summed_histogram(m=0.3, include_half_cycles=True)

# Heatmap export for matplotlib (optionally include residual half-cycles)
heatmap, bins = ctx.to_heatmap(include_half_cycles=True)

# Example plotting
# import matplotlib.pyplot as plt
# plt.imshow(heatmap, origin="lower")
# plt.xticks(range(len(bins)), bins, rotation=90)
# plt.yticks(range(len(bins)), bins)
# plt.xlabel("to")
# plt.ylabel("from")
# plt.colorbar(label="count")
# plt.tight_layout()
# plt.show()
```

### Helper utilities (`typhoon.helper`)

The helper module provides convenience tools for post-processing and analysis.

- `merge_cycle_counters(counters)`
  - Merge multiple `dict`/`Counter` objects of the form `{(from, to): count}`.
  - Useful when combining rainflow results from multiple chunks or channels.

- `add_residual_half_cycles(counter, residual_peaks)`
  - Convert the trailing `residual_peaks` from `rainflow` into half-cycles and add them to an existing counter.
  - Each adjacent pair of peaks `(p_i, p_{i+1})` contributes `0.5` to the corresponding cycle key.

- `counter_to_full_interval_df(counter, bin_size=0.1, closed="right", round_decimals=12)`
  - Convert a sparse `(from, to): count` mapping into a dense 2D `pandas.DataFrame` over all intervals.
  - Returns a DataFrame with a `(from, to)` `MultiIndex` of `pd.Interval` and a single `"value"` column.

### Woehler curves (`typhoon.woehler`)

The `typhoon.woehler` module provides helpers for evaluating S–N (Woehler) curves.

Key entry points are:

- `WoehlerCurveParams(sd, nd, k1, k2=None, ts=None, tn=None)` - Container for the curve parameters: - `sd`: fatigue strength at `nd` cycles for the reference failure probability. - `nd`: reference number of cycles (e.g. 1e6). - `k1`: slope in the finite-life region. - `k2`: optional slope in the endurance region; derived from the Miner
  rule if omitted. - `ts` / `tn`: optional scattering parameters controlling probability
  transforms of `sd` and `nd`.
- `WoehlerCurveParams.with_predamage(d_predamage, q=None)` - Returns a new set of curve parameters modified by a pre-damage value $D_{predamage}$. - Uses the FKM-style transformation: - $\sigma_{D,dam} = \sigma_D\,(1-D_{predamage})^{1/q}$ - $N_{D,dam} = N_D\,(\sigma_{D,dam}/\sigma_D)^{-(k-q)}$ with $k=k_1$ - If `q` is omitted, defaults to `k1 - 1`.
- `MinerType` enum - Miner damage rule variant that determines the second slope `k2`:
  `NONE`, `ORIGINAL`, `ELEMENTARY`, `HAIBACH`.
- `woehler_log_space(minimum=1.0, maximum=1e8, n=101)` - Convenience helper to generate a logarithmically spaced cycle axis for
  plotting Woehler curves.
- `woehler_loads_basic(cycles, params, miner=MinerType.NONE)` - Compute a "native" Woehler curve without probability/scattering
  transformation, but honouring the selected Miner type.
- `woehler_loads(cycles, params, miner=MinerType.NONE, failure_probability=0.5)` - Compute a probability-dependent Woehler curve using an internal
  approximation of the normal inverse CDF.

## Example Usage

### Basic rainflow counting

```python
import numpy as np
import typhoon

waveform = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0], dtype=np.float32)

cycles, residual_peaks = typhoon.rainflow(
        waveform=waveform,
        last_peaks=None,
        bin_size=1.0,
)

print("Cycles:", cycles)
print("Residual peaks:", residual_peaks)
```

### Streaming / chunked processing with helpers

```python
from collections import Counter

import numpy as np
import typhoon
from typhoon import helper

waveform1 = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0], dtype=np.float32)
waveform2 = np.array([3.0, 5.0, 4.0, 2.0], dtype=np.float32)

# First chunk
cycles1, residual1 = typhoon.rainflow(waveform1, last_peaks=None, bin_size=1.0)

# Second chunk, passing residual peaks from the first
cycles2, residual2 = typhoon.rainflow(waveform2, last_peaks=residual1, bin_size=1.0)

# Merge cycle counts from both chunks
merged = helper.merge_cycle_counters([cycles1, cycles2])

# Optionally add remaining half-cycles from the final residual peaks
merged_with_residuals = helper.add_residual_half_cycles(merged, residual2)

print("Merged cycles:", merged_with_residuals)
```

### Goodman transform and summed histogram

```python
import typhoon
from typhoon import helper

cycles, residual_peaks = typhoon.rainflow(waveform, last_peaks=None, bin_size=1.0)

# Apply Goodman transform
hist = typhoon.goodman_transform(cycles, m=0.3)

# Summed histogram from the Goodman result
summed = typhoon.summed_histogram(hist)

print("Goodman result:", hist)
print("Summed histogram:", summed)
```

### FKM Modified Miner damage (from Goodman result)

```python
import typhoon

# hist is the Goodman-transformed collective: {sigma_a: count}
hist = typhoon.goodman_transform(cycles, m=0.3)

# FKM modified Miner damage (D_MM)
d_mm = typhoon.fkm_miner_damage(hist, n_d=1e6, sigma_d=100.0, k=5.0)

# Original Miner damage (D_OM) only (ignore amplitudes below sigma_d)
d_om = typhoon.fkm_miner_damage(
    hist,
    n_d=1e6,
    sigma_d=100.0,
    k=5.0,
    mode=typhoon.MinerDamageMode.Original,
)

print("D_MM:", d_mm)
print("D_OM:", d_om)
```

## Testing

```sh
pipx install nox

nox -s build
nox -s test
nox -s develop
```
