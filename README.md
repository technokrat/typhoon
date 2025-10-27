# typhoon
[![CI](https://github.com/technokrat/typhoon/actions/workflows/CI.yml/badge.svg)](https://github.com/technokrat/typhoon/actions/workflows/CI.yml)

Typhoon is a rainflow counting Python module written in Rust by Markus Wegmann (mw@technokrat.ch)

It uses a new windowed four-point counting method which can be run in parallel on multiple cores and allows for chunk-based sample stream processing, preserving half cycles for future chunks.

It is therefore intended for real-time processing of load captures and serves as as crucial part of i-Spring's in-edge data processing chain.

## Installation
Add the package `typhoon-rainflow` to your Python project, e.g.

```python
poetry add typhoon-rainflow
```

## Example Usage

### Single Waveform

```python
import typhoon
import numpy as np

waveform = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0])
last_peaks = np.array([])
cycles, remaining_peaks = typhoon.rainflow(
    waveform=waveform,
    last_peaks=last_peaks,
    bin_size=1.0,
)

print("Cycles:", cycles)
print("Remaining Peaks:", remaining_peaks)
```

### Multiple Waveform Chunks

```python
waveform1 = np.array([0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 3.0, 4.0])
waveform2 = np.array([3.0, 5.0])

cycles1, remaining_peaks = typhoon.rainflow(
    waveform=waveform1,
    last_peaks=None,
    bin_size=1.0,
)

print("Waveform 1:", waveform1)
print("Cycles 1:", cycles1)
print("Remaining Peaks 1:", remaining_peaks)

cycles2, remaining_peaks = typhoon.rainflow(
    waveform=waveform2,
    last_peaks=remaining_peaks,
    bin_size=1.0,
)

print("Waveform 2:", waveform2)
print("Cycles 2:", cycles2)
print("Remaining Peaks 2:", remaining_peaks)

cycles_merged = Counter(cycles1) + Counter(cycles2)

print("Merged Cycles:", cycles_merged)
```

## Testing

```sh
pipx install nox

nox -s build
nox -s test
nox -s develop
```