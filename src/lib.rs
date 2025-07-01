use std::collections::HashMap;

use pyo3::prelude::*;

use numpy::ndarray::{ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use ordered_float::OrderedFloat;
use pyo3::types::PyDict;

type CyclesKey = (OrderedFloat<f64>, OrderedFloat<f64>);
type CyclesMap = HashMap<CyclesKey, i64>;

fn quantize(x: f64, bin_size: f64) -> f64 {
    let half_bin = bin_size / 2.0;
    let shifted = x + half_bin.copysign(x); // shift by half a bin in the direction of x
    (shifted / bin_size).floor() * bin_size
}

#[pyfunction]
fn rainflow<'py>(
    py: Python<'py>,
    waveform: PyReadonlyArray1<f64>,
    last_peaks: Option<PyReadonlyArray1<f64>>,
    bin_size: Option<f64>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyArray1<f64>>)> {
    let bin_size: f64 = bin_size.unwrap_or_default();

    let waveform = waveform.as_array();
    let last_peaks_vec: Vec<f64> = last_peaks
        .map(|arr| arr.as_array().to_owned().into_raw_vec_and_offset().0)
        .unwrap_or_else(Vec::new);
    let last_peaks_array = ArrayView1::from(&last_peaks_vec);

    let mut peaks = Vec::new();
    let last_len = last_peaks_array.len();
    if last_len > 1 {
        peaks.extend(last_peaks_array.iter().take(last_len).cloned());
    }
    let n = waveform.len();

    let mut last_dir = 0;

    if last_len > 1 {
        let diff: f64 = last_peaks_array[last_len - 1] - last_peaks_array[last_len - 2];
        last_dir = diff.signum() as i32;
    }

    if n > 0 {
        if last_len > 0 {
            let diff = waveform[0] - last_peaks_array[last_len - 1];
            let dir = diff.signum() as i32;
            if dir != 0 && dir != last_dir && last_dir != 0 {
            } else {
                peaks.remove(last_len - 1);
            }
            last_dir = dir;
        } else {
            peaks.push(waveform[0]);
        }
    }

    for i in 1..n {
        let diff = waveform[i] - waveform[i - 1];
        let dir = diff.signum() as i32;
        if dir != 0 && dir != last_dir && last_dir != 0 {
            peaks.push(waveform[i - 1]);
        }
        last_dir = dir;
    }

    if n > 1 {
        peaks.push(waveform[n - 1]);
    }

    let mut cycles: CyclesMap = CyclesMap::new();

    while peaks.len() > 3 {
        let mut i = 0;
        let mut cycle_found = false;
        let mut new_peaks = Vec::new();

        loop {
            let (lower, upper) = if peaks[i] <= peaks[i + 3] {
                (peaks[i], peaks[i + 3])
            } else {
                (peaks[i + 3], peaks[i])
            };

            if peaks[i + 1] >= lower
                && peaks[i + 1] <= upper
                && peaks[i + 2] >= lower
                && peaks[i + 2] <= upper
            {
                let key: CyclesKey;

                if bin_size > 0.0 {
                    let from: f64;
                    let to: f64;

                    if (peaks[i + 2] - peaks[i + 1]).abs() < (bin_size / 2.0) {
                        let abs_max: f64 = if peaks[i + 1].abs() >= peaks[i + 2].abs() {
                            peaks[i + 1]
                        } else {
                            peaks[i + 2]
                        };

                        from = abs_max;
                        to = abs_max;
                    } else {
                        from = peaks[i + 1];
                        to = peaks[i + 2];
                    }

                    let quantized_from = quantize(from, bin_size);
                    let quantized_to = quantize(to, bin_size);

                    key = (
                        OrderedFloat::from(quantized_from),
                        OrderedFloat::from(quantized_to),
                    );
                } else {
                    key = (
                        OrderedFloat::from(peaks[i + 1]),
                        OrderedFloat::from(peaks[i + 2]),
                    );
                }

                *cycles.entry(key).or_insert(0) += 1;

                cycle_found = true;

                new_peaks.push(peaks[i]);

                i += 3;
            } else {
                new_peaks.push(peaks[i]);
                i += 1;
            }

            if i >= (peaks.len() - 3) {
                new_peaks.extend_from_slice(&peaks[i..=peaks.len()-1]);
                break;
            }
        }

        if cycle_found {
            peaks = new_peaks;
            continue;
        } else {
            break;
        }
    }

    let py_cycles: Bound<'_, PyDict> = PyDict::new(py);
    for ((k1, k2), v) in &cycles {
        let key = (k1.into_inner(), k2.into_inner());
        py_cycles.set_item(key, *v)?;
    }

    let py_peaks = PyArray1::from_vec(py, peaks);

    Ok((py_cycles, py_peaks))
}

/// A Python module implemented in Rust.
#[pymodule]
fn typhoon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rainflow, m)?)?;
    Ok(())
}
