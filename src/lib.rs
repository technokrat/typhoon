use rayon::prelude::*;
use tracing_subscriber::fmt::format::FmtSpan;

use pyo3::prelude::*;

use numpy::ndarray::{ArrayView1, Axis};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::types::PyDict;

use ahash::AHashMap;
use ordered_float::OrderedFloat;

type CyclesKey = (OrderedFloat<f64>, OrderedFloat<f64>);
type CyclesMap = AHashMap<(OrderedFloat<f64>, OrderedFloat<f64>), usize>;

use std::sync::Once;
use tracing::{instrument, span, trace, Level};
use tracing_subscriber::fmt;

static TRACING_INIT: Once = Once::new();

#[pyfunction]
fn init_tracing() {
    TRACING_INIT.call_once(|| {
        let subscriber = fmt()
            .with_max_level(Level::TRACE)
            .with_writer(std::io::stdout)
            .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT)
            .compact()
            .finish();

        tracing::subscriber::set_global_default(subscriber)
            .expect("Failed to set tracing subscriber");
    });
}

fn quantize(x: f64, bin_size: f64) -> f64 {
    let half_bin = bin_size / 2.0;
    let shifted = x + half_bin.copysign(x); // shift by half a bin in the direction of x
    (shifted / bin_size).floor() * bin_size
}

#[instrument(fields(), skip_all)]
fn peak_peak_and_rainflow_counting(
    waveform: ArrayView1<f64>,
    last_peaks: Option<ArrayView1<f64>>,
    bin_size: f64,
) -> (CyclesMap, Vec<f64>) {
    // Peak-Peak Waveform Construction
    trace!("peak-peak waveform construction");

    let last_peaks_array = match last_peaks {
        Some(arr) => arr,
        None => ArrayView1::from(&[]),
    };

    let mut peaks = Vec::with_capacity(last_peaks_array.len() + waveform.len());
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

    // Rainflow Counting with windowed 4 point method
    trace!("rainflow counting");

    let mut cycles: CyclesMap = CyclesMap::new();

    while peaks.len() > 3 {
        let mut i = 0;
        let mut cycle_found = false;
        let mut new_peaks = Vec::with_capacity(peaks.len() / 2);

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
                new_peaks.extend_from_slice(&peaks[i..=peaks.len() - 1]);
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

    trace!("rainflow counting finished");

    return (cycles, peaks);
}

#[pyfunction]
fn rainflow<'py>(
    py: Python<'py>,
    waveform: PyReadonlyArray1<f64>,
    last_peaks: Option<PyReadonlyArray1<f64>>,
    bin_size: Option<f64>,
    min_chunk_size: Option<usize>,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyArray1<f64>>)> {
    let span = span!(Level::TRACE, "rainflow");
    let _enter = span.enter();

    let bin_size = bin_size.unwrap_or(0.0);
    let waveform = waveform.as_array();
    let num_cores = rayon::current_num_threads();
    let min_chunk_size = min_chunk_size.unwrap_or(64 * 1024); // Set your minimum chunk size

    let chunk_size = std::cmp::max(waveform.len() / num_cores, min_chunk_size);
    let chunks: Vec<_> = waveform.axis_chunks_iter(Axis(0), chunk_size).collect();

    // Prepare last_peaks for the first chunk, None for others
    let mut last_peaks_vec = vec![last_peaks.as_ref().map(|arr| arr.as_array())];
    last_peaks_vec.extend((1..chunks.len()).map(|_| None));

    // Parallel processing
    let (mut total_cycles, all_peaks) = chunks
        .par_iter()
        .zip(last_peaks_vec.into_par_iter())
        .map(|(chunk, last_peaks)| {
            peak_peak_and_rainflow_counting(chunk.view(), last_peaks, bin_size)
        })
        .reduce(
            || (CyclesMap::new(), Vec::new()),
            |(mut acc_cycles, mut acc_peaks), (new_cycles, new_peaks)| {
                for (k, v) in new_cycles {
                    *acc_cycles.entry(k).or_insert(0) += v;
                }
                acc_peaks.extend(new_peaks);
                (acc_cycles, acc_peaks)
            },
        );

    let (final_cycles, final_peaks) =
        peak_peak_and_rainflow_counting(ArrayView1::from(&all_peaks), None, bin_size);
    for (k, v) in final_cycles {
        *total_cycles.entry(k).or_insert(0) += v;
    }

    let py_cycles: Bound<'_, PyDict> = PyDict::new(py);
    for ((k1, k2), v) in &total_cycles {
        let key = (k1.into_inner(), k2.into_inner());
        py_cycles.set_item(key, *v)?;
    }

    let py_peaks = PyArray1::from_vec(py, final_peaks);

    Ok((py_cycles, py_peaks))
}

/// A Python module implemented in Rust.
#[pymodule]
fn typhoon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(rainflow, m)?)?;

    Ok(())
}
