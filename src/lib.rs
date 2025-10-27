use rayon::prelude::*;
use tracing_subscriber::fmt::format::FmtSpan;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use numpy::ndarray::{Array1, ArrayView1, Axis};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::types::PyDict;

use ahash::AHashMap;
use ordered_float::OrderedFloat;

type WaveformSampleValueType = f32;

type CyclesKey = (
    OrderedFloat<WaveformSampleValueType>,
    OrderedFloat<WaveformSampleValueType>,
);
type CyclesMap = AHashMap<
    (
        OrderedFloat<WaveformSampleValueType>,
        OrderedFloat<WaveformSampleValueType>,
    ),
    usize,
>;

use std::collections::VecDeque;
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

fn quantize(
    x: WaveformSampleValueType,
    bin_size: WaveformSampleValueType,
) -> WaveformSampleValueType {
    let half_bin = bin_size / 2.0;
    let shifted = x + half_bin.copysign(x); // shift by half a bin in the direction of x
    (shifted / bin_size).floor() * bin_size
}

#[instrument(fields(), skip_all)]
fn peak_peak_and_rainflow_counting(
    waveform: ArrayView1<WaveformSampleValueType>,
    last_peaks: Option<ArrayView1<WaveformSampleValueType>>,
    bin_size: WaveformSampleValueType,
) -> (CyclesMap, Vec<WaveformSampleValueType>) {
    // Peak-Peak Waveform Construction
    trace!("peak-peak waveform construction");

    let waveform: Array1<WaveformSampleValueType> = if let Some(last_peaks) = last_peaks {
        let mut combined = last_peaks.to_owned().to_vec();
        combined.extend_from_slice(waveform.as_slice().unwrap());
        combined.into()
    } else {
        waveform.to_owned()
    };

    let waveform = waveform.view();

    let n = waveform.len();
    let mut signums = Vec::with_capacity(n.saturating_sub(1));
    if n > 1 {
        for i in 0..(n - 1) {
            let diff = waveform[i + 1] - waveform[i];
            let mut sign = diff.signum();
            if sign == 0.0 {
                sign = 1.0; // Assume positive signum for zero difference
            }
            signums.push(sign);
        }
    }

    let mut peaks: VecDeque<WaveformSampleValueType> = VecDeque::new();
    let n = waveform.len();
    if n > 0 {
        peaks.push_back(waveform[0]);
        let mut last_sign = if !signums.is_empty() { signums[0] } else { 0.0 };
        for i in 1..signums.len() {
            if signums[i] != last_sign {
                peaks.push_back(waveform[i]);
                last_sign = signums[i];
            }
        }
        if n > 1 {
            peaks.push_back(waveform[n - 1]);
        }
    }

    // Rainflow Counting with windowed 4 point method
    trace!("rainflow counting");

    let mut cycles: CyclesMap = CyclesMap::new();

    let mut peak_stack: Vec<WaveformSampleValueType> = Vec::with_capacity(n / 2);

    'sample_loop: while !peaks.is_empty() {
        peak_stack.push(peaks.pop_front().unwrap());

        while peak_stack.len() > 3 {
            let len = peak_stack.len();
            let lower = peak_stack[len - 1].min(peak_stack[len - 4]);
            let upper = peak_stack[len - 1].max(peak_stack[len - 4]);

            let b = peak_stack[len - 3];
            let c = peak_stack[len - 2];

            if b >= lower && b <= upper && c >= lower && c <= upper {
                let key: CyclesKey;

                if bin_size > 0.0 {
                    let from: WaveformSampleValueType;
                    let to: WaveformSampleValueType;

                    if (c - b).abs() < (bin_size / 2.0) {
                        let abs_max: WaveformSampleValueType =
                            if b.abs() >= c.abs() { b } else { c };

                        from = abs_max;
                        to = abs_max;
                    } else {
                        from = b;
                        to = c;
                    }

                    let quantized_from = quantize(from, bin_size);
                    let quantized_to = quantize(to, bin_size);
                    key = (
                        OrderedFloat::from(quantized_from),
                        OrderedFloat::from(quantized_to),
                    );
                } else {
                    key = (OrderedFloat::from(b), OrderedFloat::from(c));
                }

                *cycles.entry(key).or_insert(0) += 1;

                let d = peak_stack.pop().unwrap();
                peak_stack.pop();
                peak_stack.pop();
                peak_stack.push(d);
            } else {
                continue 'sample_loop;
            }
        }
    }

    trace!("rainflow counting finished");

    (cycles, peak_stack)
}

#[pyfunction]
#[pyo3(signature = (waveform, last_peaks=None, bin_size=0.0, min_chunk_size=64*1024))]
fn rainflow<'py>(
    py: Python<'py>,
    waveform: Bound<'py, PyAny>,
    last_peaks: Option<Bound<'py, PyAny>>,
    bin_size: Option<WaveformSampleValueType>,
    min_chunk_size: Option<usize>,
) -> PyResult<(
    Bound<'py, PyDict>,
    Bound<'py, PyArray1<WaveformSampleValueType>>,
)> {
    let span = span!(Level::TRACE, "rainflow");
    let _enter = span.enter();

    let bin_size = bin_size.unwrap_or(0.0);
    // Accept np.float32 or np.float64 and convert to f32 if needed
    let waveform: Array1<WaveformSampleValueType> =
        if let Ok(arr_f32) = waveform.extract::<PyReadonlyArray1<WaveformSampleValueType>>() {
            arr_f32.as_array().to_owned()
        } else if let Ok(arr_f64) = waveform.extract::<PyReadonlyArray1<f64>>() {
            arr_f64.as_array().mapv(|x| x as WaveformSampleValueType)
        } else {
            return Err(PyTypeError::new_err(
                "waveform must be a 1D numpy array of dtype float32 or float64",
            ));
        };

    let num_cores = rayon::current_num_threads();
    let min_chunk_size = min_chunk_size.unwrap_or(64 * 1024); // default consistent with Python stub

    let waveform_view = waveform.view();
    let chunk_size = std::cmp::max(waveform_view.len() / num_cores, min_chunk_size);
    let chunks: Vec<_> = waveform_view
        .axis_chunks_iter(Axis(0), chunk_size)
        .collect();

    // Prepare last_peaks for the first chunk, None for others
    let last_peaks_owned: Option<Array1<WaveformSampleValueType>> = if let Some(lp_any) = last_peaks
    {
        if let Ok(arr_f32) = lp_any.extract::<PyReadonlyArray1<WaveformSampleValueType>>() {
            Some(arr_f32.as_array().to_owned())
        } else if let Ok(arr_f64) = lp_any.extract::<PyReadonlyArray1<f64>>() {
            Some(arr_f64.as_array().mapv(|x| x as WaveformSampleValueType))
        } else {
            return Err(PyTypeError::new_err(
                "last_peaks must be a 1D numpy array of dtype float32 or float64",
            ));
        }
    } else {
        None
    };

    let mut last_peaks_vec: Vec<Option<ArrayView1<WaveformSampleValueType>>> =
        vec![last_peaks_owned.as_ref().map(|a| a.view())];
    last_peaks_vec.extend((1..chunks.len()).map(|_| None));

    // Parallel processing
    let (mut total_cycles, all_peaks) = if chunks.len() > 1 {
        chunks
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
            )
    } else {
        // Fallback to single-threaded if only one chunk
        peak_peak_and_rainflow_counting(chunks[0].view(), last_peaks_vec[0], bin_size)
    };

    let (final_cycles, final_peaks) = if chunks.len() > 1 {
        peak_peak_and_rainflow_counting(ArrayView1::from(&all_peaks), None, bin_size)
    } else {
        (CyclesMap::new(), all_peaks)
    };

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
