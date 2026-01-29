use rayon::prelude::*;
use tracing_subscriber::fmt::format::FmtSpan;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use numpy::ndarray::{Array1, Array2, ArrayView1, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::types::{PyDict, PyList};

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
type CyclesMapWithHalfCycles = AHashMap<
    (
        OrderedFloat<WaveformSampleValueType>,
        OrderedFloat<WaveformSampleValueType>,
    ),
    WaveformSampleValueType,
>;

use std::collections::VecDeque;
use std::sync::Once;
use tracing::{instrument, span, trace, Level};
use tracing_subscriber::fmt;

use std::collections::BTreeSet;

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
    let shifted = x + half_bin; // shift by half a bin
    (shifted / bin_size).floor() * bin_size
}

fn extract_1d_f32(any: &Bound<'_, PyAny>) -> PyResult<Array1<WaveformSampleValueType>> {
    if let Ok(arr_f32) = any.extract::<PyReadonlyArray1<WaveformSampleValueType>>() {
        Ok(arr_f32.as_array().to_owned())
    } else if let Ok(arr_f64) = any.extract::<PyReadonlyArray1<f64>>() {
        Ok(arr_f64.as_array().mapv(|x| x as WaveformSampleValueType))
    } else {
        Err(PyTypeError::new_err(
            "expected a 1D numpy array of dtype float32 or float64",
        ))
    }
}

#[instrument(fields(), skip_all)]
fn peak_peak_and_rainflow_counting_inplace_with_cycles(
    waveform: ArrayView1<WaveformSampleValueType>,
    last_peaks: Option<ArrayView1<WaveformSampleValueType>>,
    bin_size: WaveformSampleValueType,
    threshold: WaveformSampleValueType,
    cycles: &mut CyclesMap,
) -> Vec<WaveformSampleValueType> {
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

                let difference = (c - b).abs();
                if difference >= threshold {
                    if bin_size > 0.0 {
                        let from: WaveformSampleValueType;
                        let to: WaveformSampleValueType;

                        if difference < (bin_size / 2.0) {
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
                }

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

    peak_stack
}

#[instrument(fields(), skip_all)]
fn peak_peak_and_rainflow_counting(
    waveform: ArrayView1<WaveformSampleValueType>,
    last_peaks: Option<ArrayView1<WaveformSampleValueType>>,
    bin_size: WaveformSampleValueType,
    threshold: WaveformSampleValueType,
) -> (CyclesMap, Vec<WaveformSampleValueType>) {
    let mut cycles: CyclesMap = CyclesMap::new();
    let peaks = peak_peak_and_rainflow_counting_inplace_with_cycles(
        waveform,
        last_peaks,
        bin_size,
        threshold,
        &mut cycles,
    );
    (cycles, peaks)
}

#[pyclass]
struct RainflowContext {
    bin_size: WaveformSampleValueType,
    threshold: WaveformSampleValueType,
    cycles: CyclesMap,
    last_peaks: Vec<WaveformSampleValueType>,
}

#[pymethods]
impl RainflowContext {
    #[new]
    #[pyo3(signature = (bin_size=0.0, threshold=0.0))]
    fn new(bin_size: WaveformSampleValueType, threshold: WaveformSampleValueType) -> Self {
        Self {
            bin_size,
            threshold,
            cycles: CyclesMap::new(),
            last_peaks: Vec::new(),
        }
    }

    #[pyo3(signature = (waveform))]
    fn process(&mut self, waveform: Bound<'_, PyAny>) -> PyResult<()> {
        let waveform = extract_1d_f32(&waveform).map_err(|_| {
            PyTypeError::new_err("waveform must be a 1D numpy array of dtype float32 or float64")
        })?;

        let last_peaks_view = if self.last_peaks.is_empty() {
            None
        } else {
            Some(ArrayView1::from(&self.last_peaks))
        };

        let new_last_peaks = peak_peak_and_rainflow_counting_inplace_with_cycles(
            waveform.view(),
            last_peaks_view,
            self.bin_size,
            self.threshold,
            &mut self.cycles,
        );

        self.last_peaks = new_last_peaks;

        Ok(())
    }

    fn reset(&mut self) {
        self.cycles.clear();
        self.last_peaks.clear();
    }

    fn cycles_len(&self) -> usize {
        self.cycles.len()
    }

    fn get_last_peaks<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<WaveformSampleValueType>> {
        PyArray1::from_vec(py, self.last_peaks.clone())
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let py_cycles: Bound<'_, PyDict> = PyDict::new(py);
        for ((k1, k2), v) in &self.cycles {
            let key = (k1.into_inner(), k2.into_inner());
            py_cycles.set_item(key, *v)?;
        }
        Ok(py_cycles)
    }

    fn to_counter<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let dict = self.to_dict(py)?;
        let collections = py.import("collections")?;
        let counter_cls = collections.getattr("Counter")?;
        let counter = counter_cls.call1((dict,))?;
        Ok(counter)
    }

    /// Returns (heatmap, bins) where bins are the stress levels used as both axes.
    /// When bin_size > 0, bins are a full uniform grid from min..max in steps of bin_size
    /// (missing values are included as all-zero rows/cols).
    fn to_heatmap<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<WaveformSampleValueType>>,
    )> {
        if self.cycles.is_empty() {
            let heatmap = Array2::<f64>::zeros((0, 0));
            let py_heatmap = PyArray2::from_owned_array(py, heatmap);
            let py_bins = PyArray1::from_vec(py, Vec::<WaveformSampleValueType>::new());
            return Ok((py_heatmap, py_bins));
        }

        let mut unique: BTreeSet<OrderedFloat<WaveformSampleValueType>> = BTreeSet::new();
        for ((from, to), _) in &self.cycles {
            unique.insert(*from);
            unique.insert(*to);
        }

        let bins: Vec<WaveformSampleValueType> = if self.bin_size > 0.0 {
            let min_v = unique.first().unwrap().into_inner();
            let max_v = unique.last().unwrap().into_inner();

            let min_bin = quantize(min_v, self.bin_size) as f64;
            let max_bin = quantize(max_v, self.bin_size) as f64;
            let step = self.bin_size as f64;

            let mut out: Vec<WaveformSampleValueType> = Vec::new();
            let mut v = min_bin;
            let eps = step * 1e-6;
            while v <= max_bin + eps {
                out.push(v as WaveformSampleValueType);
                v += step;
            }
            out
        } else {
            unique.into_iter().map(|v| v.into_inner()).collect()
        };
        let n = bins.len();

        let mut idx: AHashMap<OrderedFloat<WaveformSampleValueType>, usize> = AHashMap::new();
        for (i, v) in bins.iter().enumerate() {
            idx.insert(OrderedFloat::from(*v), i);
        }

        let mut data: Vec<f64> = vec![0.0; n * n];
        for ((from, to), count) in &self.cycles {
            if let (Some(&i), Some(&j)) = (idx.get(from), idx.get(to)) {
                data[i * n + j] += *count as f64;
            }
        }

        let heatmap = Array2::<f64>::from_shape_vec((n, n), data)
            .map_err(|e| PyTypeError::new_err(format!("failed to build heatmap: {e}")))?;
        let py_heatmap = PyArray2::from_owned_array(py, heatmap);
        let py_bins = PyArray1::from_vec(py, bins);
        Ok((py_heatmap, py_bins))
    }

    #[pyo3(signature = (m, m2=None, include_half_cycles=false))]
    fn goodman_transform<'py>(
        &self,
        py: Python<'py>,
        m: WaveformSampleValueType,
        m2: Option<WaveformSampleValueType>,
        include_half_cycles: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let m2_value = m2.unwrap_or(m / 3.0);

        let mut rust_cycles: CyclesMapWithHalfCycles = CyclesMapWithHalfCycles::new();
        for ((from, to), count) in &self.cycles {
            rust_cycles.insert(
                (
                    OrderedFloat::from(from.into_inner()),
                    OrderedFloat::from(to.into_inner()),
                ),
                *count as WaveformSampleValueType,
            );
        }

        if include_half_cycles && self.last_peaks.len() >= 2 {
            for i in 0..(self.last_peaks.len() - 1) {
                let f = self.last_peaks[i];
                let t = self.last_peaks[i + 1];
                let difference = (t - f).abs();

                if difference < self.threshold {
                    continue;
                }

                let key = if self.bin_size > 0.0 {
                    let from: WaveformSampleValueType;
                    let to: WaveformSampleValueType;

                    if difference < (self.bin_size / 2.0) {
                        let abs_max: WaveformSampleValueType =
                            if f.abs() >= t.abs() { f } else { t };
                        from = abs_max;
                        to = abs_max;
                    } else {
                        from = f;
                        to = t;
                    }

                    (
                        OrderedFloat::from(quantize(from, self.bin_size)),
                        OrderedFloat::from(quantize(to, self.bin_size)),
                    )
                } else {
                    (OrderedFloat::from(f), OrderedFloat::from(t))
                };

                *rust_cycles.entry(key).or_insert(0.0) += 0.5;
            }
        }

        let transformed = goodman_transform_internal(&rust_cycles, m, m2_value);

        let py_result = PyDict::new(py);
        for (k, v) in transformed {
            py_result.set_item(k.into_inner(), v)?;
        }

        Ok(py_result)
    }

    #[pyo3(signature = (m, m2=None, include_half_cycles=false))]
    fn summed_histogram<'py>(
        &self,
        py: Python<'py>,
        m: WaveformSampleValueType,
        m2: Option<WaveformSampleValueType>,
        include_half_cycles: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let m2_value = m2.unwrap_or(m / 3.0);

        let mut rust_cycles: CyclesMapWithHalfCycles = CyclesMapWithHalfCycles::new();
        for ((from, to), count) in &self.cycles {
            rust_cycles.insert(
                (
                    OrderedFloat::from(from.into_inner()),
                    OrderedFloat::from(to.into_inner()),
                ),
                *count as WaveformSampleValueType,
            );
        }

        if include_half_cycles && self.last_peaks.len() >= 2 {
            for i in 0..(self.last_peaks.len() - 1) {
                let f = self.last_peaks[i];
                let t = self.last_peaks[i + 1];
                let difference = (t - f).abs();

                if difference < self.threshold {
                    continue;
                }

                let key = if self.bin_size > 0.0 {
                    let from: WaveformSampleValueType;
                    let to: WaveformSampleValueType;

                    if difference < (self.bin_size / 2.0) {
                        let abs_max: WaveformSampleValueType =
                            if f.abs() >= t.abs() { f } else { t };
                        from = abs_max;
                        to = abs_max;
                    } else {
                        from = f;
                        to = t;
                    }

                    (
                        OrderedFloat::from(quantize(from, self.bin_size)),
                        OrderedFloat::from(quantize(to, self.bin_size)),
                    )
                } else {
                    (OrderedFloat::from(f), OrderedFloat::from(t))
                };

                *rust_cycles.entry(key).or_insert(0.0) += 0.5;
            }
        }

        let transformed = goodman_transform_internal(&rust_cycles, m, m2_value);
        let summed = summed_histogram_internal(&transformed);

        let py_list = PyList::empty(py);
        for (s_a_ers, cumulative) in summed {
            let pair = (s_a_ers as f64, cumulative as f64);
            py_list.append(pair)?;
        }

        Ok(py_list.into_any())
    }
}

fn goodman_transform_internal(
    cycles: &CyclesMapWithHalfCycles,
    m: WaveformSampleValueType,
    m2: WaveformSampleValueType,
) -> AHashMap<OrderedFloat<WaveformSampleValueType>, WaveformSampleValueType> {
    let mut result: AHashMap<OrderedFloat<WaveformSampleValueType>, WaveformSampleValueType> =
        AHashMap::new();

    let one_minus_m = 1.0 - m;
    let one_plus_m = 1.0 + m;
    let one_plus_m2 = 1.0 + m2;
    let factor_low = one_plus_m / one_plus_m2;
    let factor_high = (one_plus_m * one_plus_m) / one_plus_m2;

    for ((from, to), count) in cycles.iter() {
        let from = from.into_inner();
        let to = to.into_inner();

        let s_upper = from.max(to);
        let s_lower = from.min(to);

        let s_a = (s_upper - s_lower) / 2.0;
        let s_m = (s_upper + s_lower) / 2.0;

        let s_a_ers = if s_upper == 0.0 {
            // R -> +inf, treat as Druckschwellbereich (R >= 1)
            one_minus_m * s_a
        } else {
            let r = s_lower / s_upper;

            if r >= 1.0 {
                one_minus_m * s_a
            } else if r <= 0.0 {
                s_a + m * s_m
            } else if r <= 0.5 {
                factor_low * (s_a + m2 * s_m)
            } else {
                factor_high * s_a
            }
        };

        let key = OrderedFloat::from(s_a_ers);
        *result.entry(key).or_insert(0.0) += *count;
    }

    result
}

#[pyfunction]
#[pyo3(signature = (waveform, last_peaks=None, bin_size=0.0, threshold=None, min_chunk_size=64*1024))]
fn rainflow<'py>(
    py: Python<'py>,
    waveform: Bound<'py, PyAny>,
    last_peaks: Option<Bound<'py, PyAny>>,
    bin_size: Option<WaveformSampleValueType>,
    threshold: Option<WaveformSampleValueType>,
    min_chunk_size: Option<usize>,
) -> PyResult<(
    Bound<'py, PyDict>,
    Bound<'py, PyArray1<WaveformSampleValueType>>,
)> {
    let span = span!(Level::TRACE, "rainflow");
    let _enter = span.enter();

    let bin_size = bin_size.unwrap_or(0.0);
    let threshold = threshold.unwrap_or(0.0);
    // Accept np.float32 or np.float64 and convert to f32 if needed
    let waveform: Array1<WaveformSampleValueType> = extract_1d_f32(&waveform).map_err(|_| {
        PyTypeError::new_err("waveform must be a 1D numpy array of dtype float32 or float64")
    })?;

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
        Some(extract_1d_f32(&lp_any).map_err(|_| {
            PyTypeError::new_err("last_peaks must be a 1D numpy array of dtype float32 or float64")
        })?)
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
                peak_peak_and_rainflow_counting(chunk.view(), last_peaks, bin_size, threshold)
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
        peak_peak_and_rainflow_counting(chunks[0].view(), last_peaks_vec[0], bin_size, threshold)
    };

    let (final_cycles, final_peaks) = if chunks.len() > 1 {
        peak_peak_and_rainflow_counting(ArrayView1::from(&all_peaks), None, bin_size, threshold)
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

#[pyfunction]
#[pyo3(signature = (cycles, m, m2=None))]
fn goodman_transform<'py>(
    py: Python<'py>,
    cycles: Bound<'py, PyAny>,
    m: WaveformSampleValueType,
    m2: Option<WaveformSampleValueType>,
) -> PyResult<Bound<'py, PyDict>> {
    let m2_value = m2.unwrap_or(m / 3.0);
    let dict = cycles.downcast_into::<PyDict>()?;

    let mut rust_cycles: CyclesMapWithHalfCycles = CyclesMapWithHalfCycles::new();

    for (key, value) in dict.iter() {
        let (s_lower, s_upper): (WaveformSampleValueType, WaveformSampleValueType) =
            key.extract()?;
        let count: WaveformSampleValueType =
            if let Ok(c) = value.extract::<WaveformSampleValueType>() {
                c
            } else if let Ok(c) = value.extract::<f64>() {
                c as WaveformSampleValueType
            } else if let Ok(c) = value.extract::<usize>() {
                c as WaveformSampleValueType
            } else {
                return Err(PyTypeError::new_err(
                    "cycle counts must be int or float (e.g. 1 or 0.5)",
                ));
            };

        rust_cycles.insert(
            (OrderedFloat::from(s_lower), OrderedFloat::from(s_upper)),
            count,
        );
    }

    let transformed = goodman_transform_internal(&rust_cycles, m, m2_value);

    let py_result = PyDict::new(py);
    for (k, v) in transformed {
        py_result.set_item(k.into_inner(), v)?;
    }

    Ok(py_result)
}

fn summed_histogram_internal(
    goodman_result: &AHashMap<OrderedFloat<WaveformSampleValueType>, WaveformSampleValueType>,
) -> Vec<(WaveformSampleValueType, WaveformSampleValueType)> {
    let mut entries: Vec<(WaveformSampleValueType, WaveformSampleValueType)> = goodman_result
        .iter()
        .map(|(k, v)| (k.into_inner(), *v))
        .collect();

    entries.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());

    let mut cumulative = 0.0_f32;
    let mut result: Vec<(WaveformSampleValueType, WaveformSampleValueType)> =
        Vec::with_capacity(entries.len());

    for (s_a_ers, count) in entries {
        cumulative += count;
        result.push((s_a_ers, cumulative));
    }

    result
}

#[pyfunction]
fn summed_histogram<'py>(py: Python<'py>, hist: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let dict = hist.downcast_into::<PyDict>()?;

    let mut goodman_result: AHashMap<
        OrderedFloat<WaveformSampleValueType>,
        WaveformSampleValueType,
    > = AHashMap::new();

    for (key, value) in dict.iter() {
        let s_a_ers: WaveformSampleValueType = key.extract()?;
        let count: WaveformSampleValueType = value.extract()?;
        goodman_result.insert(OrderedFloat::from(s_a_ers), count);
    }

    let summed = summed_histogram_internal(&goodman_result);
    let py_list = PyList::empty(py);
    for (s_a_ers, cumulative) in summed {
        let pair = (s_a_ers as f64, cumulative as f64);
        py_list.append(pair)?;
    }

    Ok(py_list.into_any())
}

/// A Python module implemented in Rust.
#[pymodule]
fn typhoon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(rainflow, m)?)?;
    m.add_function(wrap_pyfunction!(goodman_transform, m)?)?;
    m.add_function(wrap_pyfunction!(summed_histogram, m)?)?;
    m.add_class::<RainflowContext>()?;

    Ok(())
}
