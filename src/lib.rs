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

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MinerDamageMode {
    Original,
    Modified,
    KonsequentMiner,
    ElementarMiner,
}

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

impl RainflowContext {
    fn for_each_residual_half_cycle(&self, mut f: impl FnMut(CyclesKey)) {
        if self.last_peaks.len() < 2 {
            return;
        }

        for i in 0..(self.last_peaks.len() - 1) {
            let from_peak = self.last_peaks[i];
            let to_peak = self.last_peaks[i + 1];
            let difference = (to_peak - from_peak).abs();

            if difference < self.threshold {
                continue;
            }

            let key: CyclesKey = if self.bin_size > 0.0 {
                let from: WaveformSampleValueType;
                let to: WaveformSampleValueType;

                if difference < (self.bin_size / 2.0) {
                    let abs_max: WaveformSampleValueType = if from_peak.abs() >= to_peak.abs() {
                        from_peak
                    } else {
                        to_peak
                    };
                    from = abs_max;
                    to = abs_max;
                } else {
                    from = from_peak;
                    to = to_peak;
                }

                (
                    OrderedFloat::from(quantize(from, self.bin_size)),
                    OrderedFloat::from(quantize(to, self.bin_size)),
                )
            } else {
                (OrderedFloat::from(from_peak), OrderedFloat::from(to_peak))
            };

            f(key);
        }
    }

    fn cycles_with_optional_half_cycles(
        &self,
        include_half_cycles: bool,
    ) -> CyclesMapWithHalfCycles {
        let mut rust_cycles: CyclesMapWithHalfCycles = CyclesMapWithHalfCycles::new();

        for (key, count) in &self.cycles {
            rust_cycles.insert(*key, *count as WaveformSampleValueType);
        }

        if include_half_cycles {
            self.for_each_residual_half_cycle(|key| {
                *rust_cycles.entry(key).or_insert(0.0) += 0.5;
            });
        }

        rust_cycles
    }

    fn goodman_transform_map(
        &self,
        m: WaveformSampleValueType,
        m2: Option<WaveformSampleValueType>,
        include_half_cycles: bool,
        bin_size_compensation: WaveformSampleValueType,
    ) -> AHashMap<OrderedFloat<WaveformSampleValueType>, WaveformSampleValueType> {
        let m2_value = m2.unwrap_or(m / 3.0);
        let rust_cycles = self.cycles_with_optional_half_cycles(include_half_cycles);
        goodman_transform_internal(&rust_cycles, m, m2_value, bin_size_compensation)
    }
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
    #[pyo3(signature = (include_half_cycles=false))]
    fn to_heatmap<'py>(
        &self,
        py: Python<'py>,
        include_half_cycles: bool,
    ) -> PyResult<(
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray1<WaveformSampleValueType>>,
    )> {
        // Build a float-valued cycles map so we can optionally add half cycles (0.5).
        let mut cycles: AHashMap<CyclesKey, f64> = AHashMap::new();
        for (k, v) in &self.cycles {
            *cycles.entry(*k).or_insert(0.0) += *v as f64;
        }

        if include_half_cycles {
            self.for_each_residual_half_cycle(|key| {
                *cycles.entry(key).or_insert(0.0) += 0.5;
            });
        }

        if cycles.is_empty() {
            let heatmap = Array2::<f64>::zeros((0, 0));
            let py_heatmap = PyArray2::from_owned_array(py, heatmap);
            let py_bins = PyArray1::from_vec(py, Vec::<WaveformSampleValueType>::new());
            return Ok((py_heatmap, py_bins));
        }

        let mut unique: BTreeSet<OrderedFloat<WaveformSampleValueType>> = BTreeSet::new();
        for ((from, to), _) in &cycles {
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
        for ((from, to), count) in &cycles {
            if let (Some(&i), Some(&j)) = (idx.get(from), idx.get(to)) {
                data[i * n + j] += *count;
            }
        }

        let heatmap = Array2::<f64>::from_shape_vec((n, n), data)
            .map_err(|e| PyTypeError::new_err(format!("failed to build heatmap: {e}")))?;
        let py_heatmap = PyArray2::from_owned_array(py, heatmap);
        let py_bins = PyArray1::from_vec(py, bins);
        Ok((py_heatmap, py_bins))
    }

    #[pyo3(signature = (m, m2=None, include_half_cycles=false, bin_size_compensation=0.0))]
    fn goodman_transform<'py>(
        &self,
        py: Python<'py>,
        m: WaveformSampleValueType,
        m2: Option<WaveformSampleValueType>,
        include_half_cycles: bool,
        bin_size_compensation: WaveformSampleValueType,
    ) -> PyResult<Bound<'py, PyDict>> {
        validate_bin_size_compensation(bin_size_compensation)?;
        let transformed =
            self.goodman_transform_map(m, m2, include_half_cycles, bin_size_compensation);

        let py_result = PyDict::new(py);
        for (k, v) in transformed {
            py_result.set_item(k.into_inner(), v)?;
        }

        Ok(py_result)
    }

    #[pyo3(signature = (m, m2=None, include_half_cycles=false, bin_size_compensation=0.0))]
    fn summed_histogram<'py>(
        &self,
        py: Python<'py>,
        m: WaveformSampleValueType,
        m2: Option<WaveformSampleValueType>,
        include_half_cycles: bool,
        bin_size_compensation: WaveformSampleValueType,
    ) -> PyResult<Bound<'py, PyAny>> {
        validate_bin_size_compensation(bin_size_compensation)?;
        let transformed =
            self.goodman_transform_map(m, m2, include_half_cycles, bin_size_compensation);
        let summed = summed_histogram_internal(&transformed);

        let py_list = PyList::empty(py);
        for (s_a_ers, cumulative) in summed {
            let pair = (s_a_ers as f64, cumulative as f64);
            py_list.append(pair)?;
        }

        Ok(py_list.into_any())
    }

    #[pyo3(signature = (m, n_d, sigma_d, k, m2=None, include_half_cycles=false, bin_size_compensation=0.0, q=None, mode=MinerDamageMode::Modified))]
    fn fkm_miner_damage(
        &self,
        m: WaveformSampleValueType,
        n_d: f64,
        sigma_d: f64,
        k: f64,
        m2: Option<WaveformSampleValueType>,
        include_half_cycles: bool,
        bin_size_compensation: WaveformSampleValueType,
        q: Option<f64>,
        mode: MinerDamageMode,
    ) -> PyResult<f64> {
        validate_bin_size_compensation(bin_size_compensation)?;
        let transformed =
            self.goodman_transform_map(m, m2, include_half_cycles, bin_size_compensation);
        fkm_miner_damage_from_goodman_internal(&transformed, n_d, sigma_d, k, q, mode)
    }
}

fn goodman_transform_internal(
    cycles: &CyclesMapWithHalfCycles,
    m: WaveformSampleValueType,
    m2: WaveformSampleValueType,
    bin_size_compensation: WaveformSampleValueType,
) -> AHashMap<OrderedFloat<WaveformSampleValueType>, WaveformSampleValueType> {
    let mut result: AHashMap<OrderedFloat<WaveformSampleValueType>, WaveformSampleValueType> =
        AHashMap::new();

    let bin_size_compensation = if bin_size_compensation.is_finite() && bin_size_compensation > 0.0
    {
        bin_size_compensation
    } else {
        0.0
    };
    let half_bin_size_compensation = bin_size_compensation * 0.5;

    let one_minus_m = 1.0 - m;
    let one_plus_m = 1.0 + m;
    let one_plus_m2 = 1.0 + m2;
    let factor_low = one_plus_m / one_plus_m2;
    let factor_high = (one_plus_m * one_plus_m) / one_plus_m2;

    for ((from, to), count) in cycles.iter() {
        let from = from.into_inner();
        let to = to.into_inner();

        let mut s_upper = from.max(to);
        let mut s_lower = from.min(to);

        // Worst-case compensation for quantization to bin centers: widen the cycle.
        if half_bin_size_compensation != 0.0 {
            s_upper += half_bin_size_compensation;
            s_lower -= half_bin_size_compensation;
        }

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

fn validate_bin_size_compensation(bin_size_compensation: WaveformSampleValueType) -> PyResult<()> {
    if !bin_size_compensation.is_finite() || bin_size_compensation < 0.0 {
        return Err(PyTypeError::new_err(
            "bin_size_compensation must be a finite number >= 0",
        ));
    }
    Ok(())
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
#[pyo3(signature = (cycles, m, m2=None, bin_size_compensation=0.0))]
fn goodman_transform<'py>(
    py: Python<'py>,
    cycles: Bound<'py, PyAny>,
    m: WaveformSampleValueType,
    m2: Option<WaveformSampleValueType>,
    bin_size_compensation: WaveformSampleValueType,
) -> PyResult<Bound<'py, PyDict>> {
    let m2_value = m2.unwrap_or(m / 3.0);
    validate_bin_size_compensation(bin_size_compensation)?;
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

    let transformed = goodman_transform_internal(&rust_cycles, m, m2_value, bin_size_compensation);

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

fn fkm_miner_damage_from_goodman_internal(
    goodman_result: &AHashMap<OrderedFloat<WaveformSampleValueType>, WaveformSampleValueType>,
    n_d: f64,
    sigma_d: f64,
    k: f64,
    q: Option<f64>,
    mode: MinerDamageMode,
) -> PyResult<f64> {
    if mode == MinerDamageMode::KonsequentMiner {
        return fkm_konsequent_miner_damage_from_goodman_internal(
            goodman_result,
            n_d,
            sigma_d,
            k,
            q,
        );
    }

    let (n_d, sigma_d, k, k_plus_q) = fkm_modified_miner_params(n_d, sigma_d, k, q)?;

    let mut damage = 0.0_f64;
    for (sigma_a, count) in goodman_result.iter() {
        let sigma_a = sigma_a.into_inner() as f64;
        let count = *count as f64;
        fkm_modified_miner_accumulate(
            &mut damage,
            sigma_a,
            count,
            n_d,
            sigma_d,
            k,
            k_plus_q,
            mode,
        )?;
    }

    Ok(damage)
}

fn fkm_miner_damage_from_goodman_pydict(
    dict: &Bound<'_, PyDict>,
    n_d: f64,
    sigma_d: f64,
    k: f64,
    q: Option<f64>,
    mode: MinerDamageMode,
) -> PyResult<f64> {
    let goodman_result = fkm_parse_goodman_pydict(dict)?;
    fkm_miner_damage_from_goodman_internal(&goodman_result, n_d, sigma_d, k, q, mode)
}

fn fkm_elementary_damage_increment(
    sigma_a: f64,
    count: f64,
    n_d: f64,
    sigma_d: f64,
    k: f64,
) -> PyResult<f64> {
    fkm_validate_goodman_inputs(sigma_a, count)?;
    Ok(fkm_damage_increment(sigma_a, count, n_d, sigma_d, k))
}

fn fkm_parse_goodman_pydict(
    dict: &Bound<'_, PyDict>,
) -> PyResult<AHashMap<OrderedFloat<WaveformSampleValueType>, WaveformSampleValueType>> {
    let mut goodman_result: AHashMap<
        OrderedFloat<WaveformSampleValueType>,
        WaveformSampleValueType,
    > = AHashMap::new();

    for (key, value) in dict.iter() {
        let sigma_a: WaveformSampleValueType = key.extract()?;
        let count: WaveformSampleValueType = if let Ok(c) = value.extract::<f64>() {
            c as WaveformSampleValueType
        } else if let Ok(c) = value.extract::<WaveformSampleValueType>() {
            c
        } else if let Ok(c) = value.extract::<usize>() {
            c as WaveformSampleValueType
        } else {
            return Err(PyTypeError::new_err(
                "goodman cycle counts must be int or float",
            ));
        };

        goodman_result.insert(OrderedFloat::from(sigma_a), count);
    }

    Ok(goodman_result)
}

fn fkm_konsequent_miner_damage_from_goodman_internal(
    goodman_result: &AHashMap<OrderedFloat<WaveformSampleValueType>, WaveformSampleValueType>,
    n_d: f64,
    sigma_d: f64,
    k: f64,
    q: Option<f64>,
) -> PyResult<f64> {
    let (n_d, sigma_d, k, _) = fkm_modified_miner_params(n_d, sigma_d, k, q)?;

    let q_value = q.unwrap_or(k - 1.0);
    if !q_value.is_finite() || q_value <= 0.0 {
        return Err(PyTypeError::new_err(
            "q must be a positive finite number for KonsequentMiner",
        ));
    }

    let mut entries: Vec<(f64, f64)> = Vec::with_capacity(goodman_result.len() + 1);
    let mut has_zero = false;

    for (sigma_a, count) in goodman_result.iter() {
        let sigma_a = sigma_a.into_inner() as f64;
        let count = *count as f64;

        if sigma_a == 0.0 {
            has_zero = true;
        }

        entries.push((sigma_a, count));
    }

    if !has_zero {
        entries.push((0.0, 0.0));
    }

    entries.sort_by(|(a, _), (b, _)| b.partial_cmp(a).unwrap());

    let mut d_sum = 0.0_f64;
    let mut d_min_prev = 0.0_f64;
    let mut w = 0.0_f64;

    for (sigma_a, count) in entries {
        let d_sum_prev = d_sum;
        d_sum += fkm_elementary_damage_increment(sigma_a, count, n_d, sigma_d, k)?;
        if d_sum >= 1.0 {
            break;
        }

        if sigma_a < sigma_d {
            let ratio = if sigma_a == 0.0 {
                0.0
            } else {
                sigma_a / sigma_d
            };
            let d_min = 1.0 - ratio.powf(q_value);
            let mut delta = d_min - d_min_prev;
            if delta < 0.0 {
                delta = 0.0;
            }

            if d_sum_prev <= 0.0 {
                return Ok(0.0);
            }

            w += delta / d_sum_prev;
            d_min_prev = d_min;
        }
    }

    if w <= 0.0 || !w.is_finite() {
        return Ok(1.0)
    }

    Ok(1.0 / w)
}

fn fkm_modified_miner_params(
    n_d: f64,
    sigma_d: f64,
    k: f64,
    q: Option<f64>,
) -> PyResult<(f64, f64, f64, f64)> {
    if !n_d.is_finite() || n_d <= 0.0 {
        return Err(PyTypeError::new_err("N_D must be a positive finite number"));
    }
    if !sigma_d.is_finite() || sigma_d <= 0.0 {
        return Err(PyTypeError::new_err(
            "sigma_D must be a positive finite number",
        ));
    }
    if !k.is_finite() || k <= 0.0 {
        return Err(PyTypeError::new_err("k must be a positive finite number"));
    }

    let q = q.unwrap_or(k - 1.0);
    if !q.is_finite() {
        return Err(PyTypeError::new_err("q must be a finite number"));
    }
    let k_plus_q = k + q;
    if !k_plus_q.is_finite() || k_plus_q <= 0.0 {
        return Err(PyTypeError::new_err("k + q must be positive"));
    }

    Ok((n_d, sigma_d, k, k_plus_q))
}

fn fkm_modified_miner_accumulate(
    damage: &mut f64,
    sigma_a: f64,
    count: f64,
    n_d: f64,
    sigma_d: f64,
    k: f64,
    k_plus_q: f64,
    mode: MinerDamageMode,
) -> PyResult<()> {
    fkm_validate_goodman_inputs(sigma_a, count)?;

    if mode == MinerDamageMode::Original && sigma_a < sigma_d {
        return Ok(());
    }

    let exp = if mode == MinerDamageMode::ElementarMiner || sigma_a >= sigma_d {
        k
    } else {
        k_plus_q
    };
    *damage += fkm_damage_increment(sigma_a, count, n_d, sigma_d, exp);
    Ok(())
}

fn fkm_validate_goodman_inputs(sigma_a: f64, count: f64) -> PyResult<()> {
    if !sigma_a.is_finite() || sigma_a < 0.0 {
        return Err(PyTypeError::new_err(
            "goodman amplitudes must be finite and >= 0",
        ));
    }
    if !count.is_finite() || count < 0.0 {
        return Err(PyTypeError::new_err(
            "goodman cycle counts must be finite and >= 0",
        ));
    }
    Ok(())
}

fn fkm_damage_increment(sigma_a: f64, count: f64, n_d: f64, sigma_d: f64, exp: f64) -> f64 {
    if sigma_a == 0.0 || count == 0.0 {
        return 0.0;
    }

    let ratio = sigma_a / sigma_d;
    (count / n_d) * ratio.powf(exp)
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

#[pyfunction]
#[pyo3(signature = (goodman_result, n_d, sigma_d, k, q=None, mode=MinerDamageMode::Modified))]
fn fkm_miner_damage<'py>(
    _py: Python<'py>,
    goodman_result: Bound<'py, PyAny>,
    n_d: f64,
    sigma_d: f64,
    k: f64,
    q: Option<f64>,
    mode: MinerDamageMode,
) -> PyResult<f64> {
    let dict = goodman_result.downcast_into::<PyDict>()?;
    fkm_miner_damage_from_goodman_pydict(&dict, n_d, sigma_d, k, q, mode)
}

/// A Python module implemented in Rust.
#[pymodule]
fn typhoon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(init_tracing, m)?)?;
    m.add_function(wrap_pyfunction!(rainflow, m)?)?;
    m.add_function(wrap_pyfunction!(goodman_transform, m)?)?;
    m.add_function(wrap_pyfunction!(summed_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(fkm_miner_damage, m)?)?;
    m.add_class::<MinerDamageMode>()?;
    m.add_class::<RainflowContext>()?;

    Ok(())
}
