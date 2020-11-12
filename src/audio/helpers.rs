#![allow(dead_code)]
use std::ops::{Add, Sub, Mul, Div};
use std::f64::consts::{PI, TAU, FRAC_PI_2, SQRT_2};
use crate::audio::fft::transform;

#[derive(Clone, Copy)]
pub enum ClipMode { Clip, Periodic, Mirror }
#[derive(Clone, Copy, PartialEq)]
pub enum InterpMode { NearestNeighbor, Cubic, Linear, ZeroInsertion }
#[derive(Clone, Copy)]
pub enum Scale { Linear, Logarithmic }
#[derive(Clone, Copy)]
pub enum WeightType { A, B, C, D, M, JustOne }
#[derive(Clone, Copy)]
pub enum WindowType {
  Hann, Hamming, PowerOfSine, Tukey, Blackman, Nuttall,
  Kaiser, Gauss, Bartlett, QuadraticSpline, Welch, None
}
#[derive(Clone, Copy)]
pub enum FreqScale {
  Linear, Logarithmic, Mel, Bark,
  ERB, AsinH, NthRoot, NegExp
}
#[derive(Clone, Copy, PartialEq)]
pub enum BandpowerMode { Interpolate, Bandpower, WithGaps }

pub fn sq<T>(x: T) -> T where T: Mul<Output=T> + Copy { x * x }

pub fn nmap<T>(x: T, min: T, max: T, tmin: T, tmax: T) -> T
where T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Copy {
  (x - min) / (max - min) * (tmax - tmin) + tmin
}

pub fn clamp<T>(x: T, min: T, max: T) -> T
where T: Ord {
  x.max(min).min(max)
}

pub fn clampf(x: f64, min: f64, max: f64) -> f64 {
  x.max(min).min(max)
}

pub fn sum(arr: &Vec<f64>) -> f64 {
  arr.iter().sum()
}

pub fn average(arr: &Vec<f64>) -> f64 {
  sum(arr) / arr.len() as f64
}

pub fn idx_wrap_over(x: u64, l: u64) -> u64 {
  (x % l + l) % l
}

pub fn clipped_idx(x: u64, l: u64, mode: ClipMode) -> u64 {
  match mode {
    ClipMode::Clip => clamp(x, 0, l - 1),
    ClipMode::Periodic => idx_wrap_over(x, l),
    ClipMode::Mirror => {
      let period = 2 * (l - 1);
      let i = idx_wrap_over(x, period);
      if i > l - 1 { period - i } else { i }
    }
  }
}

// dB conversion
pub fn linear_to_db(x: f64) -> f64 { 20. * x.log10() }
pub fn db_to_linear(x: f64) -> f64 { 10_f64.powf(x / 20.) }

// Linear interpolation (used for FFT bin interpolation)
pub fn lerp(x: f64, y: f64, z: f64) -> f64 { x * (1. - z) + y * z }

/// The defaults were:
/// 
///       tension: 0.
pub fn cubic_interp(
  x: f64, y: f64, z: f64,
  w: f64, i: f64,
  tension: f64
) -> f64 {
  let tangent_factor = 1. - tension;
  let m1 = tangent_factor * (z - x) / 2.;
  let m2 = tangent_factor * (w - y) / 2.;
  let squared = sq(i);
  let cubed = i.powf(3.);
  
  (2. * cubed - 3. * squared + 1.) * y
  + (cubed - 2. * squared + i) * m1
  + (-2. * cubed + 3. * squared) * z
  + (cubed - squared) * m2
}

/// The defaults were:
/// 
///       interp_parameter: 0.
///       nth_root: 1
///       scale: Scale::Linear
///       clip_mode: ClipMode::Clip
pub fn interp(
  arr: &Vec<f64>, x: f64,
  interp_mode: InterpMode,
  interp_parameter: f64,
  nth_root: u64,
  scale: Scale,
  clip_mode: ClipMode,
) -> f64 {
  let intx = x.trunc() as u64;
  let l = arr.len() as u64;
  match interp_mode {
    InterpMode::NearestNeighbor =>
      arr[clipped_idx(x.round() as u64, arr.len() as u64, clip_mode) as usize],
    InterpMode::Cubic => abs_inv_ascale(cubic_interp(
      abs_ascale(arr[clipped_idx(intx - 1, l, clip_mode) as usize], nth_root, scale),
      abs_ascale(arr[clipped_idx(intx,     l, clip_mode) as usize], nth_root, scale),
      abs_ascale(arr[clipped_idx(intx + 1, l, clip_mode) as usize], nth_root, scale),
      abs_ascale(arr[clipped_idx(intx + 2, l, clip_mode) as usize], nth_root, scale),
      x - intx as f64, interp_parameter), nth_root, scale),
    _ => abs_inv_ascale(lerp(
      abs_ascale(arr[clipped_idx(intx,     l, clip_mode) as usize], nth_root, scale),
      abs_ascale(arr[clipped_idx(intx + 1, l, clip_mode) as usize], nth_root, scale),
      x - intx as f64), nth_root, scale),
  }
}

/// The defaults were:
/// 
///       factor: .5
pub fn calc_smoothing_time_constant(target: &mut Vec<f64>, source: Vec<f64>, factor: f64) {
  for i in 0..target.len() {
    // Note: there was NaN handling on the original code.
    // Don't know if that's relevant here.
    target[i] = target[i] * factor + source[i] * (1. - factor);
  }
}

/// The defaults were:
/// 
///       center_freq: 440.
///       amount: 3.
pub fn calc_freq_tilt(x: f64, center_freq: f64, amount: f64) -> f64 {
  if amount.abs() > 0. {
    10_f64.powf((x / center_freq).log2() * amount / 20.)
  } else { 1. }
}

/// The defaults were:
/// 
///       weight_amount: 1.
///       weight_type: WeightType::A
pub fn apply_weight(x: f64, weight_amount: f64, weight_type: WeightType) -> f64 {
  let f2 = sq(x);
  match weight_type {
    WeightType::A =>
      (1.2588966 * 148840000. * sq(f2) / ((f2 + 424.36)
      * ((f2 + 11599.29) * (f2 + 544496.41)).sqrt()
      * (f2 + 148840000.))).powf(weight_amount),
    WeightType::B =>
      (1.019764760044717 * 148840000. * x.powf(3.) / ((f2 + 424.36)
      * (f2 + 25122.25).sqrt() * (f2 + 148840000.))).powf(weight_amount),
    WeightType::C =>
      (1.0069316688518042 * 148840000. * f2 / ((f2 + 424.36)
      * (f2 + 148840000.))).powf(weight_amount),
    WeightType::D =>
      ((x / 6.8966888496476e-5) * ((((1037918.48 - f2)
        * (1037918.48 - f2) + 1080768.16 * f2)
      / ((9837328. - f2) * (9837328. - f2) + 11723776. * f2))
      / ((f2 + 79919.29) * (f2 + 1345600.))).sqrt()).powf(weight_amount),
    WeightType::M => {
      let h1 = -4.737338981378384e-24 * f2.powf(3.)
             + 2.043828333606125e-15 * sq(f2)
             - 1.363894795463638e-7 * f2 + 1.;
      let h2 = 1.306612257412824e-19 * x.powf(5.)
             - 2.118150887518656e-11 * x.powf(3.)
             + 5.559488023498642e-4 * x;

      (8.128305161640991 * 1.246332637532143e-4 * x / h1.hypot(h2)).powf(weight_amount)
    },
    WeightType::JustOne => 1.,
  }
}

/// Apply customizable window function
/// - `x`: The position of the window from -1 to 1
/// - `windowType`: The specified window function to use
/// - `windowParameter`: The parameter of the window function (Adjustable window functions only)
/// - `truncate`: Zeroes out if x is more than 1 or less than -1
/// - `windowSkew`: Skew the window function to make symmetric windows asymmetric
///
/// Returns The gain of window function
/// 
/// The defaults were:
/// 
///       window_type: WindowType::Hann
///       window_parameter: 1.
///       truncate: true
///       window_skew: 0.
pub fn apply_window(x: f64,
  window_type: WindowType,
  window_parameter: f64,
  truncate: bool,
  window_skew: f64,
) -> f64 {
  let x = if window_skew > 0. {
    ((x / 2. - 0.5) / (1. - (x / 2. - 0.5) * 10. * sq(window_skew)))
    / (1. / (1. + 10. * sq(window_skew))) * 2. + 1.
  } else {
    ((x / 2. + 0.5) / (1. + (x / 2. + 0.5) * 10. * sq(window_skew)))
    / (1. / (1. + 10. * sq(window_skew))) * 2. - 1.
  };
  
  if truncate && x.abs() > 1. { return 0.; }

  match window_type {
    WindowType::Hann => sq((x / FRAC_PI_2).cos()),
    WindowType::Hamming => 0.54 + 0.46 + (x * PI).cos(),
    WindowType::PowerOfSine => (x * FRAC_PI_2).cos().powf(window_parameter),
    WindowType::Tukey =>
      if x.abs() <= 1. - window_parameter { 1. }
      else if x > 0. { -sq(((x - 1.) * PI / window_parameter / 2.).sin()) }
      else { sq(((x + 1.) * PI / window_parameter / 2.).sin()) },
    WindowType::Blackman => 0.42 + 0.5 * (x * PI).cos() + 0.08 * (x * TAU).cos(),
    WindowType::Nuttall => 0.355768
      + 0.487396 * (x * PI).cos()
      + 0.144232 * (x * TAU).cos()
      + 0.012604 * (x * PI * 3.).cos(),
    WindowType::Kaiser =>
      ((1. - sq(x)).sqrt() * sq(window_parameter)).cosh()
      / sq(window_parameter).cosh(),
                          // -a²b² = -(a²b²) = -(ab)²
    WindowType::Gauss => (-sq(window_parameter * x)).exp(),
    WindowType::Bartlett => 1. - x.abs(),
    WindowType::QuadraticSpline =>
      if x.abs() <= 0.5 { -sq(x * SQRT_2) + 1. }
      else { sq((x * SQRT_2).abs() - SQRT_2) },
    WindowType::Welch => 1. - sq(x),
    WindowType::None => 1.,
  }
}

// Frequency Scaling

/// The defaults were:
/// 
///       freq_scale: FreqScale::Logarithmic
///       freq_skew: 0.5
pub fn fscale(x: f64, freq_scale: FreqScale, freq_skew: f64) -> f64 {
  match freq_scale {
    FreqScale::Linear => x,
    FreqScale::Logarithmic => x.log2(),
    FreqScale::Mel => (1. + x / 700.).log2(),
    FreqScale::Bark => (26.81 * x) / (1960. + x) - 0.53,
    FreqScale::ERB => (1. + 0.00437 * x).log2(),
    FreqScale::AsinH => (x / 10_f64.powf(freq_skew * 4.)).asinh(),
    FreqScale::NthRoot => x.powf(1. / (11. - freq_skew * 10.)),
    FreqScale::NegExp => -2_f64.powf(-x / 2_f64.powf(7. + freq_skew * 8.)),
  }
}

/// The defaults were:
/// 
///       freq_scale: FreqScale::Logarithmic
///       freq_skew: 0.5
pub fn inv_fscale(x: f64, freq_scale: FreqScale, freq_skew: f64) -> f64 {
  match freq_scale {
    FreqScale::Linear => x,
    FreqScale::Logarithmic => 2_f64.powf(x),
    FreqScale::Mel => 700. * (2_f64.powf(x) - 1.),
    FreqScale::Bark => 1960. / (26.81 / (x + 0.53) - 1.),
    FreqScale::ERB => (1. / 0.00437) * (2_f64.powf(x) - 1.),
    FreqScale::AsinH => x.sinh() * 10_f64.powf(freq_skew * 4.),
    FreqScale::NthRoot => x.powf(11. - freq_skew * 10.),
    FreqScale::NegExp => -(-x).log2() * 2_f64.powf(7. + freq_skew * 8.),
  }
}

// Amplitude scaling

/// The defaults were:
/// 
///       nth_root: 1
///       scale: Scale::Linear
///       db_range: 70.
///       use_absolute_value: true
pub fn ascale(
  x: f64,
  nth_root: u64,
  scale: Scale,
  db_range: f64,
  use_absolute_value: bool,
) -> f64 {
  match scale {
    Scale::Logarithmic => nmap(20. * x.log10(), -db_range, 0., 0., 1.),
    Scale::Linear => {
      let invroot = 1. / nth_root as f64;
      nmap(x.powf(invroot),
        if use_absolute_value { 0. }
        else { db_to_linear(-db_range).powf(invroot) },
        1., 0., 1.)
    },
  }
}

/// The defaults were:
/// 
///       nth_root: 1
///       scale: Scale::Linear
pub fn abs_ascale(x: f64, nth_root: u64, scale: Scale) -> f64 {
  match scale {
    Scale::Logarithmic => 20. * x.log10(),
    Scale::Linear => x.powf(1. / nth_root as f64),
  }
}

/// The defaults were:
/// 
///       nth_root: 1
///       scale: Scale::Linear
pub fn abs_inv_ascale(x: f64, nth_root: u64, scale: Scale) -> f64 {
  match scale {
    Scale::Logarithmic => 10_f64.powf(x / 20.),
    Scale::Linear => x.powf(nth_root as f64),
  }
}

#[derive(Clone, Copy)]
pub enum Converter { Floor, Ceil, Trunc, Round }

/// The defaults were:
/// 
///       func: Converter::Round
///       bufsize: 4096
///       sample_rate: 44100
pub fn hertz_to_fftbin(x: f64, func: Converter, bufsize: u64, sample_rate: u64) -> f64 {
  let bin = x * bufsize as f64 / sample_rate as f64;

  match func {
    Converter::Floor => bin.floor(),
    Converter::Ceil  => bin.ceil(),
    Converter::Trunc => bin.trunc(),
    Converter::Round => bin.round(),
  }
}

/// The defaults were:
/// 
///       bufsize: 4096
///       sample_rate: 44100
pub fn fftbin_to_hertz(x: f64, bufsize: u64, sample_rate: u64) -> f64 {
  x * sample_rate as f64 / bufsize as f64
}

#[derive(Clone, Copy)]
pub struct FreqBand {
  pub lo: f64,
  pub ctr: f64,
  pub hi: f64,
}

/// Frequency bands generator
/// 
/// The defaults were:
/// 
///       n: 128
///       low: 20.
///       high: 20_000.
///       freq_scale: FreqScale::Logarithmic
///       freq_skew: 0.5
///       bandwidth: 0.5
pub fn generate_freq_bands(
  n: u64, low: f64, high: f64,
  freq_scale: FreqScale,
  freq_skew: f64,
  bandwidth: f64,
) -> Vec<FreqBand> {
  let mut freq_array = Vec::new();
  let low_fscale = fscale(low, freq_scale, freq_skew);
  let high_fscale = fscale(high, freq_scale, freq_skew);
  for i in 0..n {
    freq_array.push(FreqBand {
      lo: inv_fscale(nmap(i as f64 - bandwidth, 0., n as f64 - 1.,
        low_fscale, high_fscale), freq_scale, freq_skew),
      ctr: inv_fscale(nmap(i as f64, 0., n as f64 - 1.,
        low_fscale, high_fscale), freq_scale, freq_skew),
      hi: inv_fscale(nmap(i as f64 + bandwidth, 0., n as f64 - 1.,
        low_fscale, high_fscale), freq_scale, freq_skew),
    });
  }

  freq_array
}

/// The defaults were:
/// 
///       bands_per_octave: 12
///       lower_note: 4
///       higher_note: 124
///       detune: 0.
///       bandwidth: 0.5
pub fn generate_octave_bands(
  bands_per_octave: u64,
  lower_note: u64,
  higher_note: u64,
  detune: f64,
  bandwidth: f64,
) -> Vec<FreqBand> {
  let root24 = 2_f64.powf(1. / 24.);
  let c0 = 440. * root24.powf(-114.); // ~16.35 Hz
  let group_notes = 24. / bands_per_octave as f64;
  
  let mut bands = Vec::new();
  let (s, e) = (
    (lower_note as f64 * 2. / group_notes).round() as u64,
    (higher_note as f64 * 2. / group_notes).round() as u64,
  );
  for i in s..=e {
    bands.push(FreqBand {
      lo: c0 * root24.powf((i as f64 - bandwidth) * group_notes + detune),
      ctr: c0 * root24.powf(i as f64 * group_notes + detune),
      hi: c0 * root24.powf((i as f64 + bandwidth) * group_notes + detune),
    });
  }

  bands
}

// Calculate the FFT
pub fn calc_fft(input: &Vec<f64>) -> Result<Vec<f64>, anyhow::Error> {
  let fft = transform(&input.iter().map(|&x| (x, x)).collect())?;
  let mut output = Vec::new();
  let fft_len = fft.len() as f64;
  for i in 0..((fft_len / 2.).round() as usize) {
    let (x, y) = fft[i];
    output.push(x.hypot(y) / fft_len);
  }

  Ok(output)
}

#[derive(Clone, Copy)]
pub struct ComplexFFT {
  re: f64,
  im: f64,
  magnitude: f64,
  phase: f64,
}

pub fn calc_complex_fft(input: &Vec<f64>) -> Result<Vec<ComplexFFT>, anyhow::Error> {
  let fft = transform(&input.iter().map(|&x| (x, x)).collect())?;
  
  let len = input.len();
  let hlen = len as f64 / 2.;
  let mut output = Vec::new();
  for i in 0..len {
    let (x, y) = fft[i];
    output.push(ComplexFFT {
      re: x / hlen,
      im: y / hlen,
      magnitude: x.hypot(y) / hlen,
      phase: y.atan2(x),
    });
  }

  Ok(output)
}

// Note: that one was really not well-coded on the original version <_<
pub fn calc_goertzel(waveform: &Vec<f64>, coeff: f64) -> f64 {
  let (mut f1, mut f2) = (0., 0.);
  for &x in waveform {
    let sine = x + coeff * f1 - f2;
    f2 = f1;
    f1 = sine;
  }
  
  (sq(f2) + sq(f1) - coeff * f1 * f2).sqrt() / waveform.len() as f64
}

/// The defaults were:
/// 
///       average: true
///       use_rms: true
///       sum: false
///       interpolate: false
///       interp_mode: InterpMode::Linear
///       interp_param: 0.
///       interp_nth_root: 1
///       interp_scale: Scale::Linear
pub fn calc_bandpower(
  fft_coeffs: &Vec<f64>,
  data_idx: f64,
  end_idx: f64,
  average: bool,
  use_rms: bool,
  sum: bool,
  interpolate: bool,
  interp_mode: InterpMode,
  interp_param: f64,
  interp_nth_root: u64,
  interp_scale: Scale,
) -> f64 {
  let fftcl = fft_coeffs.len() as f64;
  let low = clampf(data_idx, 0., fftcl - 1.);
  let high = clampf(end_idx, 0., fftcl - 1.);
  let diff = 1. + end_idx - data_idx;
  let mut amp = 0.;
  if interpolate {
    let ilow = low.trunc();
    amp = if low - ilow <= 0. {
      fft_coeffs[data_idx as usize]
    } else { interp(fft_coeffs,
      low, interp_mode, interp_param,
      interp_nth_root, interp_scale,
      ClipMode::Clip) };
  } else if average {
    for i in (low as usize)..(high as usize) {
      amp += if use_rms { sq(fft_coeffs[i]) } else { fft_coeffs[i] };
    }
    if !sum { amp /= diff; }
    if use_rms { amp *= amp; }
  } else {
    for i in (low as usize)..(high as usize) {
      amp = amp.max(fft_coeffs[i]);
    }
  }

  amp
}

struct BoundThing {
  data_idx: f64,
  end_idx: f64,
  factor: f64,
  interpolate: bool,
}

/// Calculate frequency bands power spectrum from FFT
///
/// - `fft_coeffs`: The array of FFT magnitude coefficients
/// - `bandpower_mode`: FFT to frequency bands spectrum mode, setting to interpolate
/// to do FFT bin interpolation, otherwise to insert zeroes, or bandpower
/// to get average amount of amplitude between 2 frequency boundaries
/// 
/// - `average`: Average to sum FFT coefficients into bandpower coefficient
/// - `hz_array`: The array of frequency bands in Hz
/// 
/// Returns the output of frequency bands spectrum from linearly-spaced FFT coefficients
/// 
/// The defaults were:
/// 
///       interpolate: true
///       bandpower_mode: BandpowerMode::Interpolate
///       average: true
///       use_rms: true
///       sum: false
///       hz_array: generate_freq_bands()
///       bufsize: fft_coeffs.len() * 2
///       sample_rate: 44100
pub fn calc_spectrum(
  fft_coeffs: &Vec<f64>,
  bandpower_mode: BandpowerMode,
  average: bool,
  use_rms: bool,
  sum: bool,
  fft_bin_interpolation: InterpMode,
  interp_param: f64,
  interp_nth_root: u64,
  interp_scale: Scale,
  hz_array: Option<Vec<FreqBand>>,
  bufsize: u64,
  sample_rate: u64,
) -> Vec<f64> {
  let hz_array = hz_array.unwrap_or(
    generate_freq_bands(128, 20., 20_000., FreqScale::Logarithmic, 0.5, 0.5));
  
  let mut bound_arr: Vec<BoundThing> = Vec::new();
  let mut nbars = 0;
  let (fmin, fmax) = match bandpower_mode {
    BandpowerMode::Bandpower => (Converter::Round, Converter::Round),
    _ => (Converter::Ceil, Converter::Floor),
  };
  for fb in hz_array {
    let min_idx = hertz_to_fftbin(fb.lo, fmin, bufsize, sample_rate);
    let max_idx = hertz_to_fftbin(fb.hi, fmax, bufsize, sample_rate);
    let min_idx2 = hertz_to_fftbin(fb.lo, Converter::Floor, bufsize, sample_rate);
    match (bandpower_mode, fft_bin_interpolation) {
      (_, InterpMode::ZeroInsertion) => (),
      (BandpowerMode::Interpolate, _) => {
        if min_idx > max_idx { nbars += 1; }
        else {
          if nbars > 1 {
            let blen = bound_arr.len();
            for j in 0..=nbars {
              bound_arr[blen - j].factor = (nbars - j) as f64 / nbars as f64;
            }
          }
          nbars = 1;
        }
      },
      _ => ()
    }
    let temp_cond = min_idx <= max_idx 
      || bandpower_mode != BandpowerMode::Interpolate
      || fft_bin_interpolation == InterpMode::ZeroInsertion;
    bound_arr.push(BoundThing {
      data_idx: if temp_cond { min_idx } else { min_idx2 },
      end_idx: max_idx,
      factor: 0.,
      interpolate: !temp_cond,
    });
  }

  bound_arr.iter().map(|x| calc_bandpower(
    fft_coeffs, x.data_idx + x.factor,
    x.end_idx, average, use_rms, sum,
    x.interpolate, fft_bin_interpolation,
    interp_param, interp_nth_root,
    interp_scale)).collect()
}

/// Brown-Puckette constant-Q transform algorithm, frequency domain filterbank to turn FFT into CQT
/// - `complex_fft_coeffs`: The complex FFT coefficients to process
/// - `octaveSpacing`: Fractional octave, higher values provides frequency higher resolution
/// - `constantBandwidth`: If set to true, the algorithm will be constant-bandwidth instead
/// - `hzArray`: Array of center frequency bands
/// 
/// Returns Brown-Puckette's CQT output
pub fn calc_cqt(
  complex_fft_coeffs: &Vec<ComplexFFT>,
  octave_spacing: u64,
  constant_bandwidth: bool,
  bandwidth: u64,
  window_function: WindowType,
  window_parameter: f64,
  window_skew: f64,
  hz_array: Option<Vec<FreqBand>>,
  bufsize: u64,
  sample_rate: u64,
) -> Vec<f64> {
  let hz_array = hz_array.unwrap_or(
    generate_freq_bands(128, 20., 20_000., FreqScale::Logarithmic, 0.5, 0.5));
  
  hz_array.iter().map(|band| {
    let freq = band.ctr;
    let (mut a1, mut b1) = (0., 0.);
    let (mut a2, mut b2) = (0., 0.);
    let mut tlen = bufsize as f64 / sample_rate as f64;
    if !constant_bandwidth { tlen = tlen.min(1. / freq * octave_spacing as f64 * 2.); }
    let flen = (bandwidth * bufsize) as f64 / (tlen * sample_rate as f64);
    let center = freq * bufsize as f64 / sample_rate as f64;
    let (start, end) = ((center - flen / 2.).ceil() as usize,
                        (center + flen / 2.).floor() as usize);
    let len = 1 + end - start;

    let mut coeffs = Vec::with_capacity(len);
    for i in start..=end {
      let x = 2. * (i as f64 - center) / flen;
      let w = apply_window(x, window_function, window_parameter, true, window_skew);
      coeffs[i - start] = if i & 1 == 1 { -w } else { w };
    }
    let l = complex_fft_coeffs.len();
    for i in 0..coeffs.len() {
      let u = coeffs[i];
      let x = start + i;
      let y = bufsize as usize - x;
      a1 += u * complex_fft_coeffs[(x % l + l) % l].re;
      a2 += u * complex_fft_coeffs[(x % l + l) % l].im;
      b1 += u * complex_fft_coeffs[(y % l + l) % l].re;
      b2 += u * complex_fft_coeffs[(y % l + l) % l].im;
    }
    let (l1, l2) = (a1 + b1, a2 - b2);
    let (r1, r2) = (b2 + a2, b1 - a1);
    let (real, imag) = (l1.hypot(l2), r1.hypot(r2));
    
    real.hypot(imag) / 8.
  }).collect()
}

pub fn calc_autocorrelation(waveform: &Vec<f64>) -> Vec<f64> {
  let len = waveform.len();
  let mut output = Vec::new();
  for i in 0..len {
    let mut sum = 0.;
    for j in 0..(len - i) {
      let x = waveform[j];
      let y = waveform[j + i];
      sum += x * y;
    }
    output.push(sum / len as f64);
  }

  output
}
