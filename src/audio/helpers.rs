#![allow(dead_code)]
use std::ops::{Add, Sub, Mul, Div};
use std::f64::consts::{PI, TAU, FRAC_PI_2, SQRT_2};

#[derive(Clone, Copy)]
pub enum ClipMode { Clip, Periodic, Mirror }
#[derive(Clone, Copy)]
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

pub fn nmap<T>(x: T, min: T, max: T, tmin: T, tmax: T) -> T
where T: Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Copy {
  (x - min) / (max - min) * (tmax - tmin) + tmin
}

pub fn clamp<T>(x: T, min: T, max: T) -> T
where T: Ord {
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
  let squared = i.powf(2.);
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
  let f2 = x.powf(2.);
  match weight_type {
    WeightType::A =>
      (1.2588966 * 148840000. * f2.powf(2.) / ((f2 + 424.36)
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
             + 2.043828333606125e-15 * f2.powf(2.)
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
/// `x`: The position of the window from -1 to 1
/// `windowType` The specified window function to use
/// `windowParameter` The parameter of the window function (Adjustable window functions only)
/// `truncate` Zeroes out if x is more than 1 or less than -1
/// `windowSkew` Skew the window function to make symmetric windows asymmetric
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
    ((x / 2. - 0.5) / (1. - (x / 2. - 0.5) * 10. * window_skew.powf(2.)))
    / (1. / (1. + 10. * window_skew.powf(2.))) * 2. + 1.
  } else {
    ((x / 2. + 0.5) / (1. + (x / 2. + 0.5) * 10. * window_skew.powf(2.)))
    / (1. / (1. + 10. * window_skew.powf(2.))) * 2. - 1.
  };
  
  if truncate && x.abs() > 1. { return 0.; }

  match window_type {
    WindowType::Hann => (x / FRAC_PI_2).cos().powf(2.),
    WindowType::Hamming => 0.54 + 0.46 + (x * PI).cos(),
    WindowType::PowerOfSine => (x * FRAC_PI_2).cos().powf(window_parameter),
    WindowType::Tukey =>
      if x.abs() <= 1. - window_parameter { 1. }
      else if x > 0. { -(((x - 1.) * PI / window_parameter / 2.).sin()).powf(2.) }
      else { ((x + 1.) * PI / window_parameter / 2.).sin().powf(2.) },
    WindowType::Blackman => 0.42 + 0.5 * (x * PI).cos() + 0.08 * (x * TAU).cos(),
    WindowType::Nuttall => 0.355768
      + 0.487396 * (x * PI).cos()
      + 0.144232 * (x * TAU).cos()
      + 0.012604 * (x * PI * 3.).cos(),
    WindowType::Kaiser =>
      ((1. - x.powf(2.)).sqrt() * window_parameter.powf(2.)).cosh()
      / window_parameter.powf(2.).cosh(),
                          // -a²b² = -(a²b²) = -(ab)²
    WindowType::Gauss => (-(window_parameter * x).powf(2.)).exp(),
    WindowType::Bartlett => 1. - x.abs(),
    WindowType::QuadraticSpline =>
      if x.abs() <= 0.5 { -(x * SQRT_2).powf(2.) + 1. }
      else { ((x * SQRT_2).abs() - SQRT_2).powf(2.) },
    WindowType::Welch => 1. - x.powf(2.),
    WindowType::None => 1.,
  }
}