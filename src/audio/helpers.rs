#![allow(dead_code)]
use std::ops::{Add, Sub, Mul, Div};

#[derive(Clone, Copy)]
pub enum ClipMode { Clip, Periodic, Mirror }
#[derive(Clone, Copy)]
pub enum InterpMode { NearestNeighbor, Cubic, Linear, ZeroInsertion }
#[derive(Clone, Copy)]
pub enum Scale { Linear, Logarithmic }

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

pub fn cubic_interp(
  x: f64, y: f64, z: f64,
  w: f64, i: f64,
  tension: Option<f64>
) -> f64 {
  let tension = tension.unwrap_or(0.);
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

pub fn interp(
  arr: &Vec<f64>, x: f64,
  interp_mode: InterpMode,
  interp_parameter: Option<f64>,
  nth_root: Option<u64>,
  scale: Option<Scale>,
  clip_mode: Option<ClipMode>,
) -> f64 {
  //let interp_parameter = interp_parameter.unwrap_or(0.);
  //let nth_root = nth_root.unwrap_or(1);
  //let scale = scale.unwrap_or(Scale::Linear);
  let clip_mode = clip_mode.unwrap_or(ClipMode::Clip);
  let intx = x.trunc() as u64;
  let l = arr.len() as u64;
  match interp_mode {
    InterpMode::NearestNeighbor =>
      arr[clipped_idx(x.round() as u64, arr.len() as u64, clip_mode) as usize],
    InterpMode::Cubic => {
      abs_inv_ascale(cubic_interp(
        abs_ascale(arr[clipped_idx(intx - 1, l, clip_mode) as usize], nth_root, scale),
        abs_ascale(arr[clipped_idx(intx,     l, clip_mode) as usize], nth_root, scale),
        abs_ascale(arr[clipped_idx(intx + 1, l, clip_mode) as usize], nth_root, scale),
        abs_ascale(arr[clipped_idx(intx + 2, l, clip_mode) as usize], nth_root, scale),
        x - intx as f64, interp_parameter), nth_root, scale)
    },
    _ => {
      abs_inv_ascale(lerp(
        abs_ascale(arr[clipped_idx(intx,     l, clip_mode) as usize], nth_root, scale),
        abs_ascale(arr[clipped_idx(intx + 1, l, clip_mode) as usize], nth_root, scale),
        x - intx as f64), nth_root, scale)
    }
  }
}

// Amplitude scaling
pub fn ascale(
  x: f64,
  nth_root: Option<u64>,
  scale: Option<Scale>,
  db_range: Option<f64>,
  use_absolute_value: Option<bool>
) -> f64 {
  let nth_root = nth_root.unwrap_or(1);
  let scale = scale.unwrap_or(Scale::Linear);
  let db_range = db_range.unwrap_or(70.);
  let use_absolute_value = use_absolute_value.unwrap_or(true);
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

pub fn abs_ascale(x: f64, nth_root: Option<u64>, scale: Option<Scale>) -> f64 {
  let nth_root = nth_root.unwrap_or(1);
  let scale = scale.unwrap_or(Scale::Linear);
  match scale {
    Scale::Logarithmic => 20. * x.log10(),
    Scale::Linear => x.powf(1. / nth_root as f64),
  }
}

pub fn abs_inv_ascale(x: f64, nth_root: Option<u64>, scale: Option<Scale>) -> f64 {
  let nth_root = nth_root.unwrap_or(1);
  let scale = scale.unwrap_or(Scale::Linear);
  match scale {
    Scale::Logarithmic => 10_f64.powf(x / 20.),
    Scale::Linear => x.powf(nth_root as f64),
  }
}