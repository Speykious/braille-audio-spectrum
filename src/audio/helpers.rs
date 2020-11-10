#![allow(dead_code)]
use std::ops::{Add, Sub, Mul, Div};

pub enum ClipMode { Clip, Periodic, Mirror }

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

