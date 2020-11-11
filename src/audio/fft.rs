/* 
 * Free FFT and convolution (Rust)
 * 
 * Copyright (c) 2020 Speykious. (MIT License)
 * 
 * Ported to Rust from Javascript code:
 * https://www.nayuki.io/page/free-small-fft-in-multiple-languages
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * - The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 * - The Software is provided "as is", without warranty of any kind, express or
 *   implied, including but not limited to the warranties of merchantability,
 *   fitness for a particular purpose and noninfringement. In no event shall the
 *   authors or copyright holders be liable for any claim, damages or other
 *   liability, whether in an action of contract, tort or otherwise, arising from,
 *   out of or in connection with the Software or the use or other dealings in the
 *   Software.
 */
use std::f64::consts::{PI, TAU};
use anyhow::anyhow;

pub fn is_pow2(x: u64) -> bool {
  (x != 0) && ((x & (x - 1)) == 0)
}

type Complex = (f64, f64);

/// Computes the discrete Fourier transform (DFT) of the given
/// complex vector, storing the result back into the vector.
/// The vector can have any length. This is a wrapper function.
pub fn transform(input: &Vec<Complex>) -> Result<Vec<Complex>, anyhow::Error> {
  if is_pow2(input.len() as u64) {
    transform_radix2(input)
  } else {
    transform_bluestein(input)
  }
}

/// Computes the inverse discrete Fourier transform (IDFT) of
/// the given complex vector, storing the result back into the vector.
/// The vector can have any length. This is a wrapper function.
/// This transform does perform scaling, so the inverse is a true inverse.
pub fn inverse_transform(input: &Vec<Complex>) -> Result<Vec<Complex>, anyhow::Error> {
  let n = input.len() as f64;
  transform(&input.iter().map(|(re, im)| (*im / n, *re / n)).collect())
}

/// Computes the discrete Fourier transform (DFT) of the given
/// complex vector, storing the result back into the vector.
/// The vector's length must be a power of 2. Uses the Cooley-Tukey
/// decimation-in-time radix-2 algorithm.
/// 
/// Note: this one has been quite heavily modified (hopefully it still works)
pub fn transform_radix2(input: &Vec<Complex>) -> Result<Vec<Complex>, anyhow::Error> {
  // Length variables
  let n = input.len() as u64;
  if n <= 1 { return Ok(input.clone()); } // Trivial transform
  if !is_pow2(n) {
    return Err(anyhow!("Length of input vector is not a power of 2"))
  }
  let logn = (n as f64).log2();
  
  // Trigonometric tables
  let (mut cos_table, mut sin_table) = (Vec::new(), Vec::new());
  for i in 0..(n / 2) {
    let el = TAU * i as f64 / n as f64;
    cos_table.push(el.cos());
    sin_table.push(el.sin());
  }
  
  let mut output = input.clone();

  fn reverse_bits(mut x: u64, bits: u8) -> u64 {
    let mut y = 0;
    for _ in 0..bits {
      y = (y << 1) | (x & 1);
      x >>= 1;
    }
    y
  }

  // Bit-reversed addressing permutation
  for i in 0..n {
    let j = reverse_bits(i, logn as u8);
    if j > i {
      let a = output[i as usize];
      let b = output[j as usize];
      output[i as usize] = b;
      output[j as usize] = a;
    }
  }

	// Cooley-Tukey decimation-in-time radix-2 FFT
  fn cooley_tukey(
    n: u64,
    output: &mut Vec<Complex>,
    cos_table: Vec<f64>,
    sin_table: Vec<f64>,
    compute: fn(Complex, Complex, Complex) -> (Complex, Complex),
  ) {
    let mut size = 1;
    while size <= n {
      let hsize = size;
      size *= 2;
      let tablestep = n / size;
      let mut i = 0;
      while i < n {
        let mut k = 0;
        for j in i..(i + hsize) {
          let (adx, bdx) = ((j + hsize) as usize, j as usize);
          let ((r1, i1), (r2, i2)) = compute(
            output[adx], output[bdx],
            (cos_table[k], sin_table[k]));
          output[adx] = (r1, i1);
          output[bdx] = (r2, i2);
          k += tablestep as usize;
        }
        i += size;
      }
    }
  }

  fn compute(
    (r1, i1): Complex,
    (r2, i2): Complex,
    (ctk, stk): Complex
  ) -> (Complex, Complex) {
    let tpre =   r1 * ctk + i1 * stk;
    let tpim = - r1 * stk + i1 * ctk;
    ((r2 - tpre, i2 - tpim), (r2 + tpre, i2 + tpim))
  }
  
  cooley_tukey(n, &mut output, cos_table, sin_table, compute);
  
  Ok(output)
}

/// Computes the discrete Fourier transform (DFT) of the given
/// complex vector, storing the result back into the vector.
/// The vector can have any length. This requires the convolution
/// function, which in turn requires the radix-2 FFT function.
/// Uses Bluestein's chirp z-transform algorithm.
pub fn transform_bluestein(input: &Vec<Complex>) -> Result<Vec<Complex>, anyhow::Error> {
	// Find a power-of-2 convolution length m such that m >= n * 2 + 1
  let n = input.len();
  let m = (1 << (n as f64 * 2.).log2() as u64) + 1;

  // Trigonometric tables
  let (mut cos_table, mut sin_table) = (Vec::new(), Vec::new());
  for i in 0..n {
    let j = i * i % (n * 2); // This is more accurate than j = i * i
    cos_table.push((PI * j as f64 / n as f64).cos());
    sin_table.push((PI * j as f64 / n as f64).sin());
  }
  
  let mut a = Vec::new();
  for i in 0..n {
    let (ri, ii) = input[i];
    let (cti, sti) = (cos_table[i], sin_table[i]);
    a.push((ri * cti + ii * sti, -ri * sti + ii * cti));
  }
  let mut b = Vec::with_capacity(m);
  for i in 0..n {
    let temp = (cos_table[i], sin_table[i]);
    b[i] = temp; b[m - i] = temp;
  }

  let mut output = convolve_complex(&a, &b)?;
  // Post-processing
  for i in 0..n {
    let (cti, sti) = (cos_table[i], sin_table[i]);
    let (or, oi) = output[i];
    output[i] = (or * cti + oi * sti, -or * sti + oi * cti);
  }

  Ok(output)
}

/*
// This one is not used anywhere from what I can tell
pub fn convolve_real(xs: &Vec<f64>, ys: &Vec<f64>) -> Result<Vec<Complex>, anyhow::Error> {
  if xs.len() == ys.len() {
    convolve_complex(
      &xs.iter().map(|&x| (x, 0.)).collect(),
      &ys.iter().map(|&y| (y, 0.)).collect())
  } else { Err(anyhow!("Mismatched lengths")) }
}
*/

/// Computes the circular convolution of the given complex
/// vectors. Each vector's length must be the same.
pub fn convolve_complex(xs: &Vec<Complex>, ys: &Vec<Complex>) -> Result<Vec<Complex>, anyhow::Error> {
  let n = xs.len();
  if n != ys.len() { return Err(anyhow!("Mismatched lengths")); }

  let mut xs = transform(xs)?;
  let ys = transform(ys)?;
  for i in 0..n {
    let ((xr, xi), (yr, yi)) = (xs[i], ys[i]);
    xs[i] = (xr * yr - xi * yi, xi * yr + xr * yi);
  }

  inverse_transform(&xs)
}
