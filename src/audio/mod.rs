pub mod helpers;
pub mod fft;

use helpers::{
  FreqScale,
  generate_freq_bands,
  apply_window,
  WindowType,
  calc_fft,
  calc_cqt,
  calc_spectrum,
  calc_freq_tilt,
  calc_complex_fft,
  apply_weight,
  WeightType,
  fftbin_to_hertz,
  BandpowerMode,
  InterpMode,
  Scale,
};

#[derive(Clone, Copy, PartialEq)]
pub enum SampleProvider { Spectrum, FreqKernel, Waveform }
pub struct SpectrumSettings {
  pub sample_provider: SampleProvider,
  pub input_size: u64,
  pub fft_size: u64,
  pub sample_out_count: u64,
  pub hz_range: (f64, f64),
  pub freq_scale: FreqScale,
  pub hz_linear_factor: f64,

  // Those settings maybe should be on a different struct
  pub sample_rate: u64,
  pub win_transform: WindowType,
  pub win_parameter: f64,
  pub win_skew: f64,
  pub per_bin_frequency_tilt: f64,
  pub per_bin_weighting_amount: f64,
  pub per_bin_weighting: WeightType,
  
  pub bandpower_mode: BandpowerMode,
  pub use_average: bool,
  pub use_rms: bool,
  pub use_sum: bool,
  pub fft_bin_interpolation: InterpMode,
  pub interp_param: f64,
  pub interp_nth_root: u64,
  pub interp_scale: Scale,
  pub cqt_resolution: u64,
  pub constant_q: bool,
  pub cqt_bandwidth: u64,
  pub hqac: bool,
}

pub fn get_spectrum(waveform: &Vec<f64>, settings: &SpectrumSettings) -> Result<Vec<f64>, anyhow::Error> {
  let bufsize = match settings.sample_provider {
    SampleProvider::Spectrum => settings.fft_size,
    _ => settings.input_size,
  };
  let waveform_size = match settings.sample_provider {
    SampleProvider::Spectrum => settings.fft_size.min(settings.input_size),
    _ => settings.input_size,
  };
  let (low, high) = settings.hz_range;
  let freq_bands = generate_freq_bands(
    settings.sample_out_count,
    low, high, settings.freq_scale,
    settings.hz_linear_factor, 0.5);
  let hqac = if settings.hqac { 1 } else { 0 };
  let sample_rate = settings.sample_rate / (2 - hqac);
  let is_spectrum = settings.sample_provider != SampleProvider::Waveform;
  
  let mut audio_buffer = Vec::with_capacity(bufsize as usize);
  let mut norm = 0.;
  for i in 0..(waveform_size as usize) {
    // Apply window function
    let x = (i * 2) as f64 / (waveform_size - 1) as f64 - 1.;
    let w = apply_window(x, settings.win_transform,
      settings.win_parameter, true, settings.win_skew);
    //println!("i: {} | fft_size: {} | waveform_size: {}", i, settings.fft_size, waveform_size);
    audio_buffer.push(waveform[i /* + settings.fft_size as usize - waveform_size as usize */] * w);
    norm += w;
  }
  if is_spectrum {
    let len = audio_buffer.len() as f64;
    audio_buffer = audio_buffer.iter().map(|x| x * len / norm).collect();
  }
  if !settings.hqac {
    let mut new_buffer = Vec::new();
    for i in 0..(audio_buffer.len() / 2) {
      new_buffer.push(audio_buffer[i * 2]);
    }
    audio_buffer = new_buffer;
  }
  
  let result;
  match settings.sample_provider {
    SampleProvider::Spectrum => {
      let spectrum = calc_fft(&audio_buffer)?;
      result = calc_spectrum(
        &{ let mut temp = Vec::new();
          for i in 0..spectrum.len() {
            temp.push(spectrum[i]
            * calc_freq_tilt(fftbin_to_hertz(i as f64 + 0.5, 4096, 44100),
              440., settings.per_bin_frequency_tilt)
            * apply_weight(fftbin_to_hertz(i as f64 + 0.5, 4096, 44100),
              settings.per_bin_weighting_amount, settings.per_bin_weighting));
          }
          temp },
        settings.bandpower_mode,
        settings.use_average,
        settings.use_rms,
        settings.use_sum,
        settings.fft_bin_interpolation,
        settings.interp_param,
        settings.interp_nth_root,
        settings.interp_scale,
        Some(freq_bands),
        audio_buffer.len() as u64,
        sample_rate);
    },
    SampleProvider::FreqKernel => {
      result = calc_cqt(
        &calc_complex_fft(&audio_buffer)?,
        settings.cqt_resolution,
        !settings.constant_q,
        settings.cqt_bandwidth,
        settings.win_transform,
        settings.win_parameter,
        settings.win_skew,
        Some(freq_bands),
        audio_buffer.len() as u64,
        sample_rate);
    },
    SampleProvider::Waveform => {
      result = audio_buffer;
    }
  }

  Ok(result)
}