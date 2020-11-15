use std::sync::{Arc, Mutex};
use cpal::{Sample, SampleRate};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{RingBuffer, Producer, Consumer};
use std::io::Write;
use anyhow::anyhow;

#[macro_use]
mod macros;
mod braille;
use braille::{BRAILLE, BraillePlot};
mod audio;
use audio::{get_spectrum, SpectrumSettings, helpers, SampleProvider};
use audio::fft::transform;
mod mp3decoder;
use mp3decoder::Mp3Decoder;

fn main() -> Result<(), anyhow::Error> {
  let braille_plot = BraillePlot {
    colgrad: (termion::color::Rgb(0xcc, 0x24, 0x1d), termion::color::Reset),
    position: (1, 1),
    dims: (240, 120),
  };

  println!("{}", BRAILLE.iter().cloned().collect::<String>());
  let host = cpal::default_host();
  let device = host.default_output_device()
    .expect("no output device available");
  
  let decoder = Mp3Decoder::new("yunomi-trackmaker.mp3").unwrap();
  println!("{:#?}", decoder);
  // let duration = decoder.duration();
  let channels = decoder.channels();
  
  let mut supported_configs_range = device
    .supported_output_configs()
    .expect("error while querying configs");
  let mut supported_config = supported_configs_range.next()
    .expect("no supported config?!");
  
  println!("Searching for a config with 2 channels...");
  while supported_config.channels() < channels as u16 {
    match supported_configs_range.next() {
      Some(next) => supported_config = next,
      _ => return Err(anyhow!("No supported config for 2 channels?!")),
    }
  }
  println!("FOUND");
  
  let supported_config = supported_config
    .with_sample_rate(SampleRate((decoder.sample_rate()) as u32));
  println!("channels: {}", supported_config.channels());
  
  let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);
  let sample_format = supported_config.sample_format();
  let config = supported_config.into();
  
  let decoder = Arc::new(Mutex::new(Some(decoder)));
  let decoder2 = decoder.clone();
  const BUFSIZE: usize = 8192;
  let rb = RingBuffer::<i16>::new(BUFSIZE);
  let mut slice = [0; BUFSIZE];
  let rbsplit = Arc::new(Mutex::new(Some(rb.split())));
  let rbsplit2 = rbsplit.clone();
  
  // Trying to follow https://github.com/RustAudio/cpal/blob/master/examples/record_wav.rs... <_<
  let stream = match sample_format {
    cpal::SampleFormat::F32 => device.build_output_stream(&config,
      move |data, _: &_| write_output_data::<f32>(data, &decoder, &rbsplit),
      err_fn)?,
    cpal::SampleFormat::I16 => device.build_output_stream(&config,
      move |data, _: &_| write_output_data::<i16>(data, &decoder, &rbsplit),
      err_fn)?,
    cpal::SampleFormat::U16 => device.build_output_stream(&config,
      move |data, _: &_| write_output_data::<u16>(data, &decoder, &rbsplit),
      err_fn)?,
  };

  stream.play()?;
  println!();

  let mut stdout = std::io::stdout();
  write!(stdout, "{}", termion::cursor::Hide)?;
  
  let settings = SpectrumSettings {
    sample_provider: SampleProvider::FreqKernel,
    input_size: 8192,
    fft_size: 32768,
    sample_out_count: 256,
    hz_range: (10., 200.),
    freq_scale: helpers::FreqScale::Linear,
    hz_linear_factor: 0.5,

    // Those settings maybe should be on a different struct
    sample_rate: 44100,
    win_transform: helpers::WindowType::PowerOfSine,
    win_parameter: 1.,
    win_skew: 0.,
    per_bin_frequency_tilt: 0.,
    per_bin_weighting_amount: 0.,
    per_bin_weighting: helpers::WeightType::A,
    
    bandpower_mode: helpers::BandpowerMode::Interpolate,
    use_average: true,
    use_rms: true,
    use_sum: false,
    fft_bin_interpolation: helpers::InterpMode::Cubic,
    interp_param: 0.,
    interp_nth_root: 1,
    interp_scale: helpers::Scale::Linear,
    cqt_resolution: 24,
    constant_q: true,
    cqt_bandwidth: 8,
    hqac: true,
  };
  
  'main: loop {
    let mut guard = rbsplit2.lock().unwrap();
    let (_, cons) = match guard.as_mut() {
      Some(rs) => rs,
      None => break 'main,
    };
    cons.pop_slice(&mut slice);
    drop(guard);
    
    let waveform = slice.iter().map(|&i| i as f64 / 32768.).collect();
    let spectrum = get_spectrum(&waveform, &settings)?;
    // let spectrum = transform(waveform);
    braille_plot.plot(&mut stdout, spectrum);

    let guard = decoder2.lock().unwrap();
    let decoder = match guard.as_ref() {
      Some(d) => d,
      None => break 'main,
    };
    let done = decoder.done();
    drop(guard);
    if done { break 'main }
    
    std::thread::sleep(std::time::Duration::from_millis(16));
  }
  
  write!(stdout, "{}", termion::cursor::Show)?;
  drop(stream);
  Ok(())
}


type Handler<T> = Arc<Mutex<Option<T>>>;
fn write_output_data<T>(
  output: &mut [T],
  decoder: &Handler<Mp3Decoder>,
  rbsplit: &Handler<(Producer<i16>, Consumer<i16>)>,
) where T: cpal::Sample {
  let mut guard = unwrap_or_return!(decoder.try_lock().ok());
  let decoder = unwrap_or_return!(guard.as_mut());
  
  let channels = decoder.channels();
  for frame in output.chunks_mut(channels) {
    for sample in frame.iter_mut() {
      let next = unwrap_or_return!(decoder.next());
      *sample = Sample::from(&next);
      
      let mut guard = rbsplit.lock().unwrap();
      let (prod, cons) = unwrap_or_return!(guard.as_mut());
      if let Err(next) = prod.push(next) {
        cons.pop();
        prod.push(next).unwrap();
      }
      drop(guard);
    }
  }

}
