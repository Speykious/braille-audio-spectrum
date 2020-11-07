use std::sync::{Arc, Mutex, MutexGuard};
use cpal::{Sample, SampleRate};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::{RingBuffer, Producer, Consumer};
use anyhow::anyhow;

#[macro_use]
mod macros;
mod mp3decoder;
use mp3decoder::Mp3Decoder;

fn main() -> Result<(), anyhow::Error> {
  let host = cpal::default_host();
  let device = host.default_output_device()
    .expect("no output device available");
  
  let decoder = Mp3Decoder::new("yunomi-trackmaker.mp3").unwrap();
  println!("{:#?}", decoder);
  let duration = decoder.duration();
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
  let rb = RingBuffer::<i16>::new(8192);
  let rbsplit = Arc::new(Mutex::new(Some(rb.split())));
  
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
  
  'main: loop {
    // TODO: CODE THE RINGBUFFER'S PERIODIC CONSUMPTION HERE
    
    std::thread::sleep(std::time::Duration::from_millis(16));
  }

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
  let mut guard = unwrap_or_return!(rbsplit.try_lock().ok());
  let (prod, cons) = unwrap_or_return!(guard.as_mut());
  
  let channels = decoder.channels();
  for frame in output.chunks_mut(channels) {
    for sample in frame.iter_mut() {
      let next = unwrap_or_return!(decoder.next());
      *sample = Sample::from(&next);
      
      if let Err(next) = prod.push(next) {
        cons.pop();
        prod.push(next).unwrap();
      }
    }
  }
}