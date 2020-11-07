use std::sync::{Arc, Mutex};
use cpal::{Data, Sample, SampleFormat};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleRate;

mod mp3decoder;
use mp3decoder::Mp3Decoder;

fn main() -> Result<(), anyhow::Error> {
  let host = cpal::default_host();
  let device = host.default_output_device()
    .expect("no output device available");
  
  let decoder = Mp3Decoder::new("yunomi-trackmaker.mp3").unwrap();
  
  let mut supported_configs_range = device
    .supported_output_configs()
    .expect("error while querying configs");
  let supported_config = supported_configs_range.next()
    .expect("no supported config?!")
    .with_sample_rate(SampleRate(decoder.sample_rate() as u32));
  
  let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);
  let sample_format = supported_config.sample_format();
  let config = supported_config.into();
  
  let decoder = Arc::new(Mutex::new(Some(decoder)));
  
  // Trying to follow https://github.com/RustAudio/cpal/blob/master/examples/record_wav.rs... <_<
  let decoder_2 = decoder.clone();
  
  let stream = match sample_format {
    cpal::SampleFormat::F32 => device.build_output_stream(&config,
        move |data, _: &_| write_output_data::<f32>(data, &decoder_2),
        err_fn)?,
    cpal::SampleFormat::I16 => device.build_output_stream(&config,
        move |data, _: &_| write_output_data::<i16>(data, &decoder_2),
        err_fn)?,
    cpal::SampleFormat::U16 => device.build_output_stream(&config,
        move |data, _: &_| write_output_data::<u16>(data, &decoder_2),
        err_fn)?,
  };

  stream.play()?;
  println!("Hello, world!");
  
  // Let recording go for roughly three seconds.
  std::thread::sleep(std::time::Duration::from_secs(3));
  drop(stream);
  Ok(())
}

/*
fn write_silence<T: Sample>(data: &mut [T], _: &cpal::OutputCallbackInfo) {
  for sample in data.iter_mut() {
    *sample = Sample::from(&0.0);
  }
}

fn write_decoder<T: Sample>(decoder: &mut Mp3Decoder, data: &mut [T], _: &cpal::OutputCallbackInfo) {
  for sample in data.iter_mut() {
    match decoder.next() {
      Some(next) => *sample = Sample::from(&next),
      _ => ()
    }
  }
}
*/

type DecoderHandle = Arc<Mutex<Option<Mp3Decoder>>>;

fn write_output_data<T>(output: &mut [T], decoder: &DecoderHandle)
where T: cpal::Sample {
  if let Ok(mut guard) = decoder.try_lock() {
    if let Some(decoder) = guard.as_mut() {
      for sample in output.iter_mut() {
        match decoder.next() {
          Some(next) => {
            *sample = Sample::from(&(next / 3))
          },
          _ => (),
        }
      }
    }
  }
}