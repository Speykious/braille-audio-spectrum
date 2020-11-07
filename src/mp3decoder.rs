
use std::fs::File;
use std::path::Path;
use std::io::{BufReader};
use minimp3::{Decoder, Frame};

#[allow(dead_code)]
pub struct Mp3Decoder {
  decoder: Decoder<BufReader<File>>,
  current_frame: Frame,
  sample_rate: i32,
  channels: usize,
  layer: usize,
  bitrate: i32,
  duration: u64,
  current_frame_offset: usize,
}

#[allow(dead_code)]
impl Mp3Decoder {
  pub fn new<P>(file: P) -> Result<Self, ()>
  where P: AsRef<Path> {
    // First consumation of `file`
    let metadata = mp3_metadata::read_from_file(&file).unwrap();
    let duration = metadata.duration.as_millis() as u64;

    // Second consumption of `file`
    let buffer = BufReader::new(File::open(file).unwrap());
    let mut decoder = Decoder::new(buffer);
    let current_frame = decoder.next_frame().map_err(|_| ())?;
    let sample_rate = current_frame.sample_rate;
    let channels = current_frame.channels;
    let layer = current_frame.layer;
    let bitrate = current_frame.bitrate;

    Ok(Mp3Decoder {
      decoder,
      current_frame,
      sample_rate,
      channels,
      layer,
      bitrate,
      duration,
      current_frame_offset: 0,
    })
  }

  pub fn sample_rate(&self) -> i32 {
    self.sample_rate
  }

  pub fn channels(&self) -> usize {
    self.channels
  }

  pub fn layer(&self) -> usize {
    self.layer
  }

  pub fn bitrate(&self) -> i32 {
    self.bitrate
  }
}

impl Iterator for Mp3Decoder {
  type Item = i16;

  #[inline]
  fn next(&mut self) -> Option<i16> {
    let l = self.current_frame.data.len();
    println!("Length of current_frame.data: {}", l);
    if self.current_frame_offset == l {
      match self.decoder.next_frame() {
        Ok(frame) => self.current_frame = frame,
        _ => return None,
      }
      self.current_frame_offset = 0;
    }

    let v = self.current_frame.data[self.current_frame_offset];

    self.current_frame_offset += 1;

    Some(v)
  }
}
