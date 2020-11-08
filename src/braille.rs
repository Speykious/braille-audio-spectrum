use std::fs::File;
use std::path::Path;
use std::io::{BufRead, BufReader};

pub struct BraillePlot {
  braille: [char; 256],
  colgrad: (u32, u32),
  position: (usize, usize),
  dims: (usize, usize),
}

impl BraillePlot {
  fn new<P>(path: P, colgrad: (u32, u32), position: (usize, usize), dims: (usize, usize)) -> Result<BraillePlot, anyhow::Error>
  where P: AsRef<Path> {
    let braille = File::open("braille.txt")?;
    let lines = BufReader::new(braille).lines();
    
    let braille_plot = BraillePlot {
      braille: [' '; 256],
      colgrad, position, dims,
    }
  }
}