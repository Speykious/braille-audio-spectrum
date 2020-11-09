use std::fs::File;
use std::path::Path;
use std::io::{BufRead, BufReader};
// use anyhow::anyhow;

pub struct BraillePlot {
  pub braille: [char; 256],
  colgrad: (u32, u32),
  position: (usize, usize),
  dims: (usize, usize),
}

impl BraillePlot {
  pub fn new<P>(path: P, colgrad: (u32, u32), position: (usize, usize), dims: (usize, usize)) -> Result<BraillePlot, anyhow::Error>
  where P: AsRef<Path> {
    let braille = File::open(path)?;
    let lines = BufReader::new(braille).lines();
    
    let mut braille_plot = BraillePlot {
      braille: [' '; 256],
      colgrad, position, dims,
    };

    let mut i = 0;
    for resline in lines {
      let line = resline?;
      for c in line.chars() {
        braille_plot.braille[i] = c;
        i += 1;
      }
    }
    
    Ok(braille_plot)
  }
}