use termion::color::{Fg, Color};
use std::io::{Write, Stdout};

pub const BRAILLE: [char; 256] = [
  '⠀','⠁','⠂','⠃','⠄','⠅','⠆','⠇','⡀','⡁','⡂','⡃','⡄','⡅','⡆','⡇',
  '⠈','⠉','⠊','⠋','⠌','⠍','⠎','⠏','⡈','⡉','⡊','⡋','⡌','⡍','⡎','⡏',
  '⠐','⠑','⠒','⠓','⠔','⠕','⠖','⠗','⡐','⡑','⡒','⡓','⡔','⡕','⡖','⡗',
  '⠘','⠙','⠚','⠛','⠜','⠝','⠞','⠟','⡘','⡙','⡚','⡛','⡜','⡝','⡞','⡟',
  '⠠','⠡','⠢','⠣','⠤','⠥','⠦','⠧','⡠','⡡','⡢','⡣','⡤','⡥','⡦','⡧',
  '⠨','⠩','⠪','⠫','⠬','⠭','⠮','⠯','⡨','⡩','⡪','⡫','⡬','⡭','⡮','⡯',
  '⠰','⠱','⠲','⠳','⠴','⠵','⠶','⠷','⡰','⡱','⡲','⡳','⡴','⡵','⡶','⡷',
  '⠸','⠹','⠺','⠻','⠼','⠽','⠾','⠿','⡸','⡹','⡺','⡻','⡼','⡽','⡾','⡿',
  '⢀','⢁','⢂','⢃','⢄','⢅','⢆','⢇','⣀','⣁','⣂','⣃','⣄','⣅','⣆','⣇',
  '⢈','⢉','⢊','⢋','⢌','⢍','⢎','⢏','⣈','⣉','⣊','⣋','⣌','⣍','⣎','⣏',
  '⢐','⢑','⢒','⢓','⢔','⢕','⢖','⢗','⣐','⣑','⣒','⣓','⣔','⣕','⣖','⣗',
  '⢘','⢙','⢚','⢛','⢜','⢝','⢞','⢟','⣘','⣙','⣚','⣛','⣜','⣝','⣞','⣟',
  '⢠','⢡','⢢','⢣','⢤','⢥','⢦','⢧','⣠','⣡','⣢','⣣','⣤','⣥','⣦','⣧',
  '⢘','⢙','⢚','⢛','⢜','⢝','⢞','⢟','⣨','⣩','⣪','⣫','⣬','⣭','⣮','⣯',
  '⢰','⢱','⢲','⢳','⢴','⢵','⢶','⢷','⣰','⣱','⣲','⣳','⣴','⣵','⣶','⣷',
  '⢸','⢹','⢺','⢻','⢼','⢽','⢾','⢿','⣸','⣹','⣺','⣻','⣼','⣽','⣾','⣿',
];

pub struct BraillePlot<C1, C2>
where C1: Color + Copy,
      C2: Color + Copy {
  pub colgrad: (C1, C2),
  pub position: (u16, u16),
  pub dims: (u16, u16),
}

impl<C1, C2> BraillePlot<C1, C2>
where C1: Color + Copy,
      C2: Color + Copy {
  pub fn plot(&self, stdout: &mut Stdout, data: Vec<f64>) {
    let (x, mut y) = self.position;
    let (w, h) = self.dims;
    let (c, _) = self.colgrad;
    
    let mut average: f64 = 0.;
    let len = data.len();
    for i in 0..len {
      average += (data[i] as f64).abs() / (len as f64 * 32768.);
    }

    write!(stdout, "{}{}Average: {}{}{}=>",
      termion::clear::All,
      termion::cursor::Goto(x, y),
      Fg(c), average,
      "=".repeat((average * 100.) as usize),
    ).unwrap();
    y += 2;
    
    for dy in 0..(h/4) {
      write!(stdout, "{}", termion::cursor::Goto(x, y + dy));
      for dx in 0..(w/2) {
        write!(stdout, "{}",
          get_braille_char(
            ((data[dx as usize * 2] * 3. * h as f64) as i16
            - h as i16 + (dy as i16 * 4)).max(0) as u16,
            ((data[dx as usize * 2 + 1] * 3. * h as f64) as i16
            - h as i16 + (dy as i16 * 4)).max(0) as u16)
        );
      }
    }
    
    write!(stdout, "{}", Fg(termion::color::Reset));
    stdout.flush().unwrap();
  }
}

pub fn get_braille_char(x1: u16, x2: u16) -> char {
  let n1: usize = match x1 {
    0 => 0b0000,
    1 => 0b1000,
    2 => 0b1100,
    3 => 0b1110,
    _ => 0b1111,
  };
  let n2: usize = match x2 {
    0 => 0b0000,
    1 => 0b1000,
    2 => 0b1100,
    3 => 0b1110,
    _ => 0b1111,
  };

  BRAILLE[(n1 << 4) + n2]
}