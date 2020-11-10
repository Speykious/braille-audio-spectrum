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

    write!(stdout, "{}{}Average: {}{}{}=>",
      termion::clear::All,
      termion::cursor::Goto(x, y),
      Fg(c),
      data[0],
      "=".repeat((data[0] * 100.) as usize),
    ).unwrap();
    y += 2;
    
    /*
    for dy in 0..5 {
      write!(stdout, "{}", termion::cursor::Goto(x, y + dy));
      for dx1 in 0..5 {
        for dx2 in 0..5 {
          write!(stdout, "{}", get_braille_char(dx1, dx2));
        }
      }
    }
    */
    
    for dy in 0..(h/4) {
      write!(stdout, "{}", termion::cursor::Goto(x, y + dy));
      for _ in 0..(w/2) {
        write!(stdout, "{}",
          get_braille_char(
            ((data[0] * 3. * h as f64) as i16 - h as i16 + (dy as i16 * 4)).max(0) as u16,
            2
          )
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