use std::ops::Mul;

use ndarray::{Array2, Array};

pub struct VecBuilder<T: Clone>{
  v: Vec<T>
}

impl<T: Clone> VecBuilder<T> {
  pub fn new() -> Self {
    VecBuilder { v: vec![] }
  }

  pub fn func(occ: usize, func: &dyn Fn() -> T) -> Vec<T> {
    VecBuilder::new().self_func(occ, func)
  }

  pub fn self_func(&mut self, occ: usize, func: &dyn Fn() -> T) -> Vec<T> {
    for i in 0..occ {
      self.v.push(func());
    }
    self.v.clone()
  }
}

pub fn fill_array2(shape: (usize, usize)) -> Array2<f64> {
  match Array::from_shape_vec(shape, VecBuilder::func((shape.0 as i32 * shape.1 as i32) as usize, &|| -> f64 {
      rand::random::<f64>()
    })) {
      Ok(r) => r,
      Err(err) => panic!("{}", err),
  }
}