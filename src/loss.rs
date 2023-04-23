use crate::formula;

pub trait Loss {
  fn loss(&self, y: &f64, y_hat: &f64) -> f64;
}

pub struct DryLoss(fn(&f64, &f64) -> f64);

impl Loss for DryLoss {
    fn loss(&self, y: &f64, y_hat: &f64) -> f64 {
        self.0(y, y_hat)
    }
}

pub fn basic_loss() -> Box<dyn Loss> {
  Box::new(DryLoss(formula::basic_loss))
}