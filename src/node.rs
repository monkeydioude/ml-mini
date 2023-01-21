use crate::{Weights, Bias, IO};

pub trait Derive {
}

pub trait Activation {
  fn get_activation_func(&self) -> &Box<dyn Fn(&IO, Option<(Weights, Bias)>) -> IO>;
}

pub trait Node: Activation + Derive {}
pub struct DryNode {
  pub a_fn: Box<dyn Fn(&IO, Option<(Weights, Bias)>) -> IO>,
}


impl Activation for DryNode {
    fn get_activation_func(&self) -> &Box<dyn Fn(&IO, Option<(Weights, Bias)>) -> IO> {
        &self.a_fn
    }
}

impl Derive for DryNode {}
impl Node for DryNode {}
