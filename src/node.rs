use crate::{Weights, Bias, IO};

pub trait Node<F>
where F: Fn(&IO, Option<(Weights, Bias)>) -> IO {
  // get_activation_func returns a NodeActivationFunc that will
  // compute input and give an output.
  fn get_activation_func(&self) -> &F;
}

// pub trait Derive {
//   fn get_derive_func
// }

pub struct DryNode<F>
where F: Fn(&IO, Option<(Weights, Bias)>) -> IO {
  pub a_fn: F,
}

impl<F> Node<F> for DryNode<F>
where F: Fn(&IO, Option<(Weights, Bias)>) -> IO {
    fn get_activation_func(&self) -> &F {
        &self.a_fn
    }
}