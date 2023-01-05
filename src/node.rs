use crate::{Weights, Bias, IO};

// NodeActivationFunc type is the core function of a Node.
// This is what will compute input and provide an output
// with respect, if provided for this node, weights and bias.
pub type NodeActivationFunc = dyn Fn(&IO, Option<(Weights, Bias)>) -> IO;

pub trait Node {
  // get_activation_func returns a NodeActivationFunc that will
  // compute input and give an output.
  fn get_activation_func(&self) -> &dyn Fn(f64) -> f64;
}

pub struct DryNode {
  pub a_fn: Box<dyn Fn(f64) -> f64>,
}

impl Node for DryNode {
    fn get_activation_func(&self) -> &dyn Fn(f64) -> f64 {
        &self.a_fn
    }
}