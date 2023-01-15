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

// impl Clone for Node {
//     fn clone(&self) -> Self {
//         Self { a_fn: Box::new(|input: &IO, wb: Option<(Weights, Bias)>| -> IO {
//           (self.a_fn)(input, wb)
//         })}
//     }
// }

impl Activation for DryNode {
    fn get_activation_func(&self) -> &Box<dyn Fn(&IO, Option<(Weights, Bias)>) -> IO> {
        &self.a_fn
    }
}

impl Derive for DryNode {}
impl Node for DryNode {}