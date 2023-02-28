use crate::{hidden_layer::{Biases, Weights, A}, formula};

pub trait Derive {}

pub trait Activation {
  fn get_activation_func(&self) -> &Box<dyn Fn(&A, (&Weights, &Biases)) -> A>;
}

pub trait Node: Activation + Derive {}
pub struct DryNode {
  pub a_fn: Box<dyn Fn(&A, (&Weights, &Biases)) -> A>,
}

impl Activation for DryNode {
    fn get_activation_func(&self) -> &Box<dyn Fn(&A, (&Weights, &Biases)) -> A> {
        &self.a_fn
    }
}

impl Derive for DryNode{}
impl Node for DryNode{}

pub fn sigmoid() -> Box<DryNode> {
  Box::new(DryNode { a_fn: Box::new(|input: &A, wb: (&Weights, &Biases)| -> A {
    let z = wb.0.dot(input) + wb.1;
    z.map(|el| formula::sigmoid(el))
  })})
}

