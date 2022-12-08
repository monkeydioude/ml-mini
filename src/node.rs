use ndarray::{Array2, Dim, Array1};

use crate::{node_io::IO};

pub type Shape = Dim<[usize; 2]>;
pub type Weights = Option<(&Array2<f64>, &Array1<f64>)>;

pub trait Node {
    fn run(&self, input: IO, weights: Weights, params: Option<Vec<IO>>) -> Result<(IO, Option<Vec<IO>>), String>;    
    // fn validate_shapes(&self, input_shape: Shape) -> bool;
    // fn set_weights(&mut self, w: Array2<f64>);
    // fn get_w(&self) -> Option<&Array2<f64>>;
    fn derive(&self, prev: Array2<f64>) -> Option<f64>;
    fn should_derive(&self) -> bool;
    fn get_runner_fn(&self) -> fn(IO, Weights, Option<Vec<IO>>) -> Result<(IO, Option<Vec<IO>>), String>;
    // fn train(&mut self, d: f64) -> Result<(), String>;
}

/* X2 struct */

pub struct MulBy {
}

impl Node for MulBy {
    fn run(&self,
        input: IO,
        weights: Option<(&Array2<f64>, &Array1<f64>)>,
        params: Option<Vec<IO>>
    ) -> Result<(IO, Option<Vec<IO>>), String> {
        if let IO::Array2(v) = input {
            return Ok((IO::Array2(v.dot(&self.w)), None));
        }
        Ok((IO::Array2(input * self.w.clone()), None))
    }

    fn derive(&self, _: Array2<f64>) -> Option<f64> {
        Some(0.)
    }

    fn should_derive(&self) -> bool {
        true
    }
}

/**
 * n = nodes spawn amount aka first dim of weights 'w' 
 */
pub fn mul_by(_: f64) -> MulBy {
    MulBy {}
}

/* Value  struct */

type Value = IO;

impl Node for Value {
    fn run(&self,
        input: IO, 
        weights: Option<(&Array2<f64>, &Array1<f64>)>, 
        params: Option<Vec<IO>>
    ) -> Result<(IO, Option<Vec<IO>>), String> {
        input.null()?;
        Ok((self.clone(), None))
    }

    fn derive(&self, _: Array2<f64>) -> Option<f64> {
        Some(0.)
    }

    fn should_derive(&self) -> bool {
        false
    }
}

pub fn value(value: f64) -> Value {
    IO::F64(value)
}

pub trait Output {
    fn predict(&self, input: Array1<f64>) -> f64;
    fn loss(&self, y: f64, y_hat: f64) -> f64;
    fn cost(&self, losses: Array2<f64>) -> f64;
    fn set_activation_fn(&mut self, a_fn: fn(Array1<f64>) -> f64);
}

#[derive(Clone)]
pub struct Activation {
    pub a_fn: fn(Array1<f64>) -> f64,
}

impl Activation {
    pub fn new(activation_func: fn(Array1<f64>) -> f64) -> Self {
        Activation { a_fn: activation_func }
    }
}

impl Output for Activation {
    fn predict(&self, input: Array1<f64>) -> f64 {
        (self.a_fn)(input)
    }

    fn loss(&self, y: f64, y_hat: f64) -> f64 {
        -((y * y_hat.ln()) + ((1. - y) * (1. - y_hat).ln()))
    }

    fn cost(&self, losses: Array2<f64>) -> f64 {
        losses.sum() / losses.len() as f64
    }

    fn set_activation_fn(&mut self, a_fn: fn(Array1<f64>) -> f64) {
        self.a_fn = a_fn;
    }
}

impl Activation {
    fn derive(&self, _: Array2<f64>) -> Option<f64> {
        Some(0.)
    }

    fn should_derive(&self) -> bool {
        true
    }    
}