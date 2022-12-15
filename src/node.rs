use ndarray::{Array2, Dim, Array1};

use crate::{node_io::IO};

pub type Shape = Dim<[usize; 2]>;
pub type Weights<'a> = Option<(Array1<f64>, f64)>;

// CloneNode allows to clone a Boxed dyn Node
pub trait CloneNode {
    fn clone_box(&self) -> Box<dyn Node>;
}

// The point of the whole CloneNode operation:
// Implementation of Clone for a Box<dyn Node>.
// Works only because Node trait extends CloneNode trait.
impl Clone for Box<dyn Node> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// Implementation of CloneNode on trait bound for any implementation of a Node. 
// Make calling "clone()" possible for any Box<dyn Node>.
// Poor man's proc macro :)
impl<T> CloneNode for T
where T: Node + Clone + 'static {
    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

// : CloneNode should be made into a proc macro call (e.g.: #[derive(CloneBox)])
pub trait Node: CloneNode {
    fn run(&self, input: IO, weights: Weights, params: Option<Vec<IO>>) -> Result<(IO, Option<Vec<IO>>), String>;    
    fn derive(&self, prev: Array2<f64>) -> Option<f64>;
    fn should_derive(&self) -> bool;
    fn get_runner_fn(&self) -> Box<dyn Fn(IO, Weights, Option<Vec<IO>>) -> Result<(IO, Option<Vec<IO>>), String>>;
}

// proc macro call to Clone, to comply with the trait bound of CloneNode implementation.
// MulBy is a test node. Will be replaced with more useful node,
// like Logistic Regression node, or Convolution, etc...
#[derive(Clone)]
pub struct MulBy;

impl Node for MulBy {
    fn run(&self,
        input: IO,
        weights: Option<(Array1<f64>, f64)>,
        params: Option<Vec<IO>>
    ) -> Result<(IO, Option<Vec<IO>>), String> {
        self.get_runner_fn()(input, weights, params)
    }

    fn derive(&self, _: Array2<f64>) -> Option<f64> {
        Some(0.)
    }

    fn should_derive(&self) -> bool {
        true
    }

    fn get_runner_fn(&self) -> Box<dyn Fn(IO, Weights, Option<Vec<IO>>) -> Result<(IO, Option<Vec<IO>>), String>> {
        Box::new(|
            input: IO,
            weights: Option<(Array1<f64>, f64)>,
            params: Option<Vec<IO>>
        | -> Result<(IO, Option<Vec<IO>>), String> {
            let (w, b) = match weights {
                Some((_w, _b)) => (_w, _b),
                None => {
                    return Ok((input, None));
                },
            };
            if let IO::Array2(v) = input {
                return Ok((IO::Array1(v.dot(&w)), None));
            }
            Ok((IO::Array1(input * w), None))
        })
    }
}

pub fn mul_by(_: f64) -> MulBy {
    MulBy {}
}

// Output trait defines how an output node should behave.
// This trait will most likely determine when to start back prop.
pub trait Output {
    // predict el famoso y hat
    fn predict(&self, input: Array1<f64>) -> IO;
    fn loss(&self, y: f64, y_hat: f64) -> f64;
    fn cost(&self, losses: Array2<f64>) -> f64;
    // set_activation_fn allows to define how to compute y_hat (classifier),
    // used in Output::predict() method.
    fn set_activation_fn(&mut self, a_fn: fn(Array1<f64>) -> IO);
}

#[derive(Clone)]
pub struct Activation {
    pub a_fn: fn(Array1<f64>) -> IO,
}

impl Activation {
    pub fn new(activation_func: fn(Array1<f64>) -> IO) -> Self {
        Activation { a_fn: activation_func }
    }
} 

impl Output for Activation {
    fn predict(&self, input: Array1<f64>) -> IO {
        (self.a_fn)(input)
    }

    fn loss(&self, y: f64, y_hat: f64) -> f64 {
        -((y * y_hat.ln()) + ((1. - y) * (1. - y_hat).ln()))
    }

    fn cost(&self, losses: Array2<f64>) -> f64 {
        losses.sum() / losses.len() as f64
    }

    fn set_activation_fn(&mut self, a_fn: fn(Array1<f64>) -> IO) {
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