use ndarray::{Array1, Array2};

use crate::{node::Node};

pub trait Layer {
    fn run(&self, input: Array1<f64>) -> Array1<f64>;
}

pub struct Hidden {
    // number of nodes 
    n: usize,
    nodes: Vec<Box<dyn Node>>,
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl Hidden {
    // weights & biases should be either wrapped in their own structs,
    // or defined by a method from a trait
    pub fn new(nodes: Vec<Box<dyn Node>>, n: usize, weights: Array2<f64>, biases: Array1<f64>) -> Self {
        Hidden { n, nodes, weights, biases }
    }
}

// LayerFactoryBuilders ??

impl Layer for Hidden {
    fn run(&self, input: Array1<f64>) -> Array1<f64> {
        input
    }
}