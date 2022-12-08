use ndarray::{Ix2, Array1, array};

use crate::{node::Node, node_factory::NodeFactory};

pub trait Builder {
    fn build(&self, prev_n: usize) -> Result<usize, String>;
}

pub trait Layer: Builder {
    fn run(&self, input: Array1<f64>) -> Array1<f64>;
}

pub struct Hidden {
    n: usize,
    factories: Vec<NodeFactory>,
    nodes: Vec<Box<dyn Node>>
}

impl Hidden {
    pub fn new(factories: Vec<NodeFactory>, n: usize) -> Self {
        Hidden { n, factories, nodes: Vec::<Box<dyn Node>>::new() }
    }
}

// LayerFactoryBuilders ??

impl Layer for Hidden {
    fn run(&self, input: Array1<f64>) -> Array1<f64> {
        input
    }
    
}

impl Builder for Hidden {
    /**
     * prev_n: usize = last node of previous layer spawn amount (n)
     */
    fn build(&self, prev_n: usize) -> Result<usize, String> {
        let mut n = prev_n;
        let mut nodes: Vec<Box<dyn Node>> = Vec::new();

        for factory in &self.factories {
            let shape = Ix2(n, self.n);
            println!("shape: {:?}", shape);
            
            n = 1;
            nodes.push(factory.build(shape)?);
        }

        self.nodes = nodes;
        Ok(self.n)
    }
}
