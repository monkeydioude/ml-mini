use ndarray::Array2;

use crate::{io_filter::Filter, wb::WeightsBiasesInitializer, node::Node};

// Weights layout is made to match previous layer's n (no need to transpose)
pub type Weights = Array2<f64>;
pub type Biases = Array2<f64>;
pub type A = Array2<f64>;

pub struct Units(pub usize);

pub trait Layer {
    // run takes an Array2<f64> as input, which will be used by each node
    // to compute a result,using weights and biases.
    // ex:  input.shape() = (1, 5)
    //      previous_layer.get_n() = 5
    //      layer.get_n() = 4
    //      weights.shape() = (5, 4)
    //      biases.shape() = (4)
    //      -> output.shape() = (1, 4)
    #[inline]
    fn run(&self, input: A) -> A {
        let mut out = input;
        let filters = self.get_filters();

        filters[0].iter().for_each(|node| {
            println!("before a filter {}", out);
            out = node.filter(&out);
            println!("after a filter {}", out);
        });

        out = self.get_activation_node().get_activation_func()(&out, (self.get_weights(), self.get_biases()));
        
        filters[1].iter().for_each(|node| {
            println!("before b filter {}", out);
            out = node.filter(&out);
            println!("after b filter {}", out);
        });
        out
    }

    #[inline]
    // get_output_n returns the n shape of the output matrix
    fn get_output_n(&self) -> usize {
        let mut res = self.get_n();
        self.get_filters()[1].iter().for_each(|f| res += f.get_n_diff());
        res
    }
    
    // get_weights returns the weights for a specific node_index (nth clone of a node).
    fn get_weights(&self) -> &Weights;
    
    // get_bias return the biases for a specific node_index (nth clone of a node) (nth clone of a node).
    fn get_biases(&self) -> &Biases;
    
    // get_n returns the amount of columns of weights (Array1<f64>) and bias (f64).
    // Is used to define the 2nd dimension of weights' shape, and first dim of bias' shape.
    // eg: (pl_n, n), pl_n = 'n' of previous layer, n = 'n' of current layer
    fn get_n(&self) -> usize;
    
    // get_nodes returns an array, of size 'n', of nodes, that will compute,
    // by order, input and provide an output.
    fn get_filters(&self) -> [&Vec<Box<dyn Filter>>; 2];

    fn get_activation_node(&self) -> &Box<dyn Node>;
}

// compute_weights_m will determine the "m" dimension of a (m, n) shaped weights Array2
// using the previous layer's output "n" dimension, and filters_pre modified "n" dimension diff
fn compute_weights_m(filters: &Vec<Box<dyn Filter>>, previous_n: usize) -> usize {
    let mut res = previous_n;
    filters.iter().for_each(|f| res += f.get_n_diff());
    res
}

pub struct Dense<const N: usize> {
    filters_pre: Vec<Box<dyn Filter>>,
    node: Box<dyn Node>,
    filters_post: Vec<Box<dyn Filter>>,
    weights: Weights,
    biases: Biases,
}

impl<const N: usize> Dense<N> {
    // new creates a new Dense struct
    // filters_pre: filters that are gonna be used before activation function
    // node: activation node, requires weights and biases to use
    // filters_post: filters that are gonna be used before activation function
    // wbi: weights and biases initializer
    // previous_n: N shape of the previous layer
    pub fn new(
        filters_pre: Vec<Box<dyn Filter>>,
        node: Box<dyn Node>,
        filters_post: Vec<Box<dyn Filter>>,
        wbi: Box<dyn WeightsBiasesInitializer>,
        previous_n: usize
    ) -> Self {
        let wb = wbi.init(N, compute_weights_m(&filters_pre, previous_n)).unwrap();
        Dense {
            filters_pre,
            node,
            filters_post,
            weights: wb.0,
            biases: wb.1,
        }
    }
}

impl<const N: usize> Layer for Dense<N> {
    fn get_weights(&self) -> &Weights {
        &self.weights
    }

    fn get_biases(&self) -> &Biases {
        &self.biases
    }

    fn get_filters(&self) -> [&Vec<Box<dyn Filter>>; 2] {
        [
            &self.filters_pre,
            &self.filters_post
        ]
    }

    fn get_n(&self) -> usize {
        N
    }

    fn get_activation_node(&self) -> &Box<dyn Node> {
        &self.node
    }
    
}

#[macro_export]
macro_rules! layers {
    ( $( $layer_builder:expr), *) => {
        {
            let mut v = Vec::<Box<dyn Fn(usize) -> Box<dyn $crate::hidden_layer::Layer>>>::new();
            $(
                v.push(Box::new($layer_builder));
            )*
            v
        }
    };
}

#[macro_export]
macro_rules! dense {
    ($n:expr, $pre:expr, $node:expr, $post:expr, $wbinit:expr) => {
        {
            |prev_n: usize| -> Box<dyn Layer> {
                Box::new($crate::hidden_layer::Dense::<{ $n.0 }>::new($pre, $node, $post, $wbinit, prev_n))
            }
        }
    };
    ($n:expr, $node:expr, $wbinit:expr) => {
        {
            dense!($n, vec![], $node, vec![], $wbinit)
        }
    };
}