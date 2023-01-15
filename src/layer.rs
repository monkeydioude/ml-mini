use ndarray::{Array2, Array1};

use crate::node::Node;

pub type Weights = Array1<f64>;
pub type Bias = f64;
pub type IO = Array2<f64>;

pub fn safe_row<T: Clone>(arr: &Array2<T>, row: usize) -> Option<Array1<T>> {
    if row >= arr.shape().len() {
        return None;
    }
    Some(arr.row(row).to_owned())
}

pub trait Layer<const N: usize> {
    // run takes an Array2<f64> as input, which will be used by each node
    // to compute a result,using weights and biases.
    // ex:  input.shape() = (1, 5)
    //      previous_layer.get_n() = 5
    //      layer.get_n() = 4
    //      weights.shape() = (5, 4)
    //      biases.shape() = (4)
    //      -> output.shape() = (1, 4)
    fn run(&self, input: IO) -> IO;
    
    // get_weights returns the weights for a specific node_index (nth clone of a node).
    fn get_weights(&self, node_index: usize) -> Weights;
    
    // get_bias return the biases for a specific node_index (nth clone of a node) (nth clone of a node).
    fn get_bias(&self, node_index: usize) -> &Bias;
    
    // get_n returns how many times a node will be cloned.
    // Is used to define the 2nd dimension of weights' shape, and first dim of bias' shape.
    // 'n' could be seen at nth node clone. eg: When iterating over 'n', 0 = 1st clone, 1 = 2nd clone etc...
    // eg: (pl_n, n), pl_n = 'n' of previous layer, n = 'n' of current layer
    fn get_n(&self) -> usize;
    
    // get_nodes returns an array, of size 'n', of nodes, that will compute,
    // by order, input and provide an output.
    fn get_nodes(&self) -> &Vec<Box<dyn Node>>;

    fn with_nodes(nodes: Vec<Box<dyn Node>>, previous_n: usize) -> Self;
}

pub struct HiddenLayer<const N: usize> {
    nodes: Vec<Box<dyn Node>>,
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl<const N: usize> HiddenLayer<N> {
    pub fn new(nodes: Vec<Box<dyn Node>>) -> Self {
        let nlen = nodes.len();
        HiddenLayer {
            nodes,
            weights: Array2::zeros((nlen, N)),
            biases: Array1::zeros(N),
        }
    }
}

impl<const N: usize> Layer<N> for HiddenLayer<N> {
    fn run(&self, input: IO) -> IO {
        let mut out = input;
        self.nodes.iter().for_each(|node| {
            out = node.get_activation_func()(&out, None);
        });
        out
    }

    fn get_weights(&self, node_index: usize) -> Weights {
        match safe_row(&self.weights, node_index) {
            Some(w) => w,
            None => panic!("invalid node_index {}", node_index), 
        }
    }

    fn get_bias(&self, node_index: usize) -> &Bias {
        match self.biases.get(node_index) {
            Some(b) => b,
            None => panic!("no bias at index {}", node_index),
        }
    }

    fn get_n(&self) -> usize {
        N
    }

    fn get_nodes(&self) -> &Vec<Box<dyn Node>> {
        &self.nodes
    }

    fn with_nodes(nodes: Vec<Box<dyn Node>>, previous_n: usize) -> Self {
        HiddenLayer {
            nodes,
            weights: Array2::zeros((previous_n, N)),
            biases: Array1::zeros(previous_n)
        }
    }

}
