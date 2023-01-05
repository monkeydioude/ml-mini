pub mod node;

use ndarray::{Array2, Array1};
use node::{Node, DryNode};

pub type Weights = Array1<f64>;
pub type Bias = f64;
pub type IO = Array2<f64>;

trait Layer<const N: usize> {
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
  fn get_weights(&self, node_index: usize) -> &Weights;

  // get_bias return the biases for a specific node_index (nth clone of a node) (nth clone of a node).
  fn get_bias(&self, node_index: usize) -> &Bias;

  // get_n returns how many times a node will be cloned.
  // Is used to define the 2nd dimension of weights' shape, and first dim of bias' shape.
  // 'n' could be seen at nth node clone. eg: When iterating over 'n', 0 = 1st clone, 1 = 2nd clone etc...
  // eg: (pl_n, n), pl_n = 'n' of previous layer, n = 'n' of current layer
  fn get_n(&self) -> usize;

  // get_nodes returns an array, of size 'n', of nodes, that will compute,
  // by order, input and provide an output.
  fn get_nodes(&self) -> &[Vec<Box<dyn Node>>; N];
}

struct HiddenLayer<const N: usize> {
  nodes: [Vec<Box<dyn Node>>; N],
}

impl<const N: usize> HiddenLayer<N> {
  pub fn new(node_vec: Vec<Box<dyn Node>>) -> Self {
    HiddenLayer {
      nodes: [node_vec.clone(); N],
    }
  }
}

fn test(input: f64) -> f64 {
  input.clone()
}

fn main() {
  HiddenLayer::<10>::new(vec![Box::new(
    DryNode{
      a_fn: Box::new(test)
    } )]);
}