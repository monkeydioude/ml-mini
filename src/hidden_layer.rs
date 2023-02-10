use ndarray::{Array2, Array1, array};

use crate::{array::Array2Random, io_filter::Filter};

pub type Weights = Array2<f64>;
pub type Bias = Array1<f64>;
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
    #[inline]
    fn run(&self, input: IO) -> IO {
        let mut out = input;
        let filters = self.get_filters();

        filters[0].iter().for_each(|node| {
            println!("before a filter {}", out);
            out = node.filter(&out);
            println!("after a filter {}", out);
        });

        out = out.dot(self.get_weights()) + self.get_bias();
        
        filters[1].iter().for_each(|node| {
            println!("before b filter {}", out);
            out = node.filter(&out);
            println!("after b filter {}", out);
        });
        out
    }

    #[inline]
    // get_output_n returns the amount of columns of the output vector
    fn get_output_n(&self) -> usize {
        let mut res = self.get_n();
        self.get_filters()[1].iter().for_each(|f| {
            res += f.get_n_diff()
        });
        res
    }
    
    // get_weights returns the weights for a specific node_index (nth clone of a node).
    fn get_weights(&self) -> &Weights;
    
    // get_bias return the biases for a specific node_index (nth clone of a node) (nth clone of a node).
    fn get_bias(&self) -> &Bias;
    
    // get_n returns the amount of columns of weights (Array1<f64>) and bias (f64).
    // Is used to define the 2nd dimension of weights' shape, and first dim of bias' shape.
    // eg: (pl_n, n), pl_n = 'n' of previous layer, n = 'n' of current layer
    fn get_n(&self) -> usize {
        N
    }
    
    // get_nodes returns an array, of size 'n', of nodes, that will compute,
    // by order, input and provide an output.
    fn get_filters(&self) -> [&Vec<Box<dyn Filter>>; 2];

    // compute_weights_m will determine the "m" dimension of a (m, n) shaped weights Array2
    // using the previous layer's output "n" dimension, and filters_pre modified "n" dimension diff
    fn compute_weights_m(&self, previous_n: usize) -> usize {
        let mut res = previous_n;
        self.get_filters()[0].iter().for_each(|f| res += f.get_n_diff());
        res
    }
}

pub struct HiddenLayer<const N: usize> {
    filters_pre: Vec<Box<dyn Filter>>,
    filters_post: Vec<Box<dyn Filter>>,
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl<const N: usize> HiddenLayer<N> {
    // new creates a new HiddenLayer struct
    // filters_pre: filters that are gonna be used before activation function
    // filters_post: filters that are gonna be used before activation function
    pub fn new(filters_pre: Vec<Box<dyn Filter>>, filters_post: Vec<Box<dyn Filter>>) -> Self {
        HiddenLayer {
            filters_pre,
            filters_post,
            weights: array![[]],
            biases: array![],
        }
    }

    pub fn init_weights_rand(&mut self, previous_n: usize) {
        self.weights = Array2Random::<f64>::fill((self.compute_weights_m(previous_n), N));
        self.biases = Array1::zeros(N);
    }

    pub fn init_weights_zeros(&mut self, previous_n: usize) {
        self.weights = Array2::zeros((self.compute_weights_m(previous_n), N));
        self.biases = Array1::zeros(N);
    }

    pub fn init_weights_value(&mut self, previous_n: usize, value: f64) {
        self.init_weights_zeros(previous_n);
        self.weights.fill(value);
        self.biases.fill(value);
    }
}

impl<const N: usize> Layer<N> for HiddenLayer<N> {
    fn get_weights(&self) -> &Weights {
        &self.weights
    }

    fn get_bias(&self) -> &Bias {
        &self.biases
    }

    fn get_filters(&self) -> [&Vec<Box<dyn Filter>>; 2] {
        [
            &self.filters_pre,
            &self.filters_post
        ]
    }
}
