pub mod node;
pub mod layer;
pub mod layer_builder;

use layer::{Weights, Bias, IO};
use ndarray::array;
use node::DryNode;

use crate::layer::{HiddenLayer, Layer};


fn test(input: &IO, _: Option<(Weights, Bias)>) -> IO {
    input.clone() * 2.0
}

fn main() {
   let hl =  HiddenLayer::<10, _>::new(vec![Box::new(
        DryNode{
            a_fn: test
        }
    )]);

    println!("{}", hl.run(array![[1., 2., 3.], [4., 5., 6.]]));
}