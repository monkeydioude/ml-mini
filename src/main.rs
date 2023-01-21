pub mod node;
pub mod layer;
pub mod array;

use layer::{Weights, Bias, IO};
use ndarray::array;

use crate::{layer::{HiddenLayer, Layer}, node::{DryNode}};

fn mul_by_2_node() -> Box<DryNode> {
    Box::new(DryNode {
        a_fn: Box::new(|input: &IO, _: Option<(Weights, Bias)>| -> IO {
            input.clone() * 2.0
        })
    })
}

fn main() {
    let mut hl =  HiddenLayer::<10>::new(vec![
        mul_by_2_node(),
        mul_by_2_node()
    ]);

        // hl.set_weights_biases(w, b)
    
    println!("{}", hl.run(array![[1., 2., 3.], [4., 5., 6.]]));
}