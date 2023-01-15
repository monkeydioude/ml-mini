pub mod node;
pub mod layer;

use layer::{Weights, Bias, IO};
use ndarray::array;

use crate::{layer::{HiddenLayer, Layer}, node::{DryNode}};


fn test(input: &IO, _: Option<(Weights, Bias)>) -> IO {
    input.clone() * 2.0
}

fn main() {
    let hl =  HiddenLayer::<10>::new(vec![
        Box::new(DryNode {
            a_fn: Box::new(test)
        })
    ]);
    
    println!("{}", hl.run(array![[1., 2., 3.], [4., 5., 6.]]));
}