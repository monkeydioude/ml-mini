pub mod node;
pub mod hidden_layer;
pub mod array;
pub mod io_filter;

use hidden_layer::{Weights, Bias, IO};
use io_filter::DryFilter;
use ndarray::array;

use crate::{hidden_layer::{HiddenLayer, Layer}};

struct DummyNodes;

impl DummyNodes {
    pub fn mul_by_2_filter() -> Box<DryFilter<0>> {
        Box::new(DryFilter {
            filter_f: Box::new(|input: &IO| -> IO {
                input.clone() * 2.0
            })
        })
    }
    pub fn new<const NDIFF: usize>(f: fn(&IO) -> IO) -> Box<DryFilter<NDIFF>> {
        Box::new(DryFilter {
            filter_f: Box::new(f),
        })
    }
}


fn main() {
    let mut hl =  HiddenLayer::<10>::new(vec![
        DummyNodes::mul_by_2_filter(),  
        DummyNodes::new::<0>(|io: &IO| -> IO {
            io.clone() + 1.
        }),
    ], vec![
        DummyNodes::new::<0>(|io: &IO| -> IO {
            io.clone() - 1.
        }),
        DummyNodes::mul_by_2_filter(),
    ]);

    // println!("a1.shape() {:?}, dot prod test {:?}", array![[1, 2, 3], [4, 5, 6]].shape(), array![[1, 2, 3], [4, 5, 6]].dot(&array![[6, 3], [5, 2], [4, 1]]));

    let input = array![[1., 4.]];
    // hl.init_weights_rand(input.shape()[1]);
    // hl.init_weights_zeros(input.shape()[1]);
    hl.init_weights_value(input.shape()[1], 1.0);

    // matrices pre-check
    let out =  hl.run(input.clone());
    println!("input.shape {:?}, weights.shape {:?}, output.shape {:?} =>\nresult {:?}", input.shape(), hl.get_weights().shape(), out.shape(), out);
}