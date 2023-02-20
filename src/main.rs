pub mod node;
pub mod hidden_layer;
pub mod array;
pub mod io_filter;
pub mod model;
pub mod input_layer;

use hidden_layer::{Weights, Bias, IO};
use io_filter::DryFilter;
use ndarray::array;

use crate::{hidden_layer::{HiddenLayer, Layer}, input_layer::color_normalizer};

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
        // DummyNodes::mul_by_2_filter(),
        // DummyNodes::new::<0>(|io: &IO| -> IO {
        //     io.clone() + 1.
        // }),
    ], vec![
        // DummyNodes::new::<0>(|io: &IO| -> IO {
        //     io.clone() - 1.
        // }),
        // DummyNodes::mul_by_2_filter(),
    ]);

    let input = array![[244., 12.], [3., 66.], [212., 42.]];
    hl.init_weights_value(input.shape()[1], 0.04);
    let hl_weights = (&hl).get_weights().clone();
    // let model = Model::<2>::new(Some(color_normalizer), vec![Box::new(hl)]);
    // let model = model!(2, vec![Box::new(hl)]);
    let model = model!(
        2,
        color_normalizer,
        layers!(hl)
    );

    let out = model.run(input.clone()).unwrap();

    println!("input.shape {:?}, weights.shape {:?}, output.shape {:?} =>\nresult {:?}", input.shape(), hl_weights.shape(), out.shape(), out);
}