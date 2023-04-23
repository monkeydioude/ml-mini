pub mod node;
pub mod hidden_layer;
pub mod array;
pub mod io_filter;
pub mod model;
pub mod input_layer;
pub mod output_layer;
pub mod wb;
pub mod formula;
pub mod loss;

use hidden_layer::A;
use io_filter::DryFilter;
use ndarray::array;

use crate::{hidden_layer::{Layer, Units}, input_layer::color_normalizer, wb::value_init, node::sigmoid, model::FeaturesAmount, loss::basic_loss};

struct DummyNodes;

impl DummyNodes {
    pub fn mul_by_2_filter() -> Box<DryFilter<0>> {
        Box::new(DryFilter {
            filter_f: Box::new(|input: &A| -> A {
                input.clone() * 2.0
            })
        })
    }
    pub fn new<const NDIFF: usize>(f: fn(&A) -> A) -> Box<DryFilter<NDIFF>> {
        Box::new(DryFilter {
            filter_f: Box::new(f),
        })
    }
}

fn main() {
    let input = array![[255.], [255.]];
    // let hl = Dense::<1>::new(vec![
    //     // DummyNodes::mul_by_2_filter(),
    //     DummyNodes::new::<0>(|io: &A| -> A {
    //         -io.clone()
    //     }),
    // ], sigmoid(),
    // vec![],
    // // zeros_init(), 
    // value_init(1.),
    // input.shape()[0]
    // );

    // hl.init_weights_value(input.shape()[0], 0.04);
    // let hl_weights = (&hl).get_weights().clone();
    let mut model = model!(
        FeaturesAmount(2),
        color_normalizer,
        layers!(
            dense!(
                Units(3),
                sigmoid(),
                value_init(1.)
            ),
            dense!(
                Units(2),
                sigmoid(),
                value_init(1.)
            ),
            dense!(
                Units(1),
                sigmoid(),
                value_init(1.)
            )
        ),
        basic_loss()
    );

    // let out = model.run(input.clone()).unwrap();
    let out = model.train(input.clone(), array![[1.]]).unwrap();


    // println!("weights.shape {:?}, input.shape {:?}, output.shape {:?}\noutput: {:?}", hl_weights.shape(), input.shape(), out.shape(), out);
    // println!("{:?}, {:?}", array![[1., 1., 1.]], array![[2., 2., 2., 2.]].t().dot(&array![[1., 2.]]) + 1.0);
    // println!("{:?}", array![[1., 2.]].t());
    println!("input.shape {:?}, output.shape {:?}\noutput: {:?}", input.shape(), out.shape(), out);
}