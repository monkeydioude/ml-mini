pub mod macros;
pub mod model;
pub mod layer;
pub mod node_io;
pub mod node;
pub mod node_factory;
pub mod activation;
pub mod train_params;
mod input_filters;

use node::{mul_by};

use crate::{activation::med, node_io::IO, input_filters::{f_to_io, do_nothing}};

fn main() {
    let res = model!(
        input_layer!(do_nothing, f_to_io(3.0)),
        hidden!(
            layer!(
                10,
                mul_by(0.2),
                mul_by(0.5)
            ),
            layer!(
                1,
                mul_by(0.5)
            )
        ),
        output_layer!(med)
    )
    .train(0.5, IO::F64(3.0))
    .run();

    println!("{}", res.unwrap());
}
