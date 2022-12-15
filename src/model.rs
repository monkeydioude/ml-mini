use ndarray::array;

use crate::{node_io::IO, node::{Activation, Output}, layer::{Layer, Hidden, Builder}, train_params::TrainParams, input_filters::Filter};

pub struct Model {
    input_layer: Vec<Box<Filter>>,
    hidden_layers: Vec<Hidden>,
    output_layer: Activation,
    train_params: Option<TrainParams>,
}

// Model's generale use is simple:
// - Every layers and nodes are instantiated automatically inside "new()".
// - "run()" starts the model computation process
// - "train()" runs backprop and update the weights according to params
impl Model {
    pub fn run(&mut self) -> Result<IO, String> {
        let mut output = array![];

        self.input_layer.iter().for_each(|o| {
            output = o(output);
        });

        for l in self.hidden_layers.iter() {
            output = l.run(output);
        }
        
        let y_hat = self.output_layer.predict(output);

        Ok(IO::F64(y_hat))
    }

    pub fn train(&mut self, learning_rate: f64, y: IO) -> &mut Self {
        if let Ok(_) = y.null() {
            panic!("Could not setup trainig: y was Null")
        }
        self.train_params = Some(TrainParams::new(learning_rate, y));
        self
    }

    pub fn new(
        input_layer: Vec<Box<Filter>>,
        hidden_layers: Vec<Hidden>,
        output_layer: Activation,
    ) -> Self {
        let mut n = 1;

        hidden_layers.iter().for_each(|v| {
           match v.build(n) {
            Ok(_n) => {
                n = _n;
            },
            Err(err) => panic!("Could not build model: {}", err),
        }
        });

        Model {
            input_layer,
            hidden_layers,
            output_layer,
            train_params: None,
        }
    }
}