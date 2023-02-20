use crate::{input_layer::InputLayer, hidden_layer::{Layer, IO}};

pub struct Model<const FA: usize> {
    input_layer: Option<InputLayer>,
    hidden_layers: Vec<Box<dyn Layer>>,
    // output_layer: PhantomData<usize>,
}

// FA = feature amount
impl<const FA: usize> Model<FA> {
    pub fn new(
        input_layer: Option<InputLayer>,
        hidden_layers: Vec<Box<dyn Layer>>,
    ) -> Model<FA> {
        Model {
            input_layer,
            hidden_layers,
            // output_layer
        }
    }

    pub fn run(&self, input: IO) -> Result<IO, String> {
        if input.shape()[1] != FA {
            return Err("unexpected amount of features from input".to_string());
        }
        let mut io = input;

        io = match self.input_layer {
            Some(il) => (il)(io)?,
            None => io,
        };
        // io = (self.input_layer.unwrap())(io)?;
        self.hidden_layers.iter().for_each(|hl| {
            io = hl.run(io.clone());
        });
        Ok(io)
    }
}

#[macro_export]
macro_rules! model {
    ( $n:expr, $il:expr, $hls:expr/*, $ol:expr*/ ) => {
        {
            $crate::model::Model::<$n>::new(Some($il), $hls)
        }
    };
    ( $n:expr, $hls:expr/*, $ol:expr*/ ) => {
        {
            $crate::model::Model::<$n>::new(None, $hls)
        }
    };
}