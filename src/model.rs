use ndarray::Array2;

use crate::{input_layer::InputLayer, hidden_layer::{Layer, A}, loss::Loss, formula::cost};

pub struct Model<const N0: usize> {
    input_layer: Option<InputLayer>,
    hidden_layers: Vec<Box<dyn Layer>>,
    loss_method: Box<dyn Loss>,
    // output_layer: PhantomData<usize>,
}

pub struct FeaturesAmount(pub usize);

// N0 = feature amount
impl<const N0: usize> Model<N0> {
    pub fn new(
        input_layer: Option<InputLayer>,
        hidden_layers: Vec<Box<dyn Layer>>,
        loss_method: Box<dyn Loss>
    ) -> Model<N0> {
        Model {
            input_layer,
            hidden_layers,
            loss_method,
            // output_layer
        }
    }

    pub fn run(&self, input: A) -> Result<A, String> {
        if input.shape()[0] != N0 {
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

    pub fn train(&mut self, input: A, ys: Array2<f64>) -> Result<A, String> {
        if input.shape()[1] != ys.shape()[1] {
            return Err(format!("input and y matrices shapes did not match: {} != {}", input.shape()[1], ys.shape()[1]))
        }

        let y_hats = match self.run(input) {
            Ok(res) => res,
            Err(msg) => return Err(msg),
        };

        if y_hats.shape() != ys.shape() { 
            return Err("y and y_hat matrices shapes did not match".to_string())
        }

        let mut losses = ys.clone(); 
        // let s = y_hats.shape();
        // let x = (y.0 / s[1]);
        // let losses = ys.iter().enumerate().map(|y| {
        //     let sx = y.0 / s[1];
        //     let sy = y.0 % s[0];
        //     self.loss_method.loss(y.1, &y_hats.get((sx, sy)).unwrap())
        // });

        for (pos, v) in ys.indexed_iter() {
            losses[pos] = self.loss_method.loss(v, &y_hats.get(pos).unwrap());
        }

        println!("y_hats: {:?}, losses: {:?}", &y_hats, &losses);

        // let cost = cost(losses);

        // y_hats.
        // find method or modulo over index to compute [x, y] coords?
        // let losses = ys.iter().enumerate().map(|y| loss(y.1, &y_hats[y.0]));
        // let cost = 
        // self.hidden_layers = vec![];
        Ok(y_hats)
    }

}

pub fn build_hidden_layers(previous_n: usize, hlsfn: Vec<Box<dyn Fn(usize) -> Box<dyn Layer>>>) -> Vec<Box<dyn Layer>> {
    // make hidden_layers vector
    let mut hidden_layers = Vec::<Box<dyn Layer>>::new();
    // init previous layer mutable param
    let mut pn = previous_n;

    // iterate over hidden_layers_functions ($hlsfn) with the goal
    // of automatically computing previous_n, necessary to
    // layer creation.
    // hlfns is a vector of builder callbacks.
    hlsfn.iter().for_each(|hlfn| {
        // Layer making, using mutable var pn as previous_n value
        let tmpl = hlfn(pn);
        // fetching this layer's post filter so we can compute next layer's pn
        let post_filter = (&tmpl).get_filters()[1];
        // set pn as this filter's N
        pn = (&tmpl).get_n();
        // computing pn using each filter's n diff value
        (&post_filter).iter().for_each(|filter| pn += filter.get_n_diff());
        // after all those borrows, finally pushing the layer into the vector
        hidden_layers.push(tmpl);
    });

    hidden_layers
}

#[macro_export]
macro_rules! model {
    ( $n:expr, $il:expr, $hlsfn:expr, $lm:expr ) => {
        {
            $crate::model::Model::<{ $n.0 }>::new(Some($il), $crate::model::build_hidden_layers($n.0, $hlsfn), $lm)
        }
    };
    ( $n:expr, $hlsfn:expr/*, $ol:expr*/ ) => {
        {
            $crate::model::Model::<{ $n.0 }>::new(None, $crate::model::build_hidden_layers($n.0, $hlsfn), $crate::loss::basic_loss())
        }
    };
}