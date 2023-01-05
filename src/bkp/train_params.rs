use crate::node_io::IO;

// TrainParams is basically a parameters bag
pub struct TrainParams {
    pub learning_rate: f64,
    // y should be of the same IO type as y hat
    pub y: IO,
}

impl TrainParams {
    pub fn new(learning_rate: f64, y: IO) -> TrainParams {
        TrainParams { learning_rate, y }
    }
}