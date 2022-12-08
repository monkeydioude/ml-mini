use crate::node_io::IO;

pub struct TrainParams {
    pub learning_rate: f64,
    pub y: IO,
}

impl TrainParams {
    pub fn new(learning_rate: f64, y: IO) -> TrainParams {
        TrainParams { learning_rate, y }
    }
}