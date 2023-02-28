use crate::hidden_layer::{Weights, Biases};

pub trait WeightsBiasesInitializer {
    fn init(&self, n: usize, m: usize) -> Result<(Weights, Biases), String>;
}

pub struct Zeros;

impl WeightsBiasesInitializer for Zeros {
    fn init(&self, n: usize, m: usize) -> Result<(Weights, Biases), String> {
        Ok((
            Weights::zeros((n, m)),
            Biases::zeros((n, 1))
        ))
    }
}

pub fn zeros_init() -> Box<Zeros> {
    Box::new(Zeros{})
}

pub struct Value(f64);

impl WeightsBiasesInitializer for Value {
    fn init(&self, n: usize, m: usize) -> Result<(Weights, Biases), String> {
        let mut wb = zeros_init().init(n, m).unwrap();
        wb.0.fill(self.0);
        wb.1.fill(self.0);
        Ok(wb)
    }
}

pub fn value_init(value: f64) -> Box<Value> {
    Box::new(Value(value))
}