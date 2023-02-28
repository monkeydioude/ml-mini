use crate::hidden_layer::A;

pub type InputLayer = fn(A) -> Result<A, String>;

pub fn color_normalizer(input: A) -> Result<A, String> {
    Ok(input / 255.0)
}