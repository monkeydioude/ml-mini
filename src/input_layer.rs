use crate::hidden_layer::IO;

pub type InputLayer = fn(IO) -> Result<IO, String>;

pub fn color_normalizer(input: IO) -> Result<IO, String> {
    Ok(input / 255.0)
}