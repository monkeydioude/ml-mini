use crate::hidden_layer::IO;

pub trait Filter {
    fn filter(&self, input: &IO) -> IO;
    fn get_n_diff(&self) -> usize;
}

pub struct DryFilter<const NDIFF: usize> {
    pub filter_f: Box<dyn Fn(&IO) -> IO>,
}

impl <const NDIFF: usize> Filter for DryFilter<NDIFF> {
    fn filter(&self, input: &IO) -> IO {
        (self.filter_f)(input)
    }

    fn get_n_diff(&self) -> usize {
        NDIFF
    }
}