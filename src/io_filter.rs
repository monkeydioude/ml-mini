use crate::hidden_layer::A;

pub trait Filter {
    fn filter(&self, input: &A) -> A;
    fn get_n_diff(&self) -> usize;
}

pub struct DryFilter<const NDIFF: usize> {
    pub filter_f: Box<dyn Fn(&A) -> A>,
}

impl <const NDIFF: usize> Filter for DryFilter<NDIFF> {
    fn filter(&self, input: &A) -> A {
        (self.filter_f)(input)
    }

    fn get_n_diff(&self) -> usize {
        NDIFF
    }
}