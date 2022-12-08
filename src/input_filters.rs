use ndarray::{Array1, array};

pub type Filter = dyn Fn(Array1<f64>) -> Array1<f64>;

pub fn do_nothing(inp: Array1<f64>) -> Array1<f64> {
    inp
}

pub fn f_to_io(v: f64) -> Box<Filter> {
    Box::new(move |_: Array1<f64>| -> Array1<f64> {
        array![v.clone()]
    })
}