use ndarray::{Array1, Axis};

pub fn sum(arr: Array1<f64>) -> f64 {
    arr.sum()
}

pub fn med(arr: Array1<f64>) -> f64 {
    let l = (arr.len_of(Axis(0)) * arr.len_of(Axis(1))) as f64;
    println!("l: {}", l);
    sum(arr) / l
}