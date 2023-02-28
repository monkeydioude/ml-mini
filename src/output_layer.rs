// use std::f64::consts::E;

// use ndarray::Array1;

// use crate::hidden_layer::A;

// pub trait Activation<T> {
//     fn activate(&self, input: A) -> Array1<T>;
// }

// pub trait Derive {
//     // fn derive(&self, output: Array2<f64>) -> f64;
// }

// pub trait Cost<T> {
//     fn loss(&self, y: Array1<T>, y_hat: Array1<T>) -> f64;
//     fn cost(&self, m: usize) -> f64;
// }

// pub trait OutputLayer<T>: Derive + Activation<T> {}

// pub struct BinaryClassifier;

// impl<T> Activation<T> for BinaryClassifier {
//     fn activate(&self, z: A) -> Array1<T> {
//         // 1.0/(1.0+ (-z).exp())
//     }
// }