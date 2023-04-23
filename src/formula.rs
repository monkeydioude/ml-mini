use ndarray::Array2;

pub fn sigmoid(z: &f64) -> f64 {
    1.0/(1.0+ (-z).exp())
}

pub fn basic_loss(y: &f64, y_hat: &f64) -> f64 {
    -((y * y_hat.ln()) + ((1. - y) * (1. - y_hat).ln()))
}

pub fn cost(losses: &Array2<f64>) -> f64 {
     (1. /(losses.len() as f64)) * (losses.sum() as f64)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_sigmoid() {
        for t in &[
            (3.0, 0.9525741268224334),
            (90.0, 1.0),
            (0.0001, 0.5000249999999792),
            (-7.5, 0.0005527786369235996),
        ] { 
            assert_eq!(sigmoid(&t.0), t.1);
        }
    }

    #[test]
    fn test_basic_loss() {
        for t in &[
            (1., 0.8, 0.2231435513142097),
            (0., 0.8, 1.6094379124341005),
            (0., 0.999, 6.907755278982136),
            (1., 0.999, 0.0010005003335835344),
            (1., 0.9506703128348142, 0.05058795074317592),
        ] {
            assert_eq!(basic_loss(&t.0, &t.1), t.2);
        }
    }
 
    #[test]
    fn test_cost() {
        for t in &[
            (array![[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]], 2.0),
            (array![[4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0, 4.0, 2.0]], 3.0),
            (array![[5.3, 9.2, 0.4, 0.0002, 3.3234, 20.212909884783, 7., 8.00009, 1.99999, 14.1414114]], 6.957800128478299),
        ] {
            assert_eq!(cost(&t.0), t.1);
        }
    }
}