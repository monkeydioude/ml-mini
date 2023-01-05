use std::fmt::Display;

use ndarray::{Array1, Array2};

#[derive(Debug, Clone, PartialEq)]
pub enum IO {
    Array1(Array1<f64>),
    Array2(Array2<f64>),
    F64(f64),
    Null
}

impl IO {
    pub fn is_f64(&self) -> bool {
        if let IO::F64(_) = self {
            return true;
        }
        false
    }

    pub fn f64(&self) -> Result<f64, String> {
        if let IO::F64(v) = self {
            return Ok(*v);
        }
        Err("Could not unwrap to f64".to_string())
    }
    
    pub fn null(&self) -> Result<(), String> {
        if let IO::Null = self {
            return Ok(())
        }
        Err("Could not unwrap to null".to_string())
    }

    pub fn type_eq(&self, against: IO) -> bool {
        match (self, against) {
            (IO::Array1(_), IO::Array1(_)) => true,
            (IO::Array2(_), IO::Array2(_)) => true,
            (IO::F64(_), IO::F64(_)) => true,
            (IO::Null, IO::Null) => true,
            (_, _) => false
        }
    }
}

impl Display for IO {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IO::Array1(v) => write!(f, "{}", v),
            IO::Array2(v) => write!(f, "{}", v),
            IO::F64(v) => write!(f, "{}", *v),
            IO::Null => write!(f, "null")
        }
    }
}

impl std::ops::Mul<IO> for Array2<f64> {
    type Output = Array2<f64>;

    fn mul(self, rhs: IO) -> Self::Output {
        match rhs {
            IO::F64(v) => self * v,
            IO::Array2(v) => self * v,
            _ => panic!("Can not multiply {} by {}", self, rhs)
        }
    }
}

impl std::ops::Mul<IO> for Array1<f64> {
    type Output = Array1<f64>;

    fn mul(self, rhs: IO) -> Self::Output {
        match rhs {
            IO::F64(v) => self * v,
            IO::Array1(v) => self * v,
            _ => panic!("Can not multiply {} by {}", self, rhs)
        }
    }
}

impl std::ops::Mul<Array2<f64>> for IO {
    type Output = Array2<f64>;

    fn mul(self, rhs: Array2<f64>) -> Self::Output {
        match self {
            IO::F64(v) => v * rhs,
            IO::Array2(v) => v * rhs,
            _ => panic!("Can not multiply {} by {}", self, rhs)
        }
    }
}

impl std::ops::Mul<Array1<f64>> for IO {
    type Output = Array1<f64>;

    fn mul(self, rhs: Array1<f64>) -> Self::Output {
        match self {
            IO::F64(v) => v * rhs,
            IO::Array1(v) => v * rhs,
            _ => panic!("Can not multiply {} by {}", self, rhs)
        }
    }
}