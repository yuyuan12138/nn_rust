use crate::tensor::Tensor;
use super::Optimizer;

pub struct SGD {
    lr: f64
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn step(&self, params: &[&Tensor]) {
        for param in params {
            let mut data = param.data.borrow_mut();
            data.value -= self.lr * data.grad;
        }
    }

    fn zero_grad(&self, params: &[&Tensor]) {
        for param in params {
            let mut data = param.data.borrow_mut();
            data.grad = 0.0;
        }
    }
}