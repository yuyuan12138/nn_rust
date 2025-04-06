use crate::tensor::{Tensor, TensorValue};
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

            // 计算梯度更新量
            let update = match &data.grad {
                TensorValue::Scalar(g) => TensorValue::Scalar(self.lr * g),
                TensorValue::Vector1D(v) => {
                    TensorValue::Vector1D(v.iter().map(|g| self.lr * g).collect())
                }
                TensorValue::Matrix2D(m) => {
                    TensorValue::Matrix2D(
                        m.iter().map(|row|
                            row.iter().map(|g| self.lr * g).collect()
                        ).collect()
                    )
                }
            };

            data.value = data.value.sub(&update);
        }
    }

    fn zero_grad(&self, params: &[&Tensor]) {
        for param in params {
            let mut data = param.data.borrow_mut();
            data.grad = match &data.grad {
                TensorValue::Scalar(_) => TensorValue::Scalar(0.0),
                TensorValue::Vector1D(v) => TensorValue::Vector1D(vec![0.0; v.len()]),
                TensorValue::Matrix2D(m) => {
                    TensorValue::Matrix2D(vec![vec![0.0; m[0].len()]; m.len()])
                }
            };
        }
    }
}