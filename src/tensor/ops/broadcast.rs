use crate::tensor::operation::Operation;
use super::super::{Tensor, TensorValue};

impl Tensor {
    pub fn broadcast(&self, target_shape: &[usize]) -> Tensor {
        let data = self.data.borrow();
        let result_value = match &data.value {
            // shape
            TensorValue::Scalar(s) => {
                // any shape
                match target_shape.len() {
                    0 => TensorValue::Scalar(*s),
                    1 => TensorValue::Vector1D(vec![*s; target_shape[0]]),
                    2 => TensorValue::Matrix2D(
                        vec![vec![*s; target_shape[1]]; target_shape[0]]
                    ),
                    _ => panic!("Unsupported broadcast dimension!")
                }
            }

            TensorValue::Vector1D(v) if target_shape == &[v.len(), 1] => {
                TensorValue::Matrix2D(v.iter().map(|x| vec![*x]).collect())
            }

            _ => panic!("Unsupported broadcast!")
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Broadcast;
            res_data.dependencies = vec![self.clone()];
        }
        result
    }
}

pub fn backward(tensor: &Tensor) {
    let data = tensor.data.borrow();
    let input = &data.dependencies[0];

    let grad = &data.grad;
    let original_shape = input.data.borrow().value.shape();

    let summed_grad = match grad {
        TensorValue::Matrix2D(m) if original_shape.len() == 1 => {
            // 例如将标量广播到矩阵，梯度求和
            TensorValue::Scalar(m.iter().flat_map(|row| row.iter()).sum())
        }
        TensorValue::Vector1D(v) if original_shape.is_empty() => {
            TensorValue::Scalar(v.iter().sum())
        }
        // TODO: 其他情况...
        _ => grad.clone()
    };

    input.data.borrow_mut().add_grad(summed_grad);
}