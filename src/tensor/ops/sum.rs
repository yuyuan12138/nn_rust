use crate::tensor::operation::Operation;
use super::super::{Tensor, TensorValue};
use anyhow::Result;
impl Tensor {
    pub fn sum(&self) -> Result<Tensor> {
        let data = self.data.borrow();
        let results_value = match &data.value {
            TensorValue::Scalar(s) => {
                TensorValue::Scalar(s.clone())
            }

            TensorValue::Vector1D(v) => {
                let sum: f64 = v.iter().sum();

                TensorValue::Scalar(
                    sum as f64
                )
            }

            TensorValue::Matrix2D(m) => {
                if m.is_empty() || m[0].is_empty() {
                    panic!("Matrix is empty");
                }


                let cols = m[0].len();
                assert!(
                    m.iter().all(|row| row.len() == cols),
                    "Inconsistent matrix columns"
                );

                let mut sum = 0.0;
                for row in m {
                    for &val in row {
                        sum += val;
                    }
                }
                TensorValue::Scalar(sum)
            }
            _ => panic!("sum only supported for 1D / 2D"),
        };

        let result = Self::from_value(results_value);

        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Sum;
            res_data.dependencies = vec![self.clone()];
        }

        Ok(result)
    }
}

pub fn backward(tensor: &Tensor) -> Result<()>{
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;
    if dependencies.len() != 1 {
        panic!("Sum operation requires exactly 1 dependency");
    }

    let input = &dependencies[0];

    let input_shape = {
        let input_data = input.data.borrow();
        input_data.value.shape()
    };

    let grad = match &data.grad {
        TensorValue::Scalar(g) => *g,
        _ => panic!("Sum gradient must be a scalar!"),
    };

    let grad_per_element = grad;

    let grad_tensor = match input_shape.len() {
        0 => TensorValue::Scalar(grad_per_element),
        1 => TensorValue::Vector1D(vec![grad_per_element; input_shape[0]]),
        2 => TensorValue::Matrix2D(vec![vec![grad_per_element; input_shape[1]]; input_shape[0]]),
        _ => panic!("Unsupported dimension for sum backward!"),
    };

    input.data.borrow_mut().add_grad(grad_tensor)?;
    Ok(())

}