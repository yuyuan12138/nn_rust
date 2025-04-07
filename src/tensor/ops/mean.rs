use crate::tensor::operation::Operation;
use super::super::{Tensor, TensorValue};

impl Tensor {
    pub fn mean(&self) -> Tensor {
        let data = self.data.borrow();
        let result_value = match &data.value {
            TensorValue::Vector1D(v) => {
                // Mean for vec
                let sum: f64 = v.iter().sum();
                let count = v.len();

                TensorValue::Scalar(
                    sum / count as f64
                )
            }

            TensorValue::Matrix2D(m ) => {
                if m.is_empty() || m[0].is_empty() {
                    panic!("Matrix is empty");
                }


                let cols = m[0].len();
                assert!(
                    m.iter().all(|row| row.len() == cols),
                    "Inconsistent matrix columns"
                );


                let mut sum = 0.0;
                let mut count = 0;
                for row in m {
                    for &val in row {
                        sum += val;
                        count += 1;
                    }
                }

                // 返回标量均值
                TensorValue::Scalar(sum / count as f64)

            }
            _ => panic!("mean only supported for 1D Vectors"),
        };
        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Mean;
            res_data.dependencies = vec![self.clone()];
        }

        result
    }
}

pub fn backward(tensor: &Tensor){
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;
    if dependencies.len() != 1 {
        panic!("Mean operation requires exactly 1 dependency");
    }

    let input = &dependencies[0];

    let input_shape = {
        let input_data = input.data.borrow();
        input_data.value.shape()
    };

    let grad = match &data.grad {
        TensorValue::Scalar(g) => *g,
        _ => panic!("Mean gradient must be a scalar!"),
    };

    let num_elements = input_shape.iter().product::<usize>() as f64;
    let grad_per_elements = grad / num_elements;

    let grad_tensor = match input_shape.len() {
        0 => {
            TensorValue::Scalar(grad_per_elements)
        }
        1 => {
            TensorValue::Vector1D(vec![grad_per_elements; input_shape[0]])
        }
        2 => {
            TensorValue::Matrix2D(vec![vec![grad_per_elements; input_shape[1]]; input_shape[0]])
        }
        _ => panic!("Unsupported dimension for mean backward!")
    };

    input.data.borrow_mut().add_grad(grad_tensor);
}