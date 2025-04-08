use crate::tensor::operation::Operation;
use crate::tensor::utils::transpose;
use super::super::{Tensor, TensorValue};

impl Tensor {
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let data = self.data.borrow();

        let result_value = match &data.value {
            TensorValue::Scalar(s) => match dim {
                0 => {
                    TensorValue::Vector1D(vec![*s])
                }

                _ => panic!("Beyond dim")
            }
            TensorValue::Vector1D(v) => match dim {
                0 => {
                    TensorValue::Matrix2D(vec![v.clone()])
                }
                1 => {
                    TensorValue::Matrix2D(transpose(&vec![v.clone()]))
                }
                _ => panic!("Beyond dim")
            }

            _ => panic!("Unsqueeze operation did not match this data format!")
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Unsqueeze(dim);
            res_data.dependencies = vec![self.clone()];
        }
        result
    }

}

pub fn backward(tensor: &Tensor, dim: usize) {
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;
    if dependencies.len() != 1 {
        panic!("Unsqueeze operation requires exactly 1 dependency");
    }

    let x = &dependencies[0];
    let unsqueeze_output = match &data.value {
        TensorValue::Vector1D(v) => TensorValue::Vector1D(v.clone()),
        TensorValue::Matrix2D(m) => TensorValue::Matrix2D(m.clone()),
        _ => panic!("Backward unsqueeze doesn't match")
    };

    match (&data.grad, &unsqueeze_output) {
        (TensorValue::Vector1D(grad), TensorValue::Vector1D(s)) => {
            x.data.borrow_mut().add_grad_scalar(grad[0]);
        }
        (TensorValue::Matrix2D(grad), TensorValue::Matrix2D(s)) => match dim {
            0 => {
                x.data.borrow_mut().add_grad(TensorValue::Vector1D(grad[0].clone()));
            },
            1 => {
                let grad_ = transpose(grad);
                x.data.borrow_mut().add_grad(TensorValue::Vector1D(grad_[0].clone()));
            },
            _ => panic!("Invalid unsqueeze gradient combination"),
        }
        _ => panic!("Invalid unsqueeze gradient combination"),
    }

}

#[test]
fn unsqueeze_works(){
    let inputs_scalar = Tensor::scalar(1.0);
    let inputs_vector = Tensor::vector(vec![1.0, 1.0]);

    let hidden_scalar = inputs_scalar.unsqueeze(0);
    let hidden_vector = inputs_vector.unsqueeze(1);

    let output_scalar = hidden_scalar.sum();
    let output_vector = hidden_vector.sum();

    output_scalar.backward();
    output_vector.backward();

    let scalar_grad = match &inputs_scalar.data.borrow().grad {
        TensorValue::Scalar(s) => *s,
        _ => panic!("Error in scalar")
    };
    let vector_grad = match &inputs_vector.data.borrow().grad {
        TensorValue::Vector1D(v) => v.clone(),
        _ => panic!("Error in scalar")
    };
    assert_eq!(scalar_grad, 1.0);
    assert_eq!(vector_grad, vec![1.0, 1.0])
}











