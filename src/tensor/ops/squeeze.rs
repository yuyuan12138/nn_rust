use crate::tensor::operation::Operation;
use crate::tensor::utils::transpose;
use super::super::{Tensor, TensorValue};
use anyhow::Result;
impl Tensor {
    pub fn squeeze(&self, dim: usize) -> Result<Tensor> {
        let data = self.data.borrow();

        let result_value = match &data.value {
            TensorValue::Vector1D(v) => match dim {
                0 => {
                    TensorValue::Scalar(v[0])
                }

                _ => panic!("Beyond dim")
            }

            TensorValue::Matrix2D(m) => match dim {
                0 => {
                    if m.len() > 1 {
                        panic!("Error in squeeze {}. For the dim > 1", dim);
                    }
                    TensorValue::Vector1D(m[0].clone())
                }
                1 => {
                    if m[0].len() > 1 {
                        panic!("Error in squeeze {}. For the dim > 1", dim);
                    }
                    let result_ = transpose(m);
                    TensorValue::Vector1D(result_[0].clone())
                }
                _ => panic!("Beyond dim")
            }

            _ => panic!("Unsqueeze operation did not match this data format!")
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Squeeze(dim);
            res_data.dependencies = vec![self.clone()];
        }
        Ok(result)
    }

}

pub fn backward(tensor: &Tensor, dim: usize) -> Result<()>{
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;
    if dependencies.len() != 1 {
        panic!("Squeeze operation requires exactly 1 dependency");
    }

    let x = &dependencies[0];


    let squeeze_output = match &data.value {
        TensorValue::Scalar(s) => TensorValue::Scalar(s.clone()),
        TensorValue::Vector1D(v) => TensorValue::Vector1D(v.clone()),
        _ => panic!("Backward unsqueeze doesn't match")
    };

    match (&data.grad, &squeeze_output) {
        (TensorValue::Scalar(grad), TensorValue::Scalar(s)) => {
            x.data.borrow_mut().add_grad(TensorValue::Scalar(grad.clone()))?;
        }
        (TensorValue::Vector1D(grad), TensorValue::Vector1D(s)) => match dim {
            0 => {
                x.data.borrow_mut().add_grad(TensorValue::Matrix2D(vec![grad.clone()]))?
            }

            1 => {
                x.data.borrow_mut().add_grad(TensorValue::Matrix2D(transpose(&vec![grad.clone()])))?
            }
            _ => panic!("Backward unsqueeze doesn't match")
        }
        _ => panic!("Invalid unsqueeze gradient combination"),
    }

    Ok(())

}

#[test]
fn squeeze_works() -> Result<()>{
    let inputs_vector = Tensor::vector(vec![1.0]);
    let inputs_matrix = Tensor::matrix(vec![
        vec![1.0, 1.0]
    ]);

    let inputs_matrix_2 = Tensor::matrix(vec![
        vec![1.0],
        vec![1.0]
    ]);

    let hidden_vector = inputs_vector.squeeze(0)?;
    let hidden_matrix = inputs_matrix.squeeze(0)?;
    let hidden_matrix_2 = inputs_matrix_2.squeeze(1)?;

    let output_vector = hidden_vector.sum()?;
    let output_matrix = hidden_matrix.sum()?;
    let output_matrix_2 = hidden_matrix_2.sum()?;

    output_vector.backward()?;
    output_matrix.backward()?;
    output_matrix_2.backward()?;

    let vector_grad = match &inputs_vector.data.borrow().grad {
        TensorValue::Vector1D(v) => v.clone(),
        _ => panic!("Error in scalar")
    };
    let matrix_grad = match &inputs_matrix.data.borrow().grad {
        TensorValue::Matrix2D(m) => m.clone(),
        _ => panic!("Error in scalar")
    };
    let matrix_grad_2 = match &inputs_matrix_2.data.borrow().grad {
        TensorValue::Matrix2D(m) => m.clone(),
        _ => panic!("Error in scalar")
    };
    assert_eq!(vector_grad, vec![1.0]);
    assert_eq!(matrix_grad, vec![vec![1.0, 1.0]]);
    assert_eq!(matrix_grad_2, vec![
        vec![1.0],
        vec![1.0],
    ]);
    Ok(())
}











