use anyhow::anyhow;
use crate::tensor::operation::Operation;
use super::super::{Tensor, TensorValue};
use anyhow::Result;

impl Tensor {
    pub fn multiply(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.data.borrow();
        let b = other.data.borrow();

        let result_value = match (&a.value, &b.value) {

            (TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
                TensorValue::Scalar(a_val * b_val)
            }

            (TensorValue::Vector1D(a_vec), TensorValue::Vector1D(b_vec)) => {
                assert_eq!(a_vec.len(), b_vec.len(), "Vector length mismatch");
                TensorValue::Vector1D(
                    a_vec.iter().zip(b_vec).map(|(a, b)| a * b).collect()
                )
            }

            (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
                assert_eq!(a_mat.len(), b_mat.len(), "Matrix rows mismatch");
                assert_eq!(a_mat[0].len(), b_mat[0].len(), "Matrix cols mismatch");
                TensorValue::Matrix2D(
                    a_mat.iter().zip(b_mat).map(|(a_row, b_row)| {
                        a_row.iter().zip(b_row).map(|(a, b)| a * b).collect()
                    }).collect()
                )
            }

            (TensorValue::Scalar(s), TensorValue::Vector1D(v)) => {
                TensorValue::Vector1D(v.iter().map(|x| s * x).collect())
            }
            (TensorValue::Vector1D(v), TensorValue::Scalar(s)) => {
                TensorValue::Vector1D(v.iter().map(|x| x * s).collect())
            }
            (TensorValue::Scalar(s), TensorValue::Matrix2D(m)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| s * x).collect()).collect()
                )
            }
            (TensorValue::Matrix2D(m), TensorValue::Scalar(s)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| x * s).collect()).collect()
                )
            }
            _ => panic!("Invalid multiply operation between types"),
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Multiply;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        Ok(result)
    }
}

pub fn backward(tensor: &Tensor) -> Result<()>{
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;
    if dependencies.len() != 2 {
        panic!("Multiply operation requires exactly 2 dependencies");
    }

    let a = &dependencies[0];
    let b = &dependencies[1];

    let (a_val, b_val) = {
        let a_data = a.data.borrow();
        let b_data = b.data.borrow();
        (a_data.value.clone(), b_data.value.clone())
    };

    match (&data.grad, &a_val, &b_val) {
        (TensorValue::Scalar(grad), TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
            a.data.borrow_mut().add_grad_scalar(grad * b_val)?;
            b.data.borrow_mut().add_grad_scalar(grad * a_val)?;
        }
        (TensorValue::Vector1D(grad), TensorValue::Vector1D(a_val), TensorValue::Vector1D(b_val)) => {
            let a_grad: Vec<_> = grad.iter().zip(b_val).map(|(g, b)| g * b).collect();
            let b_grad: Vec<_> = grad.iter().zip(a_val).map(|(g, a)| g * a).collect();
            a.data.borrow_mut().add_grad(TensorValue::Vector1D(a_grad))?;
            b.data.borrow_mut().add_grad(TensorValue::Vector1D(b_grad))?;
        }
        (TensorValue::Matrix2D(grad), TensorValue::Matrix2D(a_val), TensorValue::Matrix2D(b_val)) => {
            let mut a_grad = vec![vec![0.0; a_val[0].len()]; a_val.len()];
            let mut b_grad = vec![vec![0.0; b_val[0].len()]; b_val.len()];

            for i in 0..grad.len() {
                for j in 0..grad[0].len() {
                    a_grad[i][j] += grad[i][j] * b_val[i][j];
                    b_grad[i][j] += grad[i][j] * a_val[i][j];
                }
            }

            a.data.borrow_mut().add_grad(TensorValue::Matrix2D(a_grad))?;
            b.data.borrow_mut().add_grad(TensorValue::Matrix2D(b_grad))?;
        }
        _ => panic!("Unsupported multiply backward combination"),
    }
    Ok(())
}