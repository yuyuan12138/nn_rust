use crate::tensor::operation::Operation;
use super::super::{Tensor, TensorValue};
use anyhow::Result;

impl Tensor {
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.data.borrow();
        let b = other.data.borrow();

        let result_value = match (&a.value, &b.value) {
            // 标量相减
            (TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
                TensorValue::Scalar(a_val - b_val)
            }
            // 向量相减
            (TensorValue::Vector1D(a_vec), TensorValue::Vector1D(b_vec)) => {
                assert_eq!(a_vec.len(), b_vec.len(), "Vector length mismatch");
                TensorValue::Vector1D(
                    a_vec.iter().zip(b_vec).map(|(a, b)| a - b).collect()
                )
            }
            // 矩阵相减
            (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
                assert_eq!(a_mat.len(), b_mat.len(), "Matrix rows mismatch");
                assert_eq!(a_mat[0].len(), b_mat[0].len(), "Matrix cols mismatch");
                TensorValue::Matrix2D(
                    a_mat.iter().zip(b_mat).map(|(a_row, b_row)| {
                        a_row.iter().zip(b_row).map(|(a, b)| a - b).collect()
                    }).collect()
                )
            }
            // 标量广播
            (TensorValue::Scalar(s), TensorValue::Vector1D(v)) => {
                TensorValue::Vector1D(v.iter().map(|x| s - x).collect())
            }
            (TensorValue::Vector1D(v), TensorValue::Scalar(s)) => {
                TensorValue::Vector1D(v.iter().map(|x| x - s).collect())
            }
            (TensorValue::Scalar(s), TensorValue::Matrix2D(m)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| s - x).collect()).collect()
                )
            }
            (TensorValue::Matrix2D(m), TensorValue::Scalar(s)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| x - s).collect()).collect()
                )
            }

            // 向量广播
            (TensorValue::Matrix2D(m), TensorValue::Vector1D(v)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().zip(v).map(|(a, b)| a - b).collect()).collect()
                )
            }
            _ => panic!("Invalid sub operation between types"),
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Sub;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        Ok(result)
    }
}

pub fn backward(tensor: &Tensor) -> Result<()>{
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;
    if dependencies.len() != 2 {
        panic!("Sub operation requires exactly 2 dependencies");
    }

    let a = &dependencies[0];
    let b = &dependencies[1];
    match &data.grad {
        TensorValue::Scalar(grad) => {
            a.data.borrow_mut().add_grad_scalar(*grad)?;
            b.data.borrow_mut().add_grad_scalar(-*grad)?;
        }
        TensorValue::Vector1D(grad_vec) => {
            a.data.borrow_mut().add_grad(TensorValue::Vector1D(grad_vec.clone()))?;
            b.data.borrow_mut().add_grad(TensorValue::Vector1D(
                grad_vec.iter().map(|g| -g).collect()
            ))?;
        }
        TensorValue::Matrix2D(grad_mat) => {
            a.data.borrow_mut().add_grad(TensorValue::Matrix2D(grad_mat.clone()))?;
            b.data.borrow_mut().add_grad(TensorValue::Matrix2D(
                grad_mat.iter().map(|row|
                    row.iter().map(|g| -g).collect()
                ).collect()
            ))?;
        }
        TensorValue::Tensor3D(_) => todo!()
    }
    Ok(())
}