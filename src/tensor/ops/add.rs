use crate::tensor::operation::Operation;
use super::super::{Tensor, TensorValue};

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Tensor {
        let a = self.data.borrow();
        let b = other.data.borrow();

        let result_value = match (&a.value, &b.value) {
            // 标量相加
            (TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
                TensorValue::Scalar(a_val + b_val)
            }
            // 向量相加
            (TensorValue::Vector1D(a_vec), TensorValue::Vector1D(b_vec)) => {
                assert_eq!(a_vec.len(), b_vec.len(), "Vector length mismatch");
                TensorValue::Vector1D(
                    a_vec.iter().zip(b_vec).map(|(a, b)| a + b).collect()
                )
            }
            // 矩阵相加
            (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
                assert_eq!(a_mat.len(), b_mat.len(), "Matrix rows mismatch");
                assert_eq!(a_mat[0].len(), b_mat[0].len(), "Matrix cols mismatch");
                TensorValue::Matrix2D(
                    a_mat.iter().zip(b_mat).map(|(a_row, b_row)| {
                        a_row.iter().zip(b_row).map(|(a, b)| a + b).collect()
                    }).collect()
                )
            }

            // (TensorValue::Scalar(s), TensorValue::Vector1D(v)) => {
            //     TensorValue::Vector1D(v.iter().map(|x| x + s).collect())
            // }
            (TensorValue::Vector1D(v), TensorValue::Scalar(s)) => {
                TensorValue::Vector1D(v.iter().map(|x| x + s).collect())
            }
            // (TensorValue::Scalar(s), TensorValue::Matrix2D(m)) => {
            //     TensorValue::Matrix2D(
            //         m.iter().map(|row| row.iter().map(|x| x + s).collect()).collect()
            //     )
            // }
            (TensorValue::Matrix2D(m), TensorValue::Scalar(s)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| x + s).collect()).collect()
                )
            }

            _ => panic!("Invalid add operation between types"),
        };
        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Add;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        result
    }
}

pub fn backward(tensor: &Tensor){
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;

    if dependencies.len() != 2 {
        panic!("Add operation requires exactly 2 dependencies");
    }
    let a = &dependencies[0];
    let b = &dependencies[1];

    match &data.grad {
        TensorValue::Scalar(grad) => {
            a.data.borrow_mut().add_grad_scalar(*grad);
            b.data.borrow_mut().add_grad_scalar(*grad);
        }
        TensorValue::Vector1D(grad_vec) => {
            a.data.borrow_mut().add_grad(TensorValue::Vector1D(grad_vec.clone()));
            b.data.borrow_mut().add_grad(TensorValue::Vector1D(grad_vec.clone()));
        }
        TensorValue::Matrix2D(grad_mat) => {
            a.data.borrow_mut().add_grad(TensorValue::Matrix2D(grad_mat.clone()));
            b.data.borrow_mut().add_grad(TensorValue::Matrix2D(grad_mat.clone()));
        }
        TensorValue::Tensor3D(_) => todo!()
    }
}

#[test]
fn add_works(){
    let a = Tensor::scalar(2.0);
    let b = Tensor::scalar(3.0);
    let c = a.add(&b);
    c.backward();
    let a_grad = match &a.data.borrow().grad {
        TensorValue::Scalar(s) => *s,
        _ => panic!("Error!")
    };
    let b_grad = match &b.data.borrow().grad {
        TensorValue::Scalar(s) => *s,
        _ => panic!("Error!")
    };
    assert_eq!(a_grad, 1.0);
    assert_eq!(b_grad, 1.0);
}