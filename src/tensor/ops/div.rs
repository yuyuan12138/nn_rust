use crate::tensor::operation::Operation;
use super::super::{Tensor, TensorValue};

impl Tensor {
    pub fn div(&self, other: &Tensor) -> Tensor {
        let a = self.data.borrow();
        let b = other.data.borrow();

        let result_value = match (&a.value, &b.value) {
            // 标量相除
            (TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
                TensorValue::Scalar(a_val / b_val)
            }
            // 向量相除
            (TensorValue::Vector1D(a_vec), TensorValue::Vector1D(b_vec)) => {
                assert_eq!(a_vec.len(), b_vec.len(), "Vector length mismatch");
                TensorValue::Vector1D(
                    a_vec.iter().zip(b_vec).map(|(a, b)| a / b).collect()
                )
            }
            // 矩阵相除（逐元素）
            (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
                assert_eq!(a_mat.len(), b_mat.len(), "Matrix rows mismatch");
                assert_eq!(a_mat[0].len(), b_mat[0].len(), "Matrix cols mismatch");
                TensorValue::Matrix2D(
                    a_mat.iter().zip(b_mat).map(|(a_row, b_row)| {
                        a_row.iter().zip(b_row).map(|(a, b)| a / b).collect()
                    }).collect()
                )
            }
            // 标量广播
            (TensorValue::Scalar(s), TensorValue::Vector1D(v)) => {
                TensorValue::Vector1D(v.iter().map(|x| s / x).collect())
            }
            (TensorValue::Vector1D(v), TensorValue::Scalar(s)) => {
                TensorValue::Vector1D(v.iter().map(|x| x / s).collect())
            }
            (TensorValue::Scalar(s), TensorValue::Matrix2D(m)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| s / x).collect()).collect()
                )
            }
            (TensorValue::Matrix2D(m), TensorValue::Scalar(s)) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| row.iter().map(|x| x / s).collect()).collect()
                )
            }
            _ => panic!("Invalid div operation between types"),
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Div;
            res_data.dependencies = vec![self.clone(), other.clone()];
        }
        result
    }
}

pub fn backward(tensor: &Tensor){
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;
    if dependencies.len() != 2 {
        panic!("Div operation requires exactly 2 dependencies");
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
            let b_sq = b_val.powi(2);
            a.data.borrow_mut().add_grad_scalar(grad / b_val);
            b.data.borrow_mut().add_grad_scalar(-grad * a_val / b_sq);
        }
        // TODO: 其他类型处理类似，需要实现对应的梯度计算
        _ => unimplemented!("Div backward for non-scalar not implemented"),
    }
}