use crate::tensor::operation::Operation;
use super::super::{Tensor, TensorValue};
use anyhow::Result;
use itertools::izip;

impl Tensor {
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        let a = self.data.borrow();
        let b = other.data.borrow();

        let result_value = match (&a.value, &b.value) {

            (TensorValue::Scalar(a_val), TensorValue::Scalar(b_val)) => {
                TensorValue::Scalar(a_val / b_val)
            }

            (TensorValue::Vector1D(a_vec), TensorValue::Vector1D(b_vec)) => {
                assert_eq!(a_vec.len(), b_vec.len(), "Vector length mismatch");
                TensorValue::Vector1D(
                    a_vec.iter().zip(b_vec).map(|(a, b)| a / b).collect()
                )
            }

            (TensorValue::Matrix2D(a_mat), TensorValue::Matrix2D(b_mat)) => {
                assert_eq!(a_mat.len(), b_mat.len(), "Matrix rows mismatch");
                assert_eq!(a_mat[0].len(), b_mat[0].len(), "Matrix cols mismatch");
                TensorValue::Matrix2D(
                    a_mat.iter().zip(b_mat).map(|(a_row, b_row)| {
                        a_row.iter().zip(b_row).map(|(a, b)| a / b).collect()
                    }).collect()
                )
            }

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
        Ok(result)
    }
}

pub fn backward(tensor: &Tensor) -> Result<()>{
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
            a.data.borrow_mut().add_grad_scalar(grad / b_val)?;
            b.data.borrow_mut().add_grad_scalar(-grad * a_val / b_sq)?;
        },
        (TensorValue::Vector1D(v_grad), TensorValue::Vector1D(v_a_val), TensorValue::Vector1D(v_b_val)) => {
            let a_grad = v_grad.iter().zip(v_b_val).map(|(g, b_val_)| g / b_val_).collect();
            let a_grad = TensorValue::Vector1D(a_grad);


            // Many Thanks to bluss. Cite: https://stackoverflow.com/questions/29669287/how-can-i-zip-more-than-two-iterators
            // Where I found how to zip more than two iterators in rust.
            let mut b_grad: Vec<f64> = vec![];
            for (a_val_, b_val_, g) in izip!(v_a_val, v_b_val, v_grad) {
                b_grad.push(-g * a_val_ / b_val_.powi(2));
            }
            let b_grad = TensorValue::Vector1D(b_grad);

            a.data.borrow_mut().add_grad(a_grad)?;
            b.data.borrow_mut().add_grad(b_grad)?;
        }

        (TensorValue::Matrix2D(m_grad), TensorValue::Matrix2D(m_a_val), TensorValue::Matrix2D(m_b_val)) => {
            let a_grad = m_grad.iter().zip(m_b_val).map(|(v_grad, v_b_val)|
                v_grad.iter().zip(v_b_val).map(|(g, b_val_)| g / b_val_).collect()
            ).collect();

            let mut b_grad: Vec<Vec<f64>> = vec![];

            for(v_a_val, v_b_val, v_grad) in izip!(m_a_val, m_b_val, m_grad) {
                let mut b_grad_row: Vec<f64> = vec![];
                for(a_val_, b_val_, g) in izip!(v_a_val, v_b_val, v_grad) {
                    b_grad_row.push(-g * a_val_ / b_val_.powi(2));
                }
                b_grad.push(b_grad_row);
            }

            let a_grad = TensorValue::Matrix2D(a_grad);
            let b_grad = TensorValue::Matrix2D(b_grad);

            a.data.borrow_mut().add_grad(a_grad)?;
            b.data.borrow_mut().add_grad(b_grad)?;
        }
        // TODO: 其他类型处理类似，需要实现对应的梯度计算
        _ => unimplemented!("Div backward for non-scalar not implemented"),
    };
    Ok(())
}

#[test]
fn test_div_backward() -> Result<()>{
    let a = Tensor::vector(vec![1.0, 2.0, 3.0]);
    let b = Tensor::vector(vec![4.0, 5.0, 6.0]);

    let output = a.div(&b)?.sum()?;
    output.backward()?;

    let a_grad_vec = match &a.data.borrow().grad {
        TensorValue::Vector1D(v) => v.clone(),
        _ => panic!()
    };
    let b_grad_vec = match &b.data.borrow().grad {
        TensorValue::Vector1D(v) => v.clone(),
        _ => panic!()
    };

    let expected_a_grad = vec![
        1.0 / 4.0,      // 1/4 = 0.25
        1.0 / 5.0,      // 1/5 = 0.2
        1.0 / 6.0       // 1/6 ≈ 0.16666667
    ];

    let expected_b_grad = vec![
        -1.0 / 4.0_f64.powi(2),  // -1/16 = -0.0625
        -2.0 / 5.0_f64.powi(2),  // -2/25 = -0.08
        -3.0 / 6.0_f64.powi(2)   // -3/36 ≈ -0.08333333
    ];

    assert_eq!(a_grad_vec, expected_a_grad);
    assert_eq!(b_grad_vec, expected_b_grad);

    Ok(())
}