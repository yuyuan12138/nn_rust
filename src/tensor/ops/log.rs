use crate::tensor::operation::Operation;
use super::super::{Tensor, TensorValue};
use anyhow::Result;

impl Tensor {
    pub fn log(&self, value: f64) -> Result<Tensor> {
        let a = self.data.borrow();

        let result_value = match &a.value {
            TensorValue::Scalar(v) => {
                let s = v.log(value);
                TensorValue::Scalar(s)
            }
            TensorValue::Vector1D(v) => {
                TensorValue::Vector1D(
                    v.iter().map(|x| x.log(value)).collect()
                )
            }
            TensorValue::Matrix2D(m) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| {
                        row.iter().map(|x| x.log(value)).collect()
                    }).collect()
                )
            }
            TensorValue::Tensor3D(_) => todo!()
        };
        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Log(value);
            res_data.dependencies = vec![self.clone()];
        }

        Ok(result)
    }
}

pub fn backward(tensor: &Tensor, base: f64) -> Result<()>{
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;
    if dependencies.len() != 1 {
        panic!("Log operation requires exactly 1 dependency");
    }
    let x = &dependencies[0];

    let x_val = {
        let x_data = x.data.borrow();
        x_data.value.clone()
    };

    match (&data.grad, &x_val) {
        (TensorValue::Scalar(grad), TensorValue::Scalar(x_val)) => {
            let dx = if x_val.abs() < 1e-12 {
                0.0  // 处理log(0)的梯度爆炸
            } else {
                grad / (x_val * base.ln())
            };
            x.data.borrow_mut().add_grad_scalar(dx);
        }
        (TensorValue::Vector1D(grad), TensorValue::Vector1D(x_vec)) => {
            assert_eq!(x_vec.len(), grad.len(), "Vector length mismatch in Log backward");

            let dx_vec: Vec<f64> = x_vec.iter().zip(grad)
                .map(|(x_, g)| {
                    if x_.abs() < 1e-12 {
                        0.0
                    } else {
                        g / x_ * base.ln()
                    }
                })
                .collect();

            x.data.borrow_mut().add_grad(TensorValue::Vector1D(dx_vec));
        }
        (TensorValue::Matrix2D(grad), TensorValue::Matrix2D(x_mat)) => {
            let dx_mat: Vec<Vec<f64>> = x_mat.iter()
                .zip(grad)
                .map(|(x_row, g_row)| {
                    x_row.iter()
                        .zip(g_row)
                        .map(|(x_, g)| {
                            if x_.abs() < 1e-12 {
                                0.0
                            } else {
                                g / x_ * base.ln()
                            }
                        })
                        .collect()
                })
                .collect();

            x.data.borrow_mut().add_grad(TensorValue::Matrix2D(dx_mat));
        }
        _ => panic!("Mismatched tensor types in Log backward"),
    }

    Ok(())
}