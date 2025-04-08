use crate::tensor::operation::Operation;
use super::super::{Tensor, TensorValue};

impl Tensor {
    pub fn pow(&self, value: f64) -> Tensor {
        let a = self.data.borrow();

        let result_value = match &a.value {
            TensorValue::Scalar(v) => {
                let s = v.powf(value);
                TensorValue::Scalar(s)
            }
            TensorValue::Vector1D(v) => {
                TensorValue::Vector1D(
                    v.iter().map(|x| x.powf(value)).collect()
                )
            }
            TensorValue::Matrix2D(m) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| {
                        row.iter().map(|x| x.powf(value)).collect()
                    }).collect()
                )
            }
            TensorValue::Tensor3D(_) => todo!()
        };

        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Pow(value);
            res_data.dependencies = vec![self.clone()];
        }

        result
    }
}

pub fn backward(tensor: &Tensor, exponent: f64){
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;
    if dependencies.len() != 1 {
        panic!("Pow operation requires exactly 1 dependency");
    }

    let x = &dependencies[0];
    let x_val = {
        let x_data = x.data.borrow();
        x_data.value.clone()
    };

    // value * x
    match (&data.grad, &x_val) {
        (TensorValue::Scalar(grad), TensorValue::Scalar(s)) => {
            let dx = if s.abs() < 1e-12 && exponent < 1.0 {
                0.0
            } else {
                grad * exponent * s.powf(exponent - 1.0)
            };
            x.data.borrow_mut().add_grad_scalar(dx);
        }
        (TensorValue::Vector1D(grad), TensorValue::Vector1D(s)) => {
            assert_eq!(s.len(), grad.len(), "Vector length mismatch in Pow backward");

            let dx_vec: Vec<_> = s.iter().zip(grad).map(|(x_, g_)| {
                if x_.abs() < 1e-12 && exponent < 1.0 {
                    0.0
                }else {
                    g_ * exponent * x_.powf(exponent - 1.0)
                }
            }).collect();

            x.data.borrow_mut().add_grad(TensorValue::Vector1D(dx_vec))
        }
        (TensorValue::Matrix2D(grad), TensorValue::Matrix2D(s)) => {
            let dx_mat = s.iter()
                .zip(grad)
                .map(|(x_row, g_row)| {
                    x_row.iter()
                        .zip(g_row)
                        .map(|(x_, g_)| {
                            if x_.abs() < 1e-12 && exponent < 1.0 {
                                0.0
                            }else {
                                g_ * exponent * x_.powf(exponent - 1.0)
                            }
                        }).collect()
                }).collect();
            x.data.borrow_mut().add_grad(TensorValue::Matrix2D(dx_mat))
        }
        _ => panic!("Invalid sigmoid gradient combination"),
    }
}