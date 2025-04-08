use crate::tensor::operation::Operation;
use super::super::{Tensor, TensorValue};

impl Tensor {
    pub fn tanh(&self) -> Tensor {
        let data = self.data.borrow();
        let result_value = match &data.value {
            TensorValue::Scalar(v) => {
                let s = (v.exp() - (-v).exp()) / (v.exp() + (-v).exp());
                TensorValue::Scalar(s)
            }
            TensorValue::Vector1D(v) => {
                TensorValue::Vector1D(
                    v.iter().map(|x| (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())).collect()
                )
            }
            TensorValue::Matrix2D(m) => {
                TensorValue::Matrix2D(
                    m.iter().map(|row| {
                        row.iter().map(|x| (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())).collect()
                    }).collect()
                )
            }
            TensorValue::Tensor3D(_) => todo!()
        };
        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::Tanh;
            res_data.dependencies = vec![self.clone()];
        }

        result
    }
}

pub fn backward(tensor: &Tensor){
    let data = tensor.data.borrow();
    let dependencies = &data.dependencies;
    if dependencies.len() != 1 {
        panic!("Tanh operation requires exactly 1 dependency");
    }

    let x = &dependencies[0];
    let tanh_output = match &data.value {
        TensorValue::Scalar(s) => TensorValue::Scalar(*s),
        TensorValue::Vector1D(v) => TensorValue::Vector1D(v.clone()),
        TensorValue::Matrix2D(m) => TensorValue::Matrix2D(m.clone()),
        TensorValue::Tensor3D(_) => todo!()
    };

    match (&data.grad, &tanh_output) {
        (TensorValue::Scalar(grad), TensorValue::Scalar(s)) => {
            let grad_x = grad * (1.0 - s.powf(2.0));
            x.data.borrow_mut().add_grad_scalar(grad_x);
        }
        (TensorValue::Vector1D(grad), TensorValue::Vector1D(s)) => {
            let grad_x: Vec<_> = grad.iter()
                .zip(s)
                .map(|(g, s_)| g * (1.0 - s_.powf(2.0)))
                .collect();
            x.data.borrow_mut().add_grad(TensorValue::Vector1D(grad_x));
        }
        (TensorValue::Matrix2D(grad), TensorValue::Matrix2D(s)) => {
            let grad_x: Vec<Vec<_>> = grad.iter()
                .zip(s)
                .map(|(g_row, s_row)| {
                    g_row.iter()
                        .zip(s_row)
                        .map(|(g, s_)| g * (1.0 - s_.powf(2.0)))
                        .collect()
                })
                .collect();
            x.data.borrow_mut().add_grad(TensorValue::Matrix2D(grad_x));
        }

        _ => panic!("Invalid tanh gradient combination"),
    }
}