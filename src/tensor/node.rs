use crate::tensor::operation::Operation;
use crate::tensor::Tensor;
use crate::tensor::value::TensorValue;

#[derive(Clone, Debug)]
pub struct NodeData {
    pub value: TensorValue,
    pub grad: TensorValue,
    pub operation: Operation,
    pub dependencies: Vec<Tensor>,
}

impl NodeData {
    pub fn add_grad_scalar(&mut self, delta: f64) {
        match &mut self.grad {
            TensorValue::Scalar(v) => *v += delta,
            TensorValue::Vector1D(vec) => vec.iter_mut().for_each(|x| *x += delta),
            TensorValue::Matrix2D(mat) => mat.iter_mut()
                .for_each(|row| row.iter_mut().for_each(|x| *x += delta)),
            TensorValue::Tensor3D(_) => todo!()
        }
    }

    pub fn add_grad(&mut self, delta: TensorValue) {
        match (&mut self.grad, &delta) {
            (TensorValue::Scalar(a), TensorValue::Scalar(b)) => *a += b,
            (TensorValue::Vector1D(a), TensorValue::Vector1D(b)) => {
                a.iter_mut().zip(b).for_each(|(a, b)| *a += b)
            }
            (TensorValue::Matrix2D(a), TensorValue::Matrix2D(b)) => {
                a.iter_mut().zip(b).for_each(|(a_row, b_row)| {
                    a_row.iter_mut().zip(b_row).for_each(|(a, b)| *a += b)
                })
            }

            // 梯度广播
            (TensorValue::Vector1D(a), TensorValue::Scalar(b)) => {
                a.iter_mut().for_each(|a| *a += b)
            }
            (TensorValue::Matrix2D(a), TensorValue::Vector1D(b)) => {
                a.iter_mut().for_each(|a_row| {
                    a_row.iter_mut().zip(b.clone()).for_each(|(a, b_val)| *a += b_val)
                })
            }
            (TensorValue::Matrix2D(a), TensorValue::Scalar(b)) => {
                a.iter_mut().for_each(|a_row| {
                    a_row.iter_mut().for_each(|a| *a += b)
                })
            }

            _ => {
                println!("{:?}", self.grad);
                println!("{:?}", delta);
                panic!("Gradient shape mismatch")
            },
        }
    }
}