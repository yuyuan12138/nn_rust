pub mod value;
pub mod autodiff;
pub mod node;
pub mod operation;
pub mod utils;
pub mod ops;

use std::cell::RefCell;
use std::rc::Rc;
use crate::tensor::value::TensorValue;
use crate::tensor::node::NodeData;
use crate::tensor::operation::Operation;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Rc<RefCell<NodeData>>,
}

impl Tensor {
    pub fn detach(&self) -> Self {
        Tensor {
            data: Rc::new(RefCell::new(NodeData {
                value: self.data.borrow().value.clone(),
                grad: match &self.data.borrow().value {
                    TensorValue::Scalar(_) => TensorValue::Scalar(0.0),
                    TensorValue::Vector1D(v) => TensorValue::Vector1D(vec![0.0; v.len()]),
                    TensorValue::Matrix2D(m) => TensorValue::Matrix2D(
                        vec![vec![0.0; m[0].len()]; m.len()]
                    ),
                    TensorValue::Tensor3D(_) => todo!()
                },
                operation: Operation::None,
                dependencies: vec![]
            }))
        }
    }

    fn from_value(value: TensorValue) -> Self {
        let grad = match &value {
            TensorValue::Scalar(_) => TensorValue::Scalar(0.0),
            TensorValue::Vector1D(v) => TensorValue::Vector1D(vec![0.0; v.len()]),
            TensorValue::Matrix2D(m) => {
                TensorValue::Matrix2D(vec![vec![0.0; m[0].len()]; m.len()])
            }
            TensorValue::Tensor3D(_) => todo!()
        };

        Tensor {
            data: Rc::new(RefCell::new(NodeData {
                value,
                grad,
                operation: Operation::None,
                dependencies: vec![],
            })),
        }
    }

    pub fn scalar(value: f64) -> Self {
        Self::from_value(TensorValue::Scalar(value))
    }

    pub fn vector(data: Vec<f64>) -> Self {
        Self::from_value(TensorValue::Vector1D(data))
    }

    pub fn matrix(data: Vec<Vec<f64>>) -> Self {
        Self::from_value(TensorValue::Matrix2D(data))
    }


}