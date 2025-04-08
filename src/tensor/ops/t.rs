use crate::tensor::operation::Operation;
use crate::tensor::utils::transpose;
use super::super::{Tensor, TensorValue};

impl Tensor {
    pub fn t(&self) -> Tensor {
        let data = self.data.borrow();
        let result_value = match &data.value {
            TensorValue::Matrix2D(m) => {
                TensorValue::Matrix2D(
                    transpose(m)
                )
            }

            _ => panic!("t function only can be used in 2D Matrix!")
        };
        let result = Self::from_value(result_value);
        {
            let mut res_data = result.data.borrow_mut();
            res_data.operation = Operation::T;
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

    let grad = match &data.grad {
        TensorValue::Matrix2D(m) => TensorValue::Matrix2D(transpose(m)),
        _ => panic!("T backward only for 2D matrices")
    };
    x.data.borrow_mut().add_grad(grad);
}

#[test]
fn t_backward_work(){
    let inputs = Tensor::matrix(vec![
        vec![1.0, 1.0],
        vec![1.0, 1.0],
        vec![1.0, 1.0],
    ]);
    {
        inputs.data.borrow_mut().grad = TensorValue::Matrix2D(vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ]);
    }
    let hidden = inputs.t();
    let output = hidden.sum();
    output.backward();
    let grad = match &hidden.data.borrow().grad {
        TensorValue::Matrix2D(m) => m.clone(),
        _ => panic!("Error in shape")
    };

    assert_eq!(grad, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]])
}