use crate::tensor::Tensor;
use crate::tensor::operation::Operation;
use crate::tensor::utils::{matrix_multiply, transpose};
use crate::tensor::value::TensorValue;

pub mod add;
pub mod div;
pub mod log;
pub mod matmul;
pub mod mean;
pub mod multiply;
pub mod pow;
pub mod relu;
pub mod sigmoid;
pub mod sub;

pub fn _backward(tensor: &Tensor){
    let data = tensor.data.borrow();
    match data.operation {
        Operation::Add => add::backward(&tensor),
        Operation::Sub => sub::backward(&tensor),
        Operation::Multiply => multiply::backward(&tensor),
        Operation::Div => div::backward(&tensor),
        Operation::Sigmoid => sigmoid::backward(&tensor),
        Operation::ReLU => relu::backward(&tensor),
        Operation::Matmul => matmul::backward(&tensor),
        Operation::Mean => mean::backward(&tensor),

        Operation::Log(base) => log::backward(&tensor, base),

        Operation::Pow(exponent) => pow::backward(&tensor, exponent),
        // 其他操作...
        _ => {}
    }
}